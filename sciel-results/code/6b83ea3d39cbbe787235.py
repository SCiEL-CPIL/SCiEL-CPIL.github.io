"""Lennard-Jones cluster global minimum, 55 atoms — Mackay icosahedron search."""
N = 55

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, minimize_scalar
import time
import itertools

_DEADLINE = 115.0  # safety margin before 120s kill


def _lj_energy(pos):
    """Compute total Lennard-Jones energy for positions of shape (N,3)."""
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(r2, np.inf)
    r = np.sqrt(r2)
    r6 = 1.0 / (r2 * r2 * r2)
    r12 = r6 * r6
    e = 4.0 * (r12 - r6)
    return 0.5 * np.sum(e)


def _lj_energy_and_grad(pos):
    """Energy and gradient for L-BFGS."""
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(r2, 0.0)
    r6 = 1.0 / (r2 * r2 * r2 + 1e-30)
    r12 = r6 * r6
    # Force on atom i from atom j: 4*(12*r12 - 6*r6) * (ri - rj)/r^2
    # Actually: dU/dr = 4*(-12/r^13 + 6/r^7) = 4*(6/r^7 - 12/r^13)
    # Force = -dU/dr * r_hat = 4*(12/r^13 - 6/r^7) * r_hat
    # In vector form: F_ij = 4*(12*r12 - 6*r6) * (ri-rj) / r2
    factor = 4.0 * (12.0 * r12 - 6.0 * r6) / r2
    np.fill_diagonal(factor, 0.0)
    grad = -np.sum(factor[:, :, np.newaxis] * diff, axis=1)
    r = np.sqrt(r2)
    r6_safe = 1.0 / (r2 * r2 * r2 + 1e-30)
    r12_safe = r6_safe * r6_safe
    e = 4.0 * (r12_safe - r6_safe)
    np.fill_diagonal(e, 0.0)
    return 0.5 * np.sum(e), grad


def _relax(pos, max_iter=5000, tol=1e-10):
    """L-BFGS relaxation with tight tolerance."""
    def fun(x):
        p = x.reshape(-1, 3)
        e, g = _lj_energy_and_grad(p)
        return e, g.reshape(-1)

    res = minimize(fun, pos.ravel(), jac=True, method='L-BFGS-B',
                   options={'maxiter': max_iter, 'gtol': tol, 'ftol': 1e-14})
    return res.x.reshape(-1, 3), res.fun


def _icosahedral_points():
    """Generate the 12 vertices of a regular icosahedron (unit circumradius)."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    pts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    pts = pts / norms  # unit circumradius
    return pts


def _mackay_positions(r1, r2, r3):
    """
    Generate 55-atom Mackay icosahedron positions.
    Shell 1: 12 vertices at radius r1
    Shell 2: 42 points (12 face-centers at r2, 30 edge-midpoints at r3)
    Center: 1 atom at origin
    Returns array of shape (55, 3).
    """
    ico = _icosahedral_points()
    # 12 vertices
    shell1 = ico * r1

    # 12 face centers: normalize sum of 3 adjacent vertices
    # For icosahedron, each face is an equilateral triangle.
    # The 20 faces can be identified by triples of mutually adjacent vertices.
    # A simpler approach: the 20 face normals of a regular icosahedron.
    # The 20 face centers of an icosahedron form a dodecahedron.
    # Dodecahedron vertices (unit circumradius):
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    inv_phi = 1.0 / phi
    # 20 vertices of dodecahedron
    dodeca = np.array([
        [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
        [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
        [0, inv_phi, phi], [0, inv_phi, -phi], [0, -inv_phi, phi], [0, -inv_phi, -phi],
        [inv_phi, phi, 0], [inv_phi, -phi, 0], [-inv_phi, phi, 0], [-inv_phi, -phi, 0],
        [phi, 0, inv_phi], [phi, 0, -inv_phi], [-phi, 0, inv_phi], [-phi, 0, -inv_phi]
    ], dtype=float)
    dodeca_norms = np.linalg.norm(dodeca, axis=1, keepdims=True)
    dodeca = dodeca / dodeca_norms  # unit circumradius
    face_centers = dodeca * r2

    # 30 edge midpoints: midpoint of each edge, normalized to unit radius, then scaled by r3
    # Edges of icosahedron: pairs of vertices with distance = 2*sin(pi/5)*R... 
    # For unit circumradius icosahedron, edge length = 2*R*sin(pi/5) = 2*sin(36°) ≈ 1.1756
    # Actually for unit circumradius, edge length = sqrt(2 - 2*cos(angle)) where angle = 2*pi/5
    # cos(72°) = (sqrt(5)-1)/4 ≈ 0.309
    # edge = sqrt(2 - 2*0.309) = sqrt(1.382) ≈ 1.1756
    edge_len_sq = 2.0 - 2.0 * ((1.0 + np.sqrt(5.0)) / 4.0)  # cos(72°) = (1+sqrt(5))/4
    edge_len_sq = 2.0 - 2.0 * (np.sqrt(5.0) - 1.0) / 4.0
    # Let me just compute all pairs and check distance
    n = len(ico)
    edge_midpoints = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(ico[i] - ico[j])
            if abs(d - np.sqrt(edge_len_sq)) < 1e-8:
                mid = (ico[i] + ico[j]) / 2.0
                mid_norm = np.linalg.norm(mid)
                if mid_norm > 1e-10:
                    mid = mid / mid_norm  # unit radius
                    edge_midpoints.append(mid)
    edge_midpoints = np.array(edge_midpoints)
    edge_centers = edge_midpoints * r3

    # Center atom
    center = np.array([[0.0, 0.0, 0.0]])

    positions = np.vstack([shell1, face_centers, edge_centers, center])
    return positions


def _mackay_energy(params):
    """Energy of Mackay icosahedron with given shell radii (r1, r2, r3)."""
    r1, r2, r3 = params
    if r1 < 0.1 or r2 < 0.1 or r3 < 0.1:
        return 1e10
    pos = _mackay_positions(r1, r2, r3)
    # Check minimum distance
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2_mat = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(r2_mat, np.inf)
    min_r = np.sqrt(np.min(r2_mat))
    if min_r < 0.5:
        return 1e10
    return _lj_energy(pos)


def _apply_rotation(pos, R):
    """Apply rotation matrix R to positions."""
    return pos @ R.T


def _rotation_from_angles(alpha, beta, gamma):
    """Create rotation matrix from Euler angles (ZYZ convention)."""
    ca, sa = np.cos(alpha), np.sin(alpha)
    cb, sb = np.cos(beta), np.sin(beta)
    cg, sg = np.cos(gamma), np.sin(gamma)
    Rz = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]])
    Ry = np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]])
    R = Rz @ Ry @ Rz
    return R


def _mackay_full_energy(params):
    """
    Full parameterized Mackay energy: (r1, r2, r3, alpha, beta, gamma).
    The rotation doesn't change energy (invariant), so we only optimize radii.
    But we keep the interface for potential future use.
    """
    r1, r2, r3 = params[0], params[1], params[2]
    if r1 < 0.1 or r2 < 0.1 or r3 < 0.1:
        return 1e10
    pos = _mackay_positions(r1, r2, r3)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2_mat = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(r2_mat, np.inf)
    min_r = np.sqrt(np.min(r2_mat))
    if min_r < 0.5:
        return 1e10
    return _lj_energy(pos)


def _relax_with_deadline(pos, deadline, max_iter=3000, tol=1e-10):
    """Relax positions with deadline awareness."""
    if time.time() > deadline - 5:
        return pos, _lj_energy(pos)
    
    def fun(x):
        p = x.reshape(-1, 3)
        e, g = _lj_energy_and_grad(p)
        return e, g.reshape(-1)
    
    try:
        res = minimize(fun, pos.ravel(), jac=True, method='L-BFGS-B',
                       options={'maxiter': max_iter, 'gtol': tol, 'ftol': 1e-14})
        return res.x.reshape(-1, 3), res.fun
    except Exception:
        return pos, _lj_energy(pos)


def _validate(pos):
    """Check minimum separation constraint."""
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(r2, np.inf)
    min_r = np.sqrt(np.min(r2))
    return min_r >= 0.5 - 1e-9


def construct():
    t_start = time.time()
    deadline = t_start + _DEADLINE

    best_pos = None
    best_energy = np.inf

    # --- Phase 1: Optimize Mackay icosahedron shell radii via Nelder-Mead ---
    # Initial guess: known LJ minimum at r = 2^(1/6) ≈ 1.1225
    r0 = 2.0 ** (1.0 / 6.0)
    # For Mackay icosahedron, typical ratios: r1 ≈ 1.0, r2 ≈ 1.8, r3 ≈ 1.5 (relative)
    # Let's try several initial guesses
    initial_guesses = [
        [r0, r0 * 1.6, r0 * 1.3],
        [r0, r0 * 1.5, r0 * 1.2],
        [r0, r0 * 1.7, r0 * 1.4],
        [r0 * 0.9, r0 * 1.5, r0 * 1.2],
        [r0 * 1.1, r0 * 1.7, r0 * 1.4],
    ]

    for guess in initial_guesses:
        if time.time() > deadline - 20:
            break
        try:
            res = minimize(_mackay_full_energy, guess, method='Nelder-Mead',
                           options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10})
            if res.fun < best_energy:
                best_energy = res.fun
                r1, r2, r3 = res.x
                if r1 > 0.1 and r2 > 0.1 and r3 > 0.1:
                    best_pos = _mackay_positions(r1, r2, r3)
        except Exception:
            continue

    # --- Phase 2: Also try a more general approach - optimize all 3 radii with tighter NM ---
    if time.time() < deadline - 15:
        # Use the best radii found so far as starting point
        if best_pos is not None:
            # Extract effective radii from best_pos by clustering distances from center
            dists = np.linalg.norm(best_pos, axis=1)
            # Sort and identify clusters
            dists_sorted = np.sort(dists)
            # The Mackay structure has: center (0), 12 at r1, 12 at r2, 30 at r3
            # Just use the Nelder-Mead result directly
            pass

    # --- Phase 3: L-BFGS relaxation from Mackay structure ---
    if best_pos is not None and _validate(best_pos):
        if time.time() < deadline - 10:
            relaxed_pos, relaxed_e = _relax_with_deadline(best_pos, deadline, max_iter=5000, tol=1e-12)
            if _validate(relaxed_pos) and relaxed_e < best_energy:
                best_pos = relaxed_pos
                best_energy = relaxed_e

    # --- Phase 4: Additional relaxation with even tighter tolerance ---
    if best_pos is not None and time.time() < deadline - 5:
        if _validate(best_pos):
            relaxed_pos2, relaxed_e2 = _relax_with_deadline(best_pos, deadline, max_iter=10000, tol=1e-14)
            if _validate(relaxed_pos2) and relaxed_e2 < best_energy:
                best_pos = relaxed_pos2
                best_energy = relaxed_e2

    # --- Fallback: if Mackay approach failed, use a grid search over radii ---
    if best_pos is None or best_energy > -100:
        if time.time() < deadline - 10:
            # Coarse grid search
            r1_range = np.linspace(0.9, 1.3, 9)
            r2_range = np.linspace(1.4, 2.0, 7)
            r3_range = np.linspace(1.1, 1.7, 7)
            grid_best_e = np.inf
            grid_best_r = None
            for r1 in r1_range:
                for r2 in r2_range:
                    for r3 in r3_range:
                        e = _mackay_full_energy([r1, r2, r3])
                        if e < grid_best_e:
                            grid_best_e = e
                            grid_best_r = [r1, r2, r3]
            if grid_best_r is not None:
                # Refine with Nelder-Mead
                try:
                    res = minimize(_mackay_full_energy, grid_best_r, method='Nelder-Mead',
                                   options={'maxiter': 3000, 'xatol': 1e-9, 'fatol': 1e-11})
                    if res.fun < best_energy:
                        best_energy = res.fun
                        best_pos = _mackay_positions(res.x[0], res.x[1], res.x[2])
                except Exception:
                    pass

                # Relax
                if best_pos is not None and _validate(best_pos) and time.time() < deadline - 5:
                    relaxed_pos, relaxed_e = _relax_with_deadline(best_pos, deadline, max_iter=5000, tol=1e-12)
                    if _validate(relaxed_pos) and relaxed_e < best_energy:
                        best_pos = relaxed_pos
                        best_energy = relaxed_e

    # --- Final fallback: seed-like cubic lattice if everything fails ---
    if best_pos is None:
        side = int(np.ceil(N ** (1 / 3)))
        points = [(i, j, k) for i in range(side) for j in range(side) for k in range(side)]
        points.sort(key=lambda p: (p[0] - (side - 1) / 2) ** 2 + (p[1] - (side - 1) / 2) ** 2 + (p[2] - (side - 1) / 2) ** 2)
        best_pos = np.array(points[:N], dtype=float) * 1.12
        best_energy = _lj_energy(best_pos)

    # Final validation
    if not _validate(best_pos):
        # Nudge apart
        side = int(np.ceil(N ** (1 / 3)))
        points = [(i, j, k) for i in range(side) for j in range(side) for k in range(side)]
        points.sort(key=lambda p: (p[0] - (side - 1) / 2) ** 2 + (p[1] - (side - 1) / 2) ** 2 + (p[2] - (side - 1) / 2) ** 2)
        best_pos = np.array(points[:N], dtype=float) * 1.12

    return best_pos
# EVOLVE-BLOCK-END
