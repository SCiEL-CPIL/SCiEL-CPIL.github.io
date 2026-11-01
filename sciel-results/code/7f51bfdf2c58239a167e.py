"""Program for Lennard-Jones 55-atom cluster global minimum search.

Uses a symmetry-constrained manifold approach: parameterizes the 55-atom
Mackay icosahedron by 3 shell radii + 3 rotation parameters (6D manifold),
optimizes with Nelder-Mead, then performs a final high-precision L-BFGS
relaxation in full Cartesian space.
"""

import numpy as np
from scipy.optimize import minimize
import time

N = 55
DEADLINE = 150.0  # seconds, kill limit
SAFETY_MARGIN = 10.0  # seconds to return before deadline


def _get_icosahedron_vectors():
    """
    Returns 12 unit vectors along the 5-fold axes (icosahedron vertices)
    and 30 unit vectors along the 2-fold axes (icosahedron edge midpoints).
    """
    phi = (1 + np.sqrt(5)) / 2

    # 12 vertices of icosahedron
    verts = np.array([
        [0, 1, phi],
        [0, -1, phi],
        [0, 1, -phi],
        [0, -1, -phi],
        [1, phi, 0],
        [-1, phi, 0],
        [1, -phi, 0],
        [-1, -phi, 0],
        [phi, 0, 1],
        [-phi, 0, 1],
        [phi, 0, -1],
        [-phi, 0, -1],
    ], dtype=float)

    # Normalize to unit length
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)

    # 30 edge midpoints
    dists = np.linalg.norm(verts[:, np.newaxis, :] - verts[np.newaxis, :, :], axis=2)
    min_dist = np.min(dists[dists > 1e-8])

    edge_mids = []
    for i in range(12):
        for j in range(i + 1, 12):
            if np.isclose(dists[i, j], min_dist, atol=1e-8):
                mid = (verts[i] + verts[j]) / 2
                mid = mid / np.linalg.norm(mid)
                edge_mids.append(mid)

    edge_mids = np.array(edge_mids)
    return verts, edge_mids


def _build_mackay_template():
    """
    Build the unit-direction template for the 55-atom Mackay icosahedron.
    Returns array of shape (55, 3) where each row is a unit direction vector.
    Atom 0 is at origin (direction [0,0,0] placeholder).
    Atoms 1-12: 5-fold axes (inner shell)
    Atoms 13-42: 2-fold axes (middle shell)
    Atoms 43-54: 5-fold axes (outer shell)
    """
    v5, v2 = _get_icosahedron_vectors()
    # Template: direction for each atom
    template = np.zeros((55, 3))
    template[1:13] = v5       # inner shell directions
    template[13:43] = v2      # middle shell directions
    template[43:55] = v5      # outer shell directions
    return template, v5, v2


def _rotation_matrix_from_params(params):
    """
    Convert 3 rotation parameters (Euler angles ZYX) to a 3x3 rotation matrix.
    """
    alpha, beta, gamma = params
    ca, sa = np.cos(alpha), np.sin(alpha)
    cb, sb = np.cos(beta), np.sin(beta)
    cg, sg = np.cos(gamma), np.sin(gamma)

    # Rz * Ry * Rx
    Rz = np.array([[cg, -sg, 0], [sg, cg, 0], [0, 0, 1]])
    Ry = np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]])
    Rx = np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])

    return Rz @ Ry @ Rx


def _manifold_to_cartesian(params, template):
    """
    Convert 6D manifold parameters (r1, r2, r3, alpha, beta, gamma)
    to 55x3 Cartesian positions.
    """
    r1, r2, r3, alpha, beta, gamma = params
    R = _rotation_matrix_from_params([alpha, beta, gamma])

    pos = np.zeros((55, 3))
    # Atom 0 at origin
    pos[0] = [0, 0, 0]
    # Inner shell: 12 atoms at radius r1 along 5-fold axes
    pos[1:13] = (template[1:13] * r1) @ R.T
    # Middle shell: 30 atoms at radius r2 along 2-fold axes
    pos[13:43] = (template[13:43] * r2) @ R.T
    # Outer shell: 12 atoms at radius r3 along 5-fold axes
    pos[43:55] = (template[43:55] * r3) @ R.T
    return pos


def _lj_energy(pos):
    """Compute total Lennard-Jones energy."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    mask = r2 > 1e-10
    r2_safe = np.where(mask, r2, 1.0)
    r6 = r2_safe ** (-3)
    r12 = r6 ** 2
    energy = 4 * np.sum(np.where(mask, r12 - r6, 0))
    return energy


def _lj_energy_and_grad(pos):
    """Compute total Lennard-Jones energy and gradient."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    mask = r2 > 1e-10
    r2_safe = np.where(mask, r2, 1.0)
    r6 = r2_safe ** (-3)
    r12 = r6 ** 2
    energy = 4 * np.sum(np.where(mask, r12 - r6, 0))

    # Gradient: dE/dx_i = sum_j 4*(-12*r^-14 + 6*r^-8) * (x_i - x_j)/r
    # In terms of r2: grad = 4 * sum_j (-12*r2^-7 + 6*r2^-4) * diff
    r2_inv4 = r2_safe ** (-4)
    r2_inv7 = r2_safe ** (-7)
    coeff = 4 * (-12 * r2_inv7 + 6 * r2_inv4)
    coeff = np.where(mask, coeff, 0)
    grad = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
    return energy, grad


def _is_valid(pos):
    """Check if all inter-atomic distances are >= 0.5."""
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    return np.min(dists) >= 0.5


def _relax(pos, maxiter=10000, ftol=1e-12, gtol=1e-10, deadline=None):
    """Relax positions using L-BFGS-B with analytical gradient."""
    def energy_func(x):
        pos_flat = x.reshape(-1, 3)
        e, g = _lj_energy_and_grad(pos_flat)
        return e, g.flatten()

    if deadline is not None:
        remaining = deadline - time.time()
        if remaining < 1.0:
            return pos

    result = minimize(
        energy_func,
        pos.flatten(),
        jac=True,
        method='L-BFGS-B',
        options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol}
    )
    return result.x.reshape(-1, 3)


def _manifold_energy(params, template):
    """Compute LJ energy on the 6D manifold."""
    pos = _manifold_to_cartesian(params, template)
    return _lj_energy(pos)


def construct():
    """
    Construct the 55-atom LJ cluster using symmetry-constrained manifold
    optimization followed by full Cartesian relaxation.
    """
    start_time = time.time()
    deadline = start_time + DEADLINE - SAFETY_MARGIN

    template, v5, v2 = _build_mackay_template()

    best_energy = float('inf')
    best_pos = None

    # Initial guess for manifold parameters
    # r1, r2, r3 are shell radii; alpha, beta, gamma are rotation angles
    initial_params = np.array([1.135, 1.162, 1.283, 0.0, 0.0, 0.0])

    # Strategy 1: Nelder-Mead optimization on the 6D manifold
    # Multiple starting points with different rotations and radii
    rng = np.random.default_rng(42)

    starting_points = [
        np.array([1.135, 1.162, 1.283, 0.0, 0.0, 0.0]),
        np.array([1.130, 1.160, 1.280, 0.0, 0.0, 0.0]),
        np.array([1.140, 1.165, 1.285, 0.0, 0.0, 0.0]),
        np.array([1.135, 1.162, 1.283, 0.1, 0.0, 0.0]),
        np.array([1.135, 1.162, 1.283, 0.0, 0.1, 0.0]),
        np.array([1.135, 1.162, 1.283, 0.0, 0.0, 0.1]),
    ]

    for sp in starting_points:
        if time.time() > deadline:
            break

        try:
            result = minimize(
                lambda p: _manifold_energy(p, template),
                sp,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            e = result.fun
            if e < best_energy:
                best_energy = e
                best_params = result.x.copy()
        except Exception:
            continue

    # Strategy 2: If manifold optimization succeeded, extract positions and relax
    if best_energy < float('inf') and best_params is not None:
        pos = _manifold_to_cartesian(best_params, template)
        if _is_valid(pos):
            relaxed = _relax(pos, maxiter=5000, ftol=1e-12, gtol=1e-10, deadline=deadline)
            if _is_valid(relaxed):
                e = _lj_energy(relaxed)
                if e < best_energy:
                    best_energy = e
                    best_pos = relaxed.copy()
        else:
            # If invalid, try to fix by scaling
            scale = 0.52 / np.min(np.linalg.norm(pos, axis=1))
            pos_scaled = pos * scale
            if _is_valid(pos_scaled):
                relaxed = _relax(pos_scaled, maxiter=5000, ftol=1e-12, gtol=1e-10, deadline=deadline)
                if _is_valid(relaxed):
                    e = _lj_energy(relaxed)
                    if e < best_energy:
                        best_energy = e
                        best_pos = relaxed.copy()

    # Strategy 3: Direct Cartesian relaxation from standard Mackay geometry
    if time.time() < deadline:
        base = _manifold_to_cartesian(initial_params, template)
        if _is_valid(base):
            relaxed = _relax(base, maxiter=10000, ftol=1e-12, gtol=1e-10, deadline=deadline)
            if _is_valid(relaxed):
                e = _lj_energy(relaxed)
                if e < best_energy:
                    best_energy = e
                    best_pos = relaxed.copy()

    # Strategy 4: Small perturbations + relaxation (if time permits)
    if time.time() < deadline and best_pos is not None:
        noise_scales = [0.005, 0.01, 0.02]
        for scale in noise_scales:
            if time.time() > deadline:
                break
            noise = rng.normal(0, scale, size=best_pos.shape)
            pos = best_pos + noise
            if not _is_valid(pos):
                continue
            relaxed = _relax(pos, maxiter=5000, ftol=1e-12, gtol=1e-10, deadline=deadline)
            if _is_valid(relaxed):
                e = _lj_energy(relaxed)
                if e < best_energy:
                    best_energy = e
                    best_pos = relaxed.copy()

    # Strategy 5: Radial perturbations on manifold + relaxation
    if time.time() < deadline:
        for trial in range(15):
            if time.time() > deadline:
                break
            r1_p = 1.135 * (1 + rng.normal(0, 0.02))
            r2_p = 1.162 * (1 + rng.normal(0, 0.02))
            r3_p = 1.283 * (1 + rng.normal(0, 0.02))
            params = np.array([r1_p, r2_p, r3_p, 0.0, 0.0, 0.0])
            pos = _manifold_to_cartesian(params, template)
            if not _is_valid(pos):
                continue
            relaxed = _relax(pos, maxiter=5000, ftol=1e-12, gtol=1e-10, deadline=deadline)
            if _is_valid(relaxed):
                e = _lj_energy(relaxed)
                if e < best_energy:
                    best_energy = e
                    best_pos = relaxed.copy()

    # Final high-precision polish
    if best_pos is not None and time.time() < deadline:
        refined = _relax(best_pos, maxiter=50000, ftol=1e-15, gtol=1e-13, deadline=deadline)
        if _is_valid(refined):
            e = _lj_energy(refined)
            if e < best_energy:
                best_energy = e
                best_pos = refined

    # Fallback: return base Mackay geometry if nothing else worked
    if best_pos is None:
        best_pos = _manifold_to_cartesian(initial_params, template)

    return best_pos
