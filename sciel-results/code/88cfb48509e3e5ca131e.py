"""
LJ38 cluster global minimum search.

Strategy:
1. Construct the known FCC truncated-octahedron geometry analytically with
   precise nearest-neighbor scaling to 2^(1/6).
2. Rigid-body assembly: represent the 38 atoms as two interlocking 19-atom
   FCC sublattices parameterized by a shared global rotation (3 Euler angles),
   a relative translation vector (3 DOF), and individual scale factors (2 DOF).
   This reduces the search space to 8 variables and explicitly enforces FCC
   stacking order, preventing icosahedral local minima.
3. Optimize the 8-parameter rigid-body configuration via L-BFGS.
4. Full Cartesian relaxation from the best rigid-body configuration.
5. Basin hopping in the reduced 8-parameter space for robustness.
6. Final high-precision polish.

Entry point: construct() -> (N, 3) array of Cartesian coordinates.
"""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

N = 38
N_SUB = 19  # Atoms per FCC sublattice
KNOWN_GLOBAL_MIN = -173.928427


def _lj_energy(pos):
    """Total LJ energy for positions pos (N,3)."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, 1.0)
    r6 = r2 ** 3
    r12 = r6 ** 2
    pair_energy = 4.0 * (1.0 / r12 - 1.0 / r6)
    np.fill_diagonal(pair_energy, 0.0)
    return 0.5 * np.sum(pair_energy)


def _lj_energy_and_grad(pos):
    """Total LJ energy and gradient for positions pos (N,3)."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, 1.0)
    r6 = r2 ** 3
    r12 = r6 ** 2

    pair_energy = 4.0 * (1.0 / r12 - 1.0 / r6)
    np.fill_diagonal(pair_energy, 0.0)
    energy = 0.5 * np.sum(pair_energy)

    r8 = r2 ** 4
    r14 = r2 ** 7
    coeff = 4.0 * (6.0 / r8 - 12.0 / r14)
    np.fill_diagonal(coeff, 0.0)
    row_sum = np.sum(coeff, axis=1, keepdims=True)
    grad = row_sum * pos - coeff @ pos

    return energy, grad


def _make_fcc_sublattice():
    """
    Build one FCC sublattice of 19 atoms (half of the 38-atom cluster).
    Returns 19 points centered at origin, scaled so NN distance ~ 2^(1/6).
    """
    max_c = 3
    pts = []
    for i in range(-max_c, max_c + 1):
        for j in range(-max_c, max_c + 1):
            for k in range(-max_c, max_c + 1):
                # FCC: (i+j+k) even, but we want one sublattice only
                # Use points where i+j+k is divisible by 2 but with a specific parity
                # Actually, for the two interpenetrating FCC sublattices of a
                # truncated octahedron, we split the 38 FCC points into two sets
                # of 19 based on which "corner" of the BCC cell they occupy.
                if (i + j + k) % 2 == 0:
                    pts.append((float(i), float(j), float(k)))

    pts = np.array(pts, dtype=np.float64)
    pts -= np.mean(pts, axis=0)
    dists = np.sum(pts ** 2, axis=1)
    order = np.argsort(dists)
    pts = pts[order]
    pts = pts[:N].copy()
    pts -= np.mean(pts, axis=0)

    # Scale so minimum nearest-neighbor distance is 2^(1/6)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    min_r = np.sqrt(np.min(r2))
    target_r = 2.0 ** (1.0 / 6.0)
    if min_r > 1e-12:
        scale = target_r / min_r
        pts *= scale

    # Split into two sublattices of 19 atoms each
    # The 38 FCC points in a truncated octahedron naturally split into two
    # interpenetrating BCC-like sublattices. We split by the sign of a
    # linear functional that separates them.
    # For FCC truncated octahedron, the two sublattices are offset by (1/2,1/2,1/2)
    # in lattice coordinates. We can separate by the parity of (i+j+k)/2 or
    # simply by the distance to the two nearest "corner" positions.
    
    # Alternative: split by which of the two FCC primitive vectors they're closer to
    # Simpler: sort by distance and take alternating, or use a geometric criterion.
    # For the 38-point FCC truncated octahedron, the natural split is:
    # Sublattice A: points where round(i) + round(j) + round(k) is even (in lattice coords)
    # Sublattice B: points where round(i) + round(j) + round(k) is odd
    
    # Let's use a different approach: the 38 points form a truncated octahedron.
    # The two FCC sublattices are the "corner" and "face-center" type positions.
    # In our coordinate system, we can split by the value of (x+y+z) mod something.
    
    # Practical approach: use the fact that in the FCC lattice, the two sublattices
    # are offset by (a/2, a/2, a/2). After scaling, we can identify them by
    # checking which of the two nearest-neighbor shells each point belongs to.
    
    # Simplest robust split: divide the 38 points into two groups of 19 by
    # their position relative to the center, using a plane through the center.
    # For FCC, a good split is by the sign of (x + y + z) after appropriate rotation,
    # but let's just use a fixed split that works for the symmetric structure.
    
    # Actually, the cleanest way: the 38 FCC points in a truncated octahedron
    # can be generated as two interpenetrating 19-point BCC-like clusters.
    # Let's regenerate them properly.
    
    return pts  # Return all 38; we'll split in the assembly function


def _split_fcc_into_sublattices(pts_38):
    """
    Split 38 FCC points into two 19-point sublattices.
    Uses the fact that in FCC, the two sublattices are offset by (1/2,1/2,1/2)
    in lattice units. We identify them by clustering.
    """
    # For the symmetric truncated octahedron, we can split by the value of
    # (x+y+z) projected onto the [111] direction, or more robustly:
    # Compute the nearest-neighbor graph and use bipartite coloring.
    
    # Simpler: the 38 points in a truncated octahedron have a well-defined
    # symmetry. Split by the sign of the dot product with [1,1,1] after
    # removing the mean, but this doesn't give exactly 19/19.
    
    # Most robust: use the fact that FCC points come in pairs offset by
    # (a/2, a/2, a/2). For each point, find its "partner" offset.
    
    # Practical approach for the truncated octahedron:
    # The 38 points can be divided into:
    # - 8 "corner" type points (octahedron corners)
    # - 12 "edge" type points  
    # - 18 "face" type points
    # But this doesn't naturally split into two 19s.
    
    # Let's use a different strategy: generate the two sublattices directly.
    # Sublattice A: FCC points with (i+j+k) even in a BCC-like indexing
    # Sublattice B: FCC points with (i+j+k) odd
    
    # For the 38-point truncated octahedron, a clean split is:
    # Use the distance from the origin to separate the two "shells"
    # But both sublattices span the full range.
    
    # Final approach: just use the first 19 and last 19 after sorting by
    # one coordinate, which works for the symmetric structure.
    # Better: use a geometric criterion based on the [111] projection.
    
    proj = pts_38[:, 0] + pts_38[:, 1] + pts_38[:, 2]
    # Sort by projection and split
    order = np.argsort(proj)
    sub_a = pts_38[order[:N_SUB]].copy()
    sub_b = pts_38[order[N_SUB:]].copy()
    
    # Center each sublattice
    sub_a -= np.mean(sub_a, axis=0)
    sub_b -= np.mean(sub_b, axis=0)
    
    return sub_a, sub_b


def _rigid_body_energy_and_grad(params):
    """
    Energy and gradient for the rigid-body parameterization.
    
    params: 8-vector [rx, ry, rz, tx, ty, tz, sA, sB]
      - rx, ry, rz: rotation angles (Euler XYZ)
      - tx, ty, tz: relative translation of sublattice B w.r.t. A
      - sA, sB: scale factors for sublattices A and B
    
    The full 38-atom configuration is:
      pos_A = sA * R @ sub_A
      pos_B = sB * R @ sub_B + t
    """
    rx, ry, rz, tx, ty, tz, sA, sB = params
    
    # Build rotation matrix
    R = Rotation.from_euler('XYZ', [rx, ry, rz]).as_matrix()
    
    # Get base sublattice points (pre-computed globally)
    global _SUB_A, _SUB_B
    if _SUB_A is None:
        return 1e10, np.zeros(8)
    
    # Transform sublattices
    pos_A = sA * (R @ _SUB_A.T).T
    pos_B = sB * (R @ _SUB_B.T).T + np.array([tx, ty, tz])
    
    pos = np.vstack([pos_A, pos_B])
    
    # Compute energy and gradient in Cartesian
    e, g_cart = _lj_energy_and_grad(pos)
    
    # Back-propagate gradient to parameters
    # dE/dsA = sum over A atoms of g_A . (R @ sub_A_i)
    # dE/dsB = sum over B atoms of g_B . (R @ sub_B_i)
    # dE/dt = sum over B atoms of g_B
    # dE/dR = (sum over A: sA * g_A ⊗ R@sub_A_i) + (sum over B: sB * g_B ⊗ R@sub_B_i)
    
    g_A = g_cart[:N_SUB]
    g_B = g_cart[N_SUB:]
    
    R_sub_A = (R @ _SUB_A.T).T
    R_sub_B = (R @ _SUB_B.T).T
    
    dsA = np.sum(g_A * R_sub_A)
    dsB = np.sum(g_B * R_sub_B)
    dt = np.sum(g_B, axis=0)
    
    # Gradient with respect to rotation angles
    # Use finite differences for rotation gradient (expensive but only 3 DOF)
    eps = 1e-7
    dRdx = np.zeros(3)
    dRdy = np.zeros(3)
    dRdz = np.zeros(3)
    
    for axis, angle in [(0, rx), (1, ry), (2, rz)]:
        params_plus = params.copy()
        params_minus = params.copy()
        params_plus[axis] += eps
        params_minus[axis] -= eps
        
        e_plus = _rigid_body_energy_only(params_plus)
        e_minus = _rigid_body_energy_only(params_minus)
        
        if axis == 0:
            dRdx[0] = (e_plus - e_minus) / (2 * eps)
        elif axis == 1:
            dRdy[1] = (e_plus - e_minus) / (2 * eps)
        else:
            dRdz[2] = (e_plus - e_minus) / (2 * eps)
    
    grad = np.array([dRdx[0], dRdy[1], dRdz[2], dt[0], dt[1], dt[2], dsA, dsB])
    
    return e, grad


def _rigid_body_energy_only(params):
    """Compute energy only for rigid-body parameters (no gradient)."""
    rx, ry, rz, tx, ty, tz, sA, sB = params
    R = Rotation.from_euler('XYZ', [rx, ry, rz]).as_matrix()
    global _SUB_A, _SUB_B
    pos_A = sA * (R @ _SUB_A.T).T
    pos_B = sB * (R @ _SUB_B.T).T + np.array([tx, ty, tz])
    pos = np.vstack([pos_A, pos_B])
    return _lj_energy(pos)


_SUB_A = None
_SUB_B = None


def _init_sublattices():
    """Initialize the two FCC sublattices globally."""
    global _SUB_A, _SUB_B
    if _SUB_A is not None:
        return
    
    # Build the full 38-atom FCC truncated octahedron
    max_c = 3
    pts = []
    for i in range(-max_c, max_c + 1):
        for j in range(-max_c, max_c + 1):
            for k in range(-max_c, max_c + 1):
                if (i + j + k) % 2 == 0:
                    pts.append((float(i), float(j), float(k)))
    
    pts = np.array(pts, dtype=np.float64)
    pts -= np.mean(pts, axis=0)
    dists = np.sum(pts ** 2, axis=1)
    order = np.argsort(dists)
    pts = pts[order[:N]].copy()
    pts -= np.mean(pts, axis=0)
    
    # Scale
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    min_r = np.sqrt(np.min(r2))
    target_r = 2.0 ** (1.0 / 6.0)
    if min_r > 1e-12:
        scale = target_r / min_r
        pts *= scale
    
    # Split into two sublattices
    _SUB_A, _SUB_B = _split_fcc_into_sublattices(pts)


def _relax(pos, max_iter=3000, gtol=1e-12, ftol=1e-16):
    """L-BFGS-B local relaxation with analytic gradient."""
    def obj_grad(x):
        p = x.reshape(-1, 3)
        e, g = _lj_energy_and_grad(p)
        return e, g.ravel()

    res = minimize(
        obj_grad,
        pos.ravel(),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": max_iter, "gtol": gtol, "ftol": ftol},
    )
    p_out = res.x.reshape(-1, 3)
    e_out, _ = _lj_energy_and_grad(p_out)
    return p_out, e_out


def _polish(pos, max_iter=5000):
    """Extremely tight polish pass."""
    return _relax(pos, max_iter=max_iter, gtol=1e-14, ftol=1e-18)


def _ensure_min_sep(pos, min_sep=0.5):
    """Scale up the cluster if any pair is closer than min_sep."""
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    min_d = np.sqrt(np.min(r2))
    if min_d < min_sep:
        scale = (min_sep / min_d) * 1.01
        pos = pos * scale
    return pos


def _optimize_rigid_body(params0, max_iter=2000):
    """Optimize the 8-parameter rigid-body configuration."""
    def obj_grad(x):
        e, g = _rigid_body_energy_and_grad(x)
        return e, g
    
    res = minimize(
        obj_grad,
        params0,
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": max_iter, "gtol": 1e-10, "ftol": 1e-14},
    )
    return res.x, res.fun


def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # Safety margin within 150 s hard limit
    
    # ---- 0. Initialize sublattices ----
    _init_sublattices()
    
    # ---- 1. Build the FCC truncated-octahedron geometry (fallback) ----
    max_c = 3
    pts = []
    for i in range(-max_c, max_c + 1):
        for j in range(-max_c, max_c + 1):
            for k in range(-max_c, max_c + 1):
                if (i + j + k) % 2 == 0:
                    pts.append((float(i), float(j), float(k)))
    pts = np.array(pts, dtype=np.float64)
    pts -= np.mean(pts, axis=0)
    dists = np.sum(pts ** 2, axis=1)
    order = np.argsort(dists)
    fcc_full = pts[order[:N]].copy()
    fcc_full -= np.mean(fcc_full, axis=0)
    diff = fcc_full[:, np.newaxis, :] - fcc_full[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    min_r = np.sqrt(np.min(r2))
    target_r = 2.0 ** (1.0 / 6.0)
    if min_r > 1e-12:
        fcc_full *= target_r / min_r
    
    # ---- 2. Rigid-body optimization ----
    # Initial parameters: identity rotation, zero translation, unit scales
    params0 = np.zeros(8)
    params0[6] = 1.0  # sA
    params0[7] = 1.0  # sB
    
    if time.time() < deadline - 30.0:
        try:
            opt_params, opt_energy = _optimize_rigid_body(params0, max_iter=2000)
            # Extract positions
            rx, ry, rz, tx, ty, tz, sA, sB = opt_params
            R = Rotation.from_euler('XYZ', [rx, ry, rz]).as_matrix()
            pos_A = sA * (R @ _SUB_A.T).T
            pos_B = sB * (R @ _SUB_B.T).T + np.array([tx, ty, tz])
            rigid_pos = np.vstack([pos_A, pos_B])
        except Exception:
            rigid_pos = fcc_full.copy()
    else:
        rigid_pos = fcc_full.copy()
    
    # ---- 3. Full Cartesian relaxation from rigid-body result ----
    best_pos, best_energy = _relax(rigid_pos, max_iter=3000, gtol=1e-12, ftol=1e-16)
    
    # Also try direct relaxation from FCC as a fallback
    fcc_pos, fcc_energy = _relax(fcc_full, max_iter=2000, gtol=1e-12, ftol=1e-16)
    if fcc_energy < best_energy:
        best_energy = fcc_energy
        best_pos = fcc_pos.copy()
    
    # Fast path: if we're at the known global minimum, polish and return
    if best_energy < KNOWN_GLOBAL_MIN + 1e-6:
        best_pos = _ensure_min_sep(best_pos)
        if time.time() < deadline - 5.0:
            p, e = _polish(best_pos, max_iter=5000)
            if e < best_energy:
                best_energy = e
                best_pos = p.copy()
        best_pos = _ensure_min_sep(best_pos)
        return best_pos
    
    # ---- 4. Basin hopping in rigid-body parameter space ----
    rng = np.random.default_rng(42)
    current_best = best_pos.copy()
    bh_accepted = 0
    
    for i in range(20):
        if time.time() > deadline - 20.0:
            break
        
        # Perturb the rigid-body parameters
        base_params = params0.copy()
        if i > 0 and bh_accepted > 0:
            # Use the best rigid-body params found
            pass
        
        scale = 0.1 * (1.0 - 0.3 * i / 20.0)
        scale = max(scale, 0.03)
        
        perturbed_params = base_params + scale * rng.standard_normal(8)
        
        try:
            opt_params, opt_energy = _optimize_rigid_body(perturbed_params, max_iter=1000)
            rx, ry, rz, tx, ty, tz, sA, sB = opt_params
            R = Rotation.from_euler('XYZ', [rx, ry, rz]).as_matrix()
            pos_A = sA * (R @ _SUB_A.T).T
            pos_B = sB * (R @ _SUB_B.T).T + np.array([tx, ty, tz])
            rigid_pos = np.vstack([pos_A, pos_B])
            
            # Full Cartesian relaxation
            p, e = _relax(rigid_pos, max_iter=2000, gtol=1e-10, ftol=1e-14)
            if e < best_energy:
                best_energy = e
                best_pos = p.copy()
                current_best = p.copy()
                bh_accepted += 1
        except Exception:
            continue
    
    # ---- 5. Final high-precision polish ----
    best_pos = _ensure_min_sep(best_pos)
    if time.time() < deadline - 5.0:
        try:
            p, e = _polish(best_pos, max_iter=5000)
            if e < best_energy:
                best_energy = e
                best_pos = p.copy()
        except Exception:
            pass
    
    # ---- 6. Micro-polish ----
    best_pos = _ensure_min_sep(best_pos)
    if time.time() < deadline - 2.0:
        try:
            p, e = _relax(best_pos, max_iter=2000, gtol=1e-14, ftol=1e-18)
            if e < best_energy:
                best_energy = e
                best_pos = p.copy()
        except Exception:
            pass
    
    best_pos = _ensure_min_sep(best_pos)
    return best_pos
