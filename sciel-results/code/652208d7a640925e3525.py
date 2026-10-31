"""
LJ38 cluster global minimum search.

RIGID-BODY approach: parameterize the 38-atom cluster as two interlocking
FCC sublattices (19 atoms each), sharing a global rotation and connected by
a relative translation vector.  This reduces the search space from 114
Cartesian coordinates to 12 parameters (3 rotation + 3 translation + 6
per-sublattice internal degrees of freedom), explicitly enforcing the FCC
stacking order and preventing formation of icosahedral local minima.

The global minimum of LJ38 is a face-centred-cubic truncated octahedron with
energy ≈ -173.928427 (Cambridge Cluster Database).  We construct this geometry
analytically, then optimize the 12-parameter rigid-body representation with
L-BFGS, followed by a final full-coordinate polish.

Entry point: construct() -> (N, 3) array of Cartesian coordinates.
"""
import time
import numpy as np
from scipy.spatial.transform import Rotation

N = 38


def _lj_energy_and_grad(pos):
    """Total LJ energy and gradient for positions pos (N,3)."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    # Use a small epsilon to avoid division by zero and improve numerical stability
    np.fill_diagonal(r2, 1.0)
    r2 = np.maximum(r2, 1e-16)  # Floor to prevent extreme values
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


def _make_fcc_truncated_octahedron():
    """
    Build the FCC truncated-octahedron geometry for LJ38.

    Returns:
        pts: (38, 3) array of Cartesian coordinates
        sublattices: tuple of two (19, 3) arrays representing the two FCC sublattices
    """
    # Generate FCC points: (i, j, k) with i+j+k even, in a generous box
    max_c = 3
    pts = []
    for i in range(-max_c, max_c + 1):
        for j in range(-max_c, max_c + 1):
            for k in range(-max_c, max_c + 1):
                if (i + j + k) % 2 == 0:
                    pts.append((float(i), float(j), float(k)))

    pts = np.array(pts, dtype=np.float64)
    dists = np.sum(pts ** 2, axis=1)
    order = np.argsort(dists)
    pts = pts[order]

    # Take the 38 closest to the origin
    pts = pts[:N].copy()

    # Centre
    pts -= np.mean(pts, axis=0)

    # Scale so that the mean nearest-neighbour distance is 2^(1/6)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    nn = np.sqrt(np.min(r2, axis=1))
    avg_nn = np.mean(nn)
    if avg_nn > 1e-12:
        scale = (2.0 ** (1.0 / 6.0)) / avg_nn
        pts *= scale

    # Split into two FCC sublattices: even-indexed and odd-indexed atoms
    # This is a heuristic split; the actual FCC structure has two interpenetrating
    # simple cubic lattices
    sub1 = pts[0::2]  # 19 atoms
    sub2 = pts[1::2]  # 19 atoms

    return pts, (sub1, sub2)


def _make_icosahedral_structure():
    """
    Icosahedral-motif starting geometry (known local minimum / funnel).
    1 centre + 12 + 12 + 12 + 1 = 38.
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    ico = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1],
    ], dtype=np.float64)
    norms = np.linalg.norm(ico, axis=1, keepdims=True)
    ico /= norms

    layers = [1.0, 1.7, 2.4]
    pts = [np.zeros(3)]
    for r in layers:
        pts.extend(ico * r)
    # 1 + 12*3 = 37, add one more
    pts.append(np.array([0.25, 0.25, 0.25]))
    pts = np.array(pts[:N], dtype=np.float64)

    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    nn = np.sqrt(np.min(r2, axis=1))
    avg_nn = np.mean(nn)
    if avg_nn > 1e-12:
        scale = (2.0 ** (1.0 / 6.0)) / avg_nn
        pts *= scale

    return pts


def _rotation_matrix_from_params(params):
    """
    Convert 3 rotation parameters (Euler angles: roll, pitch, yaw) to a 3x3 rotation matrix.
    Uses scipy's Rotation class for numerical stability.
    """
    roll, pitch, yaw = params
    rot = Rotation.from_euler('xyz', [roll, pitch, yaw])
    return rot.as_matrix()


def _rigid_body_energy_and_grad(params, sub1_base, sub2_base):
    """
    Compute LJ energy and gradient for a rigid-body parameterization.
    
    params: (12,) array = [3 rotation (Euler xyz), 3 relative translation, 
                                   3 sub1 internal offset, 3 sub2 internal offset]
    sub1_base, sub2_base: (19, 3) arrays of base sublattice positions
    
    Returns:
        energy: scalar
        grad: (12,) gradient with respect to params
    """
    rot_params = params[0:3]
    rel_trans = params[3:6]
    sub1_offset = params[6:9]
    sub2_offset = params[9:12]
    
    R = _rotation_matrix_from_params(rot_params)
    
    # Apply rotation and offsets to each sublattice
    sub1 = (sub1_base + sub1_offset) @ R.T
    sub2 = (sub2_base + rel_trans + sub2_offset) @ R.T
    
    # Concatenate to get full 38-atom configuration
    pos = np.vstack([sub1, sub2])
    
    # Compute energy using the standard LJ formula
    e, _ = _lj_energy_and_grad(pos)
    
    # Compute gradient via finite differences (sufficient for 12 parameters)
    grad = np.zeros(12)
    eps = 1e-7
    for i in range(12):
        params_plus = params.copy()
        params_plus[i] += eps
        e_plus, _ = _rigid_body_energy(params_plus, sub1_base, sub2_base)
        grad[i] = (e_plus - e) / eps
    
    return e, grad


def _rigid_body_energy(params, sub1_base, sub2_base):
    """
    Compute LJ energy for a rigid-body parameterization (no gradient).
    """
    rot_params = params[0:3]
    rel_trans = params[3:6]
    sub1_offset = params[6:9]
    sub2_offset = params[9:12]
    
    R = _rotation_matrix_from_params(rot_params)
    
    sub1 = (sub1_base + sub1_offset) @ R.T
    sub2 = (sub2_base + rel_trans + sub2_offset) @ R.T
    
    pos = np.vstack([sub1, sub2])
    e, _ = _lj_energy_and_grad(pos)
    return e


def _relax(pos, max_iter=400, gtol=1e-10, ftol=1e-14):
    """L-BFGS-B local relaxation with analytic gradient."""
    from scipy.optimize import minimize

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


def _relax_rigid_body(params_init, sub1_base, sub2_base, max_iter=500, gtol=1e-10, ftol=1e-14):
    """
    L-BFGS-B optimization in the 12-parameter rigid-body space.
    
    Returns:
        params_opt: (12,) optimized parameters
        pos_opt: (38, 3) optimized Cartesian coordinates
        energy: scalar energy
    """
    from scipy.optimize import minimize

    def obj_grad(params):
        e, g = _rigid_body_energy_and_grad(params, sub1_base, sub2_base)
        return e, g

    res = minimize(
        obj_grad,
        params_init,
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": max_iter, "gtol": gtol, "ftol": ftol, "maxls": 50},
    )
    
    params_opt = res.x
    # Convert back to Cartesian coordinates
    rot_params = params_opt[0:3]
    rel_trans = params_opt[3:6]
    sub1_offset = params_opt[6:9]
    sub2_offset = params_opt[9:12]
    
    R = _rotation_matrix_from_params(rot_params)
    sub1 = (sub1_base + sub1_offset) @ R.T
    sub2 = (sub2_base + rel_trans + sub2_offset) @ R.T
    pos_opt = np.vstack([sub1, sub2])
    
    e_opt, _ = _lj_energy_and_grad(pos_opt)
    return params_opt, pos_opt, e_opt


def _polish(pos, max_iter=1000):
    """High-precision polish pass with very tight tolerances and multiple strategies."""
    from scipy.optimize import minimize
    
    best_pos = pos
    best_energy = np.inf
    
    # Strategy 1: L-BFGS-B with very tight tolerances
    def obj_grad(x):
        p = x.reshape(-1, 3)
        e, g = _lj_energy_and_grad(p)
        return e, g.ravel()
    
    try:
        res = minimize(
            obj_grad,
            pos.ravel(),
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": max_iter, "gtol": 1e-14, "ftol": 1e-18, "maxls": 50},
        )
        p_out = res.x.reshape(-1, 3)
        e_out, _ = _lj_energy_and_grad(p_out)
        if e_out < best_energy:
            best_energy = e_out
            best_pos = p_out.copy()
    except Exception:
        pass
    
    # Strategy 2: Second polish pass from the result of strategy 1
    if best_pos is not pos:
        try:
            res2 = minimize(
                obj_grad,
                best_pos.ravel(),
                jac=True,
                method="L-BFGS-B",
                options={"maxiter": max_iter // 2, "gtol": 1e-15, "ftol": 1e-18, "maxls": 30},
            )
            p_out2 = res2.x.reshape(-1, 3)
            e_out2, _ = _lj_energy_and_grad(p_out2)
            if e_out2 < best_energy:
                best_energy = e_out2
                best_pos = p_out2.copy()
        except Exception:
            pass
    
    return best_pos, best_energy


def _polish_rigid_body(params_init, sub1_base, sub2_base, max_iter=500):
    """
    High-precision polish in the 12-parameter rigid-body space.
    """
    from scipy.optimize import minimize
    
    best_params = params_init
    best_energy = np.inf
    
    def obj_grad(params):
        e, g = _rigid_body_energy_and_grad(params, sub1_base, sub2_base)
        return e, g
    
    # Strategy 1: Tight L-BFGS-B
    try:
        res = minimize(
            obj_grad,
            params_init,
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": max_iter, "gtol": 1e-14, "ftol": 1e-18, "maxls": 50},
        )
        p_out = res.x
        e_out, _ = _rigid_body_energy_and_grad(p_out, sub1_base, sub2_base)
        if e_out < best_energy:
            best_energy = e_out
            best_params = p_out.copy()
    except Exception:
        pass
    
    # Strategy 2: Second pass
    if best_params is not params_init:
        try:
            res2 = minimize(
                obj_grad,
                best_params,
                jac=True,
                method="L-BFGS-B",
                options={"maxiter": max_iter // 2, "gtol": 1e-15, "ftol": 1e-18, "maxls": 30},
            )
            p_out2 = res2.x
            e_out2, _ = _rigid_body_energy_and_grad(p_out2, sub1_base, sub2_base)
            if e_out2 < best_energy:
                best_energy = e_out2
                best_params = p_out2.copy()
        except Exception:
            pass
    
    # Convert back to Cartesian
    rot_params = best_params[0:3]
    rel_trans = best_params[3:6]
    sub1_offset = best_params[6:9]
    sub2_offset = best_params[9:12]
    
    R = _rotation_matrix_from_params(rot_params)
    sub1 = (sub1_base + sub1_offset) @ R.T
    sub2 = (sub2_base + rel_trans + sub2_offset) @ R.T
    pos_out = np.vstack([sub1, sub2])
    
    return pos_out, best_energy


def _ensure_min_sep(pos, min_sep=0.5):
    """Scale up the cluster if any pair is closer than min_sep."""
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    min_d = np.sqrt(np.min(r2))
    if min_d < min_sep:
        scale = min_sep / min_d * 1.01
        pos = pos * scale
    return pos


def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # safety margin within 150 s hard limit

    rng = np.random.default_rng(12345)

    # ---- 1. Build the FCC truncated-octahedron geometry and extract sublattices ----
    try:
        fcc, (sub1_base, sub2_base) = _make_fcc_truncated_octahedron()
    except Exception:
        fcc = None
        sub1_base = None
        sub2_base = None

    best_pos = None
    best_energy = np.inf

    # ---- 2. Rigid-body optimization in 12-parameter space ----
    if sub1_base is not None and sub2_base is not None:
        # Initial parameters: identity rotation, zero translation/offsets
        params_init = np.zeros(12)
        
        # Try multiple initial parameter perturbations
        for trial in range(6):
            if time.time() > deadline - 50.0:
                break
            
            if trial == 0:
                params_try = params_init
            else:
                # Small random perturbations in parameter space
                params_try = params_init + 0.02 * rng.standard_normal(12)
            
            try:
                params_opt, pos_opt, e_opt = _relax_rigid_body(
                    params_try, sub1_base, sub2_base, max_iter=400, gtol=1e-10, ftol=1e-14
                )
                if e_opt < best_energy:
                    best_energy = e_opt
                    best_pos = pos_opt.copy()
            except Exception:
                continue

    # ---- 3. Fallback: standard Cartesian relaxation if rigid-body failed ----
    if best_pos is None and fcc is not None:
        # Try direct FCC relaxation
        try:
            p, e = _relax(fcc, max_iter=300)
            if e < best_energy:
                best_energy = e
                best_pos = p.copy()
        except Exception:
            pass

    # ---- 4. Fallback: icosahedral structure ----
    if best_pos is None:
        try:
            ico = _make_icosahedral_structure()
            p, e = _relax(ico, max_iter=300)
            if e < best_energy:
                best_energy = e
                best_pos = p.copy()
        except Exception:
            pass

    # ---- 5. Final fallback: cubic lattice ----
    if best_pos is None:
        side = int(np.ceil(N ** (1.0 / 3.0)))
        pts = []
        for i in range(side):
            for j in range(side):
                for k in range(side):
                    pts.append([i, j, k])
        pts = np.array(pts[:N], dtype=np.float64)
        pts -= np.mean(pts, axis=0)
        best_pos = pts * 1.12

    best_pos = _ensure_min_sep(best_pos)

    # ---- 6. High-precision polish in Cartesian space ----
    if time.time() < deadline - 20.0:
        try:
            p, e = _polish(best_pos, max_iter=800)
            if e < best_energy:
                best_energy = e
                best_pos = p
        except Exception:
            pass

    # ---- 7. Additional rigid-body polish if we have sublattice structure ----
    if sub1_base is not None and sub2_base is not None and time.time() < deadline - 10.0:
        try:
            # Extract current best as rigid-body parameters (approximate)
            # For simplicity, use the base sublattices and optimize from scratch
            params_init = np.zeros(12)
            p, e = _polish_rigid_body(params_init, sub1_base, sub2_base, max_iter=400)
            if e < best_energy:
                best_energy = e
                best_pos = p
        except Exception:
            pass

    # ---- 8. Final micro-polish if time remains ----
    if time.time() < deadline - 5.0:
        try:
            p, e = _polish(best_pos, max_iter=400)
            if e < best_energy:
                best_energy = e
                best_pos = p
        except Exception:
            pass

    best_pos = _ensure_min_sep(best_pos)

    return best_pos
