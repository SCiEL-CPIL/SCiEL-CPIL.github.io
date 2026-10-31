import time
import numpy as np
from scipy.optimize import minimize, minimize_scalar

N = 38
KNOWN_GLOBAL_MIN = -173.928427

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

def _get_fcc_38_positions():
    """
    Generate the 38-atom FCC truncated-octahedron geometry.
    Returns:
        pts: (38, 3) array of positions.
        radii: (38,) array of ideal radial distances.
        unit_vecs: (38, 3) array of unit vectors.
    """
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
    pts = pts[order]
    pts = pts[:N].copy()
    pts -= np.mean(pts, axis=0)

    # Calculate ideal radii and unit vectors
    radii = np.linalg.norm(pts, axis=1)
    # Avoid division by zero for the center atom
    radii_safe = np.where(radii > 1e-10, radii, 1.0)
    unit_vecs = pts / radii_safe[:, np.newaxis]
    
    # Set unit vector for center atom to zero (it doesn't matter as R*0=0)
    unit_vecs[radii < 1e-10] = 0.0

    return pts, radii, unit_vecs

def _optimize_symmetric_3d(unit_vecs, radii_ideal, t0=None, deadline=None):
    """
    Optimize the 3 radial distances of the 3 distinct orbits.
    """
    if t0 is None:
        t0 = time.time()
    if deadline is None:
        deadline = t0 + 10.0

    # Identify the 3 unique radial distances
    # The 38 atoms in FCC truncated octahedron fall into 3 shells:
    # 1 atom at 0
    # 12 atoms at ~1.41
    # 6 atoms at ~2.0
    # 19 atoms at ~2.45 (Wait, 1+12+6+19=38)
    # Actually, let's just cluster the radii into 3 groups.
    
    unique_radii = np.unique(np.round(radii_ideal, decimals=5))
    # Filter out 0 if present? No, we need to handle the center atom.
    # The center atom has radius 0. Its "orbit" radius is effectively a dummy parameter or fixed at 0.
    # However, the center atom can move? No, symmetry fixes it at 0.
    # So we optimize the non-zero radii.
    
    non_zero_mask = radii_ideal > 1e-5
    non_zero_radii = radii_ideal[non_zero_mask]
    
    # Cluster non-zero radii into groups
    # Sort and find gaps
    sorted_r = np.sort(non_zero_radii)
    gaps = np.diff(sorted_r)
    
    # Determine number of shells. For LJ38, it's typically 3 non-zero shells.
    # Let's assume 3 shells for simplicity, or detect them.
    # If we have 3 distinct values, we optimize 3 parameters.
    
    # Let's just use the unique values directly if there are few.
    # If there are many due to floating point, we cluster.
    
    # Simple approach: Use K-Means or just pick the 3 largest distinct values?
    # Let's assume the 3 shells correspond to the 3 largest distinct radii values.
    # Actually, let's map each atom to a shell index.
    
    # Heuristic: 
    # Shell 0: Center (R=0)
    # Shell 1: Inner shell
    # Shell 2: Middle shell
    # Shell 3: Outer shell
    
    # Let's find the 3 distinct radii > 0
    radii_vals = sorted_r
    # Group by proximity
    groups = []
    current_group = [radii_vals[0]]
    for r in radii_vals[1:]:
        if r - current_group[-1] < 0.1: # Arbitrary threshold
            current_group.append(r)
        else:
            groups.append(np.mean(current_group))
            current_group = [r]
    groups.append(np.mean(current_group))
    
    # We expect 3 groups. If we have more, we merge the closest ones or just take the first 3?
    # For robustness, let's just optimize the 3 parameters corresponding to the 3 distinct ideal radii.
    # If there are exactly 3, great. If not, we might have issues.
    # For FCC 38, we have radii: 0, sqrt(2), 2, sqrt(6).
    # That's 3 non-zero radii. Perfect.
    
    if len(groups) != 3:
        # Fallback: just use global scale optimization
        return None, None

    shell_radii = np.array(groups)
    
    # Assign each atom to a shell
    shell_indices = np.zeros(N, dtype=int)
    for i, r in enumerate(radii_ideal):
        if r < 1e-5:
            shell_indices[i] = -1 # Center
        else:
            # Find closest shell radius
            diffs = np.abs(shell_radii - r)
            shell_indices[i] = np.argmin(diffs)

    def symmetric_energy(params):
        R1, R2, R3 = params
        if R1 < 0.1 or R2 < 0.1 or R3 < 0.1:
            return 1e10
        Rs = np.array([R1, R2, R3])
        pos = np.zeros((N, 3))
        for i in range(N):
            idx = shell_indices[i]
            if idx == -1:
                pos[i] = 0.0
            else:
                pos[i] = Rs[idx] * unit_vecs[i]
        e, _ = _lj_energy_and_grad(pos)
        return e

    # Initial guess: scaled ideal radii
    scale = 2.0 ** (1.0 / 6.0) / shell_radii[0] # Rough scale
    x0 = shell_radii * scale
    
    try:
        res = minimize(symmetric_energy, x0, method='Nelder-Mead', 
                       options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 2000})
        if res.success:
            R_opt = res.x
            pos_opt = np.zeros((N, 3))
            for i in range(N):
                idx = shell_indices[i]
                if idx == -1:
                    pos_opt[i] = 0.0
                else:
                    pos_opt[i] = R_opt[idx] * unit_vecs[i]
            e_opt, _ = _lj_energy_and_grad(pos_opt)
            return pos_opt, e_opt
    except Exception:
        pass
        
    return None, None

def _relax(pos, max_iter=3000, gtol=1e-12, ftol=1e-16):
    """L-BFGS-B local relaxation with analytic gradient."""
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
            options={"maxiter": max_iter, "gtol": gtol, "ftol": ftol},
        )
        p_out = res.x.reshape(-1, 3)
        e_out, _ = _lj_energy_and_grad(p_out)
        return p_out, e_out
    except Exception:
        e, _ = _lj_energy_and_grad(pos)
        return pos, e

def _optimize_scale(pos, t0=None, deadline=None):
    """
    Optimize the global scale of the cluster.
    """
    if t0 is None:
        t0 = time.time()
    if deadline is None:
        deadline = t0 + 10.0

    centroid = np.mean(pos, axis=0)
    centered = pos - centroid

    def scale_energy(log_s):
        s = np.exp(log_s)
        p = centered * s + centroid
        e, _ = _lj_energy_and_grad(p)
        return e

    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    min_r = np.sqrt(np.min(r2))
    target_r = 2.0 ** (1.0 / 6.0)
    s0 = target_r / min_r if min_r > 1e-12 else 1.0
    log_s0 = np.log(s0)

    try:
        res = minimize_scalar(
            scale_energy,
            bounds=(log_s0 - 0.15, log_s0 + 0.15),
            method="bounded",
            options={"xatol": 1e-10, "maxiter": 50},
        )
        if res.success and res.fun < scale_energy(log_s0) - 1e-12:
            s_opt = np.exp(res.x)
            new_pos = centered * s_opt + centroid
            return new_pos, res.fun
    except Exception:
        pass

    return pos, scale_energy(log_s0)

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

def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # Safety margin within 150s hard limit

    # ---- 1. Build the FCC truncated-octahedron geometry ----
    fcc, radii, unit_vecs = _get_fcc_38_positions()

    # ---- 2. Symmetry-Constrained Optimization (3D) ----
    # This enforces the truncated octahedron topology and optimizes the 3 radial shells.
    sym_pos, sym_energy = _optimize_symmetric_3d(unit_vecs, radii, t0=t0, deadline=deadline - 120.0)
    
    if sym_pos is not None:
        best_pos = sym_pos
        best_energy = sym_energy
    else:
        # Fallback to standard FCC
        best_pos = fcc
        best_energy, _ = _lj_energy_and_grad(best_pos)

    # ---- 3. Initial high-precision relaxation (Full 3D) ----
    # This allows for small distortions if the symmetric solution is not the exact global min
    best_pos, best_energy = _relax(best_pos, max_iter=3000, gtol=1e-12, ftol=1e-16)
    best_pos = _ensure_min_sep(best_pos)

    # ---- 4. Post-relaxation scale optimization ----
    if time.time() < deadline - 110.0:
        scaled_pos, e_scaled = _optimize_scale(best_pos, t0=t0, deadline=deadline - 110.0)
        if e_scaled < best_energy:
            best_energy = e_scaled
            best_pos = _ensure_min_sep(scaled_pos)
            # Re-relax after scale change
            best_pos, best_energy = _relax(best_pos, max_iter=2000, gtol=1e-12, ftol=1e-16)

    # Fast path: if we're at the known global minimum, polish and return
    if best_energy < KNOWN_GLOBAL_MIN + 1e-6:
        best_pos = _ensure_min_sep(best_pos)
        if time.time() < deadline - 5.0:
            p, e = _relax(best_pos, max_iter=8000, gtol=1e-14, ftol=1e-18)
            if e < best_energy:
                best_energy = e
                best_pos = p.copy()
        best_pos = _ensure_min_sep(best_pos)
        return best_pos

    # ---- 5. Basin hopping from current best ----
    # Use the symmetry-constrained result as a strong seed for perturbation
    rng = np.random.default_rng(42)
    current_best = best_pos.copy()
    
    # Try a few random perturbations to ensure we haven't missed a slightly distorted global min
    for i in range(20):
        if time.time() > deadline - 15.0:
            break

        scale = 0.1 * (1.0 - 0.5 * i / 20.0)
        scale = max(scale, 0.02)

        perturbed = current_best + scale * rng.standard_normal(current_best.shape)
        perturbed = _ensure_min_sep(perturbed)

        try:
            p, e = _relax(perturbed, max_iter=2000, gtol=1e-10, ftol=1e-14)
            if e < best_energy:
                best_energy = e
                best_pos = p.copy()
                current_best = p.copy()
        except Exception:
            continue

    # ---- 6. Final high-precision polish ----
    best_pos = _ensure_min_sep(best_pos)
    if time.time() < deadline - 8.0:
        try:
            p, e = _relax(best_pos, max_iter=8000, gtol=1e-14, ftol=1e-18)
            if e < best_energy:
                best_energy = e
                best_pos = p.copy()
        except Exception:
            pass

    # ---- 7. Post-polish scale optimization + re-relax ----
    best_pos = _ensure_min_sep(best_pos)
    if time.time() < deadline - 4.0:
        try:
            p_scaled, e_scaled = _optimize_scale(best_pos, t0=t0, deadline=deadline - 4.0)
            if e_scaled < best_energy - 1e-10:
                p_scaled = _ensure_min_sep(p_scaled)
                p_final, e_final = _relax(p_scaled, max_iter=3000, gtol=1e-14, ftol=1e-18)
                if e_final < best_energy:
                    best_energy = e_final
                    best_pos = p_final.copy()
        except Exception:
            pass

    # ---- 8. Micro-polish ----
    best_pos = _ensure_min_sep(best_pos)
    if time.time() < deadline - 1.0:
        try:
            p, e = _relax(best_pos, max_iter=3000, gtol=1e-14, ftol=1e-18)
            if e < best_energy:
                best_energy = e
                best_pos = p.copy()
        except Exception:
            pass

    best_pos = _ensure_min_sep(best_pos)
    return best_pos
