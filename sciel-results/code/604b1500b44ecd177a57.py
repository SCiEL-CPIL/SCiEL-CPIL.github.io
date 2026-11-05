"""Broadband antireflection coating design: Robust global search + Adaptive Nelder-Mead."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
import random

def _reflectance_array(materials, thicknesses, n_ambient, n_substrate, wls, n_mat):
    """
    Compute reflectance array for given materials and thicknesses using
    the thin-film characteristic-matrix method.
    Returns array of reflectance values (0 to 1).
    """
    if len(materials) == 0:
        r = (n_ambient - n_substrate) / (n_ambient + n_substrate)
        return np.full(len(wls), r * r)

    n_layers = len(materials)
    n_wls = len(wls)
    
    # Precompute beta = 2*pi*n*d/wl for all layers and wavelengths
    # beta shape: (n_layers, n_wls)
    beta = np.zeros((n_layers, n_wls))
    for i, m in enumerate(materials):
        n = n_mat[m]
        d = thicknesses[i]
        if d == 0:
            continue
        beta[i, :] = 2.0 * math.pi * n * d / wls

    cosb = np.cos(beta)
    sinb = np.sin(beta)

    # Build the total characteristic matrix
    M11 = np.ones(n_wls, dtype=complex)
    M12 = np.zeros(n_wls, dtype=complex)
    M21 = np.zeros(n_wls, dtype=complex)
    M22 = np.ones(n_wls, dtype=complex)

    for i, m in enumerate(materials):
        n = n_mat[m]
        c = cosb[i]
        s = sinb[i]
        
        # Matrix elements for layer i
        A = c
        B = 1j * s / n
        C = 1j * n * s
        D = c

        # Matrix multiplication: M_new = M_old * M_i
        nM11 = M11 * A + M12 * C
        nM12 = M11 * B + M12 * D
        nM21 = M21 * A + M22 * C
        nM22 = M21 * B + M22 * D
        
        M11, M12, M21, M22 = nM11, nM12, nM21, nM22

    # Admittance Y
    Y = (M21 + n_substrate * M22) / (M11 + n_substrate * M12)
    
    # Reflectance r
    r = (n_ambient - Y) / (n_ambient + Y)
    
    # Reflectance R = |r|^2
    R = np.abs(r) ** 2
    return R

def _mean_reflectance_percent(materials, thicknesses, problem, wls, n_mat):
    """
    Compute mean reflectance in percent.
    Returns penalty if constraints are violated.
    """
    if len(materials) != len(thicknesses):
        return 1e6
    if len(materials) > problem["max_layers"]:
        return 1e6
        
    max_layer = problem["max_layer_nm"]
    max_total = problem["max_total_nm"]
    
    for t in thicknesses:
        if t < -1e-9 or t > max_layer + 1e-9:
            return 1e6
            
    total_th = sum(thicknesses)
    if total_th > max_total + 1e-6:
        return 1e6 + (total_th - max_total) * 10.0

    if len(materials) == 0:
        r = (problem["ambient_index"] - problem["substrate_index"]) / (
            problem["ambient_index"] + problem["substrate_index"])
        return r * r * 100.0

    R = _reflectance_array(materials, thicknesses, problem["ambient_index"],
                           problem["substrate_index"], wls, n_mat)
    return float(np.mean(R)) * 100.0

def _nelder_mead_adaptive(objective, x0, bounds, deadline, max_iter=500, tol=1e-7):
    """
    Nelder-Mead with adaptive restarts on stagnation.
    """
    x0 = np.array(x0, dtype=float)
    n = len(x0)
    lo, hi = bounds
    
    def _init_simplex(x_center):
        simplex = [x_center.copy()]
        for i in range(n):
            xi = x_center.copy()
            range_i = hi[i] - lo[i]
            step = 0.1 * range_i if range_i > 1e-6 else 1.0
            xi[i] += step
            simplex.append(np.clip(xi, lo, hi))
        f_vals = [objective(x) for x in simplex]
        return simplex, f_vals

    simplex, f_vals = _init_simplex(x0)
    
    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    sigma = 0.5
    
    best_f = min(f_vals)
    stagnant_count = 0
    last_best_update = time.time()
    
    for iter_count in range(max_iter):
        if time.time() > deadline:
            break
            
        order = np.argsort(f_vals)
        simplex = [simplex[i] for i in order]
        f_vals = [f_vals[i] for i in order]
        
        if abs(f_vals[0] - f_vals[-1]) < tol:
            break
        
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        x_r = np.clip(centroid + alpha * (centroid - simplex[-1]), lo, hi)
        f_r = objective(x_r)
        
        if f_vals[0] <= f_r < f_vals[-2]:
            simplex[-1] = x_r
            f_vals[-1] = f_r
            if f_r < best_f - 1e-10:
                best_f = f_r
                stagnant_count = 0
            continue
            
        # Expansion
        if f_r < f_vals[0]:
            x_e = np.clip(centroid + gamma * (x_r - centroid), lo, hi)
            f_e = objective(x_e)
            if f_e < f_r:
                simplex[-1] = x_e
                f_vals[-1] = f_e
                if f_e < best_f - 1e-10:
                    best_f = f_e
                    stagnant_count = 0
            else:
                simplex[-1] = x_r
                f_vals[-1] = f_r
            continue
            
        # Contraction
        x_c = np.clip(centroid + rho * (simplex[-1] - centroid), lo, hi)
        f_c = objective(x_c)
        if f_c < f_vals[-1]:
            simplex[-1] = x_c
            f_vals[-1] = f_c
            if f_c < best_f - 1e-10:
                best_f = f_c
                stagnant_count = 0
            continue
            
        # Shrink
        for i in range(1, n + 1):
            simplex[i] = np.clip(simplex[0] + sigma * (simplex[i] - simplex[0]), lo, hi)
            f_vals[i] = objective(simplex[i])
            
        if min(f_vals) < best_f - 1e-10:
            best_f = min(f_vals)
            stagnant_count = 0
        else:
            stagnant_count += 1
            
        if stagnant_count > 20:
            # Restart from best point with larger perturbation
            x_best = simplex[0].copy()
            simplex = [x_best.copy()]
            for i in range(n):
                xi = x_best.copy()
                range_i = hi[i] - lo[i]
                step = 0.2 * range_i if range_i > 1e-6 else 2.0
                xi[i] += step
                simplex.append(np.clip(xi, lo, hi))
            f_vals = [objective(x) for x in simplex]
            stagnant_count = 0
            
    order = np.argsort(f_vals)
    best_idx = order[0]
    return np.clip(simplex[best_idx], lo, hi), f_vals[best_idx]

def _coordinate_descent(objective, x0, bounds, deadline, max_iter=50, step_init=5.0):
    """
    Fast coordinate descent for initial refinement.
    """
    x = np.array(x0, dtype=float)
    lo, hi = bounds
    n = len(x)
    f_current = objective(x)
    steps = np.full(n, step_init)
    
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        improved = False
        for i in range(n):
            # Try positive
            x_try = x.copy()
            x_try[i] = min(x[i] + steps[i], hi[i])
            f_try = objective(x_try)
            if f_try < f_current:
                x = x_try
                f_current = f_try
                steps[i] *= 1.2
                improved = True
                continue
            else:
                steps[i] *= 0.8

            # Try negative
            x_try = x.copy()
            x_try[i] = max(x[i] - steps[i], lo[i])
            f_try = objective(x_try)
            if f_try < f_current:
                x = x_try
                f_current = f_try
                steps[i] *= 1.2
                improved = True
                continue
            else:
                steps[i] *= 0.8
        
        if not improved:
            break
    
    return x, f_current

def _enforce_constraints(materials, thicknesses, max_total, max_layer):
    """
    Scale thicknesses to fit within max_total and max_layer.
    """
    ths = np.array(thicknesses, dtype=float)
    ths = np.clip(ths, 0.0, max_layer)
    
    total = np.sum(ths)
    if total > max_total:
        factor = max_total / total
        ths = ths * factor
        ths = np.clip(ths, 0.0, max_layer)
        
    return list(ths)

def design(problem):
    """
    Design a broadband antireflection coating.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    n_ambient = problem["ambient_index"]
    n_substrate = problem["substrate_index"]
    max_layers = problem["max_layers"]
    max_total = problem["max_total_nm"]
    max_layer = problem["max_layer_nm"]
    
    band_start, band_end = problem["band_nm"]
    wls = np.linspace(band_start, band_end, 401)
    n_mat = problem["materials"]
    center_wl = 0.5 * (band_start + band_end)
    
    best_val = float('inf')
    best_mats = None
    best_ths = None
    
    # Generate candidate sequences
    # We want to explore different structural arrangements
    candidates = []
    
    # 1. Alternating sequences (standard)
    for n_layers in [2, 4, 6, 8, 10, 12, 14]:
        if n_layers > max_layers: continue
        # Start with L
        mats_L = ['L' if i % 2 == 0 else 'H' for i in range(n_layers)]
        # Start with H
        mats_H = ['H' if i % 2 == 0 else 'L' for i in range(n_layers)]
        candidates.append(mats_L)
        candidates.append(mats_H)
        
    # 2. Block sequences (e.g., LLHH, HLLH)
    for n_layers in [4, 6, 8]:
        if n_layers > max_layers: continue
        # Random blocks
        for _ in range(5):
            mats = []
            for i in range(n_layers):
                if i % 2 == 0:
                    mats.append('L')
                else:
                    mats.append('H')
            # Randomly swap some to create non-alternating patterns
            if n_layers >= 4:
                swap_idx = random.randint(1, n_layers - 2)
                if mats[swap_idx] == mats[swap_idx - 1]:
                    continue # Don't create same-adjacent if we want diversity? No, allow it.
                # Actually, let's just generate random valid sequences for some counts
            candidates.append(mats)
            
    # 3. Random sequences for broader exploration
    for _ in range(20):
        n_layers = random.randint(2, min(12, max_layers))
        mats = [random.choice(['L', 'H']) for _ in range(n_layers)]
        candidates.append(mats)
        
    # Deduplicate
    unique_candidates = []
    seen = set()
    for c in candidates:
        key = tuple(c)
        if key not in seen:
            seen.add(key)
            unique_candidates.append(c)
            
    # Sort candidates by heuristic: prefer shorter, alternating, starting with L
    def heuristic_score(mats):
        n = len(mats)
        alt_score = 0
        for i in range(len(mats)-1):
            if mats[i] == mats[i+1]:
                alt_score += 1
        start_score = 0 if mats[0] == 'L' else 1
        return (n, alt_score, start_score)
        
    unique_candidates.sort(key=heuristic_score)
    
    # Evaluate and refine top candidates
    # We have ~140s. Let's spend ~10s on initial eval, rest on refinement.
    eval_deadline = start_time + 10.0
    
    scored_candidates = []
    for mats in unique_candidates:
        if time.time() > eval_deadline:
            break
        n_layers = len(mats)
        # Initial thicknesses: quarter wave at center
        ths = []
        for m in mats:
            n = n_mat[m]
            ths.append(center_wl / (4.0 * n))
            
        ths = _enforce_constraints(mats, ths, max_total, max_layer)
        
        val = _mean_reflectance_percent(mats, ths, problem, wls, n_mat)
        scored_candidates.append((val, mats, ths))
        
    # Sort by initial value
    scored_candidates.sort(key=lambda x: x[0])
    
    # Refine the top 10 candidates
    num_to_refine = min(10, len(scored_candidates))
    
    for i in range(num_to_refine):
        if time.time() > deadline:
            break
            
        init_val, init_mats, init_ths = scored_candidates[i]
        
        # Update global best
        if init_val < best_val:
            best_val = init_val
            best_mats = init_mats
            best_ths = init_ths
            
        materials = list(init_mats)
        n_layers = len(materials)
        
        # Bounds for optimizer
        lo = np.zeros(n_layers)
        hi = np.full(n_layers, max_layer)
        x0 = np.array(init_ths, dtype=float)
        
        def obj(x):
            # Ensure constraints inside objective for safety? 
            # The optimizer might go out of bounds, but we clip in NM.
            # However, sum constraint is tricky. 
            # Strategy: Optimize unconstrained, then project.
            # But projection changes the objective value.
            # Better: Use a penalty function or just rely on clipping and check sum.
            # Given the strict sum constraint, we will scale x after optimization.
            # So objective here is just the physics for the given x, assuming x is valid.
            # We must ensure x is valid before calling physics.
            ths = list(x)
            # Quick check
            if any(t < 0 or t > max_layer for t in ths):
                return 1e6
            if sum(ths) > max_total:
                return 1e6
            return _mean_reflectance_percent(materials, ths, problem, wls, n_mat)
        
        # 1. Coordinate Descent (fast)
        x_cd, f_cd = _coordinate_descent(obj, x0, (lo, hi), deadline, max_iter=50, step_init=5.0)
        
        # Enforce constraints on CD result
        x_cd = _enforce_constraints(materials, x_cd, max_total, max_layer)
        
        # 2. Nelder-Mead (robust)
        # Start NM from CD result
        x_nm, f_nm = _nelder_mead_adaptive(obj, x_cd, (lo, hi), deadline, max_iter=200)
        
        # Enforce constraints on NM result
        x_nm = _enforce_constraints(materials, x_nm, max_total, max_layer)
        
        # Final evaluation
        val = _mean_reflectance_percent(materials, x_nm, problem, wls, n_mat)
        
        if val < best_val:
            best_val = val
            best_mats = materials
            best_ths = list(x_nm)
            
    # Fallback
    if best_mats is None:
        best_mats = ["L"]
        best_ths = [center_wl / (4.0 * n_L)]
        best_val = _mean_reflectance_percent(best_mats, best_ths, problem, wls, n_mat)

    # Final safety check
    best_ths = _enforce_constraints(best_mats, best_ths, max_total, max_layer)
    
    return list(best_mats), list(best_ths)

# EVOLVE-BLOCK-END
