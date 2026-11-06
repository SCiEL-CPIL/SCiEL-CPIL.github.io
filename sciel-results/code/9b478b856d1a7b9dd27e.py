"""Broadband antireflection coating design with Analytical Initialization, Nelder-Mead, and Scaling Search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _reflectance_vectorized(wavelengths, thicknesses, indices, ambient_n, substrate_n):
    """Compute reflectance spectrum using vectorized characteristic matrix method."""
    n_layers = len(thicknesses)
    if n_layers == 0:
        r0 = (ambient_n - substrate_n) / (ambient_n + substrate_n)
        return np.full(len(wavelengths), np.abs(r0) ** 2)
    
    # Precompute phase factors
    # delta_j = 2*pi*n_j*d_j/lambda
    M00 = np.ones(len(wavelengths), dtype=complex)
    M01 = np.zeros(len(wavelengths), dtype=complex)
    M10 = np.zeros(len(wavelengths), dtype=complex)
    M11 = np.ones(len(wavelengths), dtype=complex)
    
    for j in range(n_layers):
        n_j = indices[j]
        d_j = thicknesses[j]
        
        if d_j < 1e-10:
            continue
        
        delta_j = 2.0 * np.pi * n_j * d_j / wavelengths
        cos_d = np.cos(delta_j)
        sin_d = np.sin(delta_j)
        
        # M_j elements
        m00 = cos_d
        m01 = 1j * sin_d / n_j
        m10 = 1j * n_j * sin_d
        m11 = cos_d
        
        # M = M @ M_j
        new_M00 = M00 * m00 + M01 * m10
        new_M01 = M00 * m01 + M01 * m11
        new_M10 = M10 * m00 + M11 * m10
        new_M11 = M10 * m01 + M11 * m11
        
        M00, M01, M10, M11 = new_M00, new_M01, new_M10, new_M11
    
    # B = M[0,0] + M[0,1] * n_s
    # C = M[1,0] * n_0 + M[1,1] * n_s
    # R = |(C - B) / (C + B)|^2
    B = M00 + M01 * substrate_n
    C = M10 * ambient_n + M11 * substrate_n
    
    denom = C + B
    # Avoid division by zero
    denom_safe = np.where(np.abs(denom) < 1e-15, 1e-15, denom)
    
    r = (C - B) / denom_safe
    reflectance = np.abs(r) ** 2
    
    return reflectance

def _evaluate(materials, thicknesses, problem):
    """Compute mean reflectance in percent over the band."""
    wavelengths = np.linspace(problem["band_nm"][0], problem["band_nm"][1], 401)
    indices = np.array([problem["materials"][m] for m in materials], dtype=float)
    thicknesses_arr = np.array(thicknesses, dtype=float)
    refl = _reflectance_vectorized(wavelengths, thicknesses_arr, indices,
                                   problem["ambient_index"], problem["substrate_index"])
    return 100.0 * np.mean(refl)

def _is_valid(materials, thicknesses, problem):
    """Check if the design is valid."""
    if len(materials) != len(thicknesses):
        return False
    if len(materials) > problem["max_layers"]:
        return False
    if len(materials) < 0:
        return False
    if any(t < -1e-9 or t > problem["max_layer_nm"] + 1e-9 for t in thicknesses):
        return False
    if sum(thicknesses) > problem["max_total_nm"] + 1e-9:
        return False
    if any(m not in problem["materials"] for m in materials):
        return False
    return True

def _nelder_mead(f, x0, bounds, max_iter=100, tol=1e-8, time_limit=None, start_time=None):
    """Nelder-Mead optimization with bounds."""
    n = len(x0)
    if time_limit is None:
        time_limit = 60.0
    if start_time is None:
        start_time = time.time()
    
    # Create initial simplex
    simplex = [np.array(x0, dtype=float)]
    for i in range(n):
        perturbation = np.zeros(n)
        # Use a smaller initial step for Nelder-Mead to avoid large jumps
        step = 0.05 * bounds[i][1] if bounds[i][1] - bounds[i][0] > 1e-6 else 1.0
        perturbation[i] = step
        simplex.append(np.array(x0, dtype=float) + perturbation)
    
    # Evaluate initial simplex
    fvals = [f(x) for x in simplex]
    
    alpha = 1.0   # reflection
    gamma = 2.0   # expansion
    rho = 0.5     # contraction
    sigma = 0.5   # shrink
    
    for iteration in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Sort by function value
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]
        
        # Check convergence
        if abs(fvals[-1] - fvals[0]) < tol:
            break
        
        # Centroid of all points except the worst
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        x_ref = centroid + alpha * (centroid - simplex[-1])
        x_ref = np.clip(x_ref, [b[0] for b in bounds], [b[1] for b in bounds])
        f_ref = f(x_ref)
        
        if f_ref < fvals[0]:
            # Expansion
            x_exp = centroid + gamma * (x_ref - centroid)
            x_exp = np.clip(x_exp, [b[0] for b in bounds], [b[1] for b in bounds])
            f_exp = f(x_exp)
            if f_exp < f_ref:
                simplex[-1] = x_exp
                fvals[-1] = f_exp
            else:
                simplex[-1] = x_ref
                fvals[-1] = f_ref
        elif f_ref < fvals[-2]:
            simplex[-1] = x_ref
            fvals[-1] = f_ref
        else:
            # Contraction
            x_con = centroid + rho * (simplex[-1] - centroid)
            x_con = np.clip(x_con, [b[0] for b in bounds], [b[1] for b in bounds])
            f_con = f(x_con)
            if f_con < fvals[-1]:
                simplex[-1] = x_con
                fvals[-1] = f_con
            else:
                # Shrink
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    simplex[i] = np.clip(simplex[i], [b[0] for b in bounds], [b[1] for b in bounds])
                    fvals[i] = f(simplex[i])
    
    # Return best point
    best_idx = np.argmin(fvals)
    return simplex[best_idx], fvals[best_idx]

def _scaling_search(materials, thicknesses, problem, time_limit, start_time):
    """Perform a 1D grid search on the global scaling factor."""
    best_score = float('inf')
    best_thicknesses = list(thicknesses)
    
    # Check current
    if _is_valid(materials, thicknesses, problem):
        best_score = _evaluate(materials, thicknesses, problem)
    
    # Scaling factors to try
    # The optimal scaling is often close to 1.0 but can vary.
    # We try a range of scales.
    scales = np.linspace(0.6, 1.4, 17)
    
    for s in scales:
        if time.time() - start_time > time_limit:
            break
        
        new_thick = [t * s for t in thicknesses]
        
        # Check max_layer_nm constraint
        if any(t > problem["max_layer_nm"] + 1e-9 for t in new_thick):
            # If a single layer exceeds max, we might need to clip or skip.
            # Clipping breaks the scaling logic, so we just skip this scale if it violates hard constraints
            # unless we can clip and still be valid? No, clipping changes the shape.
            # Better to just check validity.
            pass
        
        if _is_valid(materials, new_thick, problem):
            score = _evaluate(materials, new_thick, problem)
            if score < best_score:
                best_score = score
                best_thicknesses = new_thick
                
    return best_thicknesses, best_score

def design(problem):
    """Design antireflection coating using Analytical Init, Nelder-Mead, and Scaling Search."""
    start_time = time.time()
    time_limit = 145.0  # Leave 5s safety margin
    
    L_index = problem["materials"]["L"]
    H_index = problem["materials"]["H"]
    center_wl = 0.5 * (problem["band_nm"][0] + problem["band_nm"][1])
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    best_overall_score = float('inf')
    best_overall_materials = ["L"]
    best_overall_thicknesses = [center_wl / (4.0 * L_index)]
    
    # Layer counts to try
    # Broadband AR often benefits from more layers, but diminishing returns.
    layer_counts = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    
    for n_layers in layer_counts:
        if time.time() - start_time > time_limit - 5:
            break
        if n_layers > max_layers:
            continue
        
        # Try two main patterns
        patterns = []
        # Pattern 1: Start with L
        pat1 = ["L" if i % 2 == 0 else "H" for i in range(n_layers)]
        # Pattern 2: Start with H
        pat2 = ["H" if i % 2 == 0 else "L" for i in range(n_layers)]
        
        patterns.append(pat1)
        patterns.append(pat2)
        
        for materials in patterns:
            if time.time() - start_time > time_limit - 5:
                break
            
            # 1. Analytical Initialization: Quarter-wave at center wavelength
            thicknesses = []
            for m in materials:
                n_idx = problem["materials"][m]
                t_qw = center_wl / (4.0 * n_idx)
                thicknesses.append(t_qw)
            
            # Check if initial guess is valid
            if sum(thicknesses) > max_total_nm:
                # Scale down to fit
                scale = max_total_nm / sum(thicknesses)
                thicknesses = [t * scale for t in thicknesses]
            
            if not _is_valid(materials, thicknesses, problem):
                continue
            
            # 2. Local Optimization with Nelder-Mead
            bounds = [(0.0, max_layer_nm)] * n_layers
            x0 = np.array(thicknesses, dtype=float)
            
            # Objective function for NM
            def objective_nm(x):
                # Enforce total thickness constraint by scaling if necessary
                # NM might step outside, so we project back
                total = np.sum(x)
                if total > max_total_nm:
                    x_scaled = x * (max_total_nm / total)
                else:
                    x_scaled = x
                x_scaled = np.clip(x_scaled, 0, max_layer_nm)
                return _evaluate(materials, x_scaled, problem)
            
            # Allocate time for NM
            remaining_time = time_limit - (time.time() - start_time)
            nm_time_limit = min(5.0, remaining_time * 0.3) # Cap NM at 5s or 30% remaining
            if nm_time_limit < 0.5:
                break
                
            x_opt, f_opt = _nelder_mead(
                objective_nm, x0, bounds,
                max_iter=100, tol=1e-8,
                time_limit=nm_time_limit,
                start_time=start_time
            )
            
            # Post-process NM result
            x_opt = np.clip(x_opt, 0, max_layer_nm)
            total = np.sum(x_opt)
            if total > max_total_nm:
                x_opt = x_opt * (max_total_nm / total)
            
            thicknesses_nm = [float(xi) for xi in x_opt]
            
            # 3. Scaling Search
            # The NM result is a local minimum in the N-dimensional space.
            # A global scale change might be better for the *mean* reflectance.
            if time.time() - start_time > time_limit - 2:
                # Not enough time, just use NM result
                final_thicknesses = thicknesses_nm
            else:
                final_thicknesses, final_score = _scaling_search(
                    materials, thicknesses_nm, problem, time_limit, start_time
                )
            
            # Validate and update best
            if _is_valid(materials, final_thicknesses, problem):
                score = _evaluate(materials, final_thicknesses, problem)
                if score < best_overall_score:
                    best_overall_score = score
                    best_overall_materials = list(materials)
                    best_overall_thicknesses = list(final_thicknesses)
                    
    # Fallback
    if not _is_valid(best_overall_materials, best_overall_thicknesses, problem):
        best_overall_materials = ["L"]
        best_overall_thicknesses = [center_wl / (4.0 * L_index)]
    
    # Final validation
    if not _is_valid(best_overall_materials, best_overall_thicknesses, problem):
        best_overall_materials = ["L"]
        best_overall_thicknesses = [center_wl / (4.0 * L_index)]
        
    return best_overall_materials, best_overall_thicknesses
# EVOLVE-BLOCK-END
