"""Broadband antireflection coating design with Simulated Annealing optimization."""
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

def _simulated_annealing(objective, x0, bounds, max_iter=5000, T0=1.0, T1=0.01, time_limit=None, start_time=None):
    """Simulated Annealing optimization."""
    n = len(x0)
    if time_limit is None:
        time_limit = 60.0
    if start_time is None:
        start_time = time.time()
    
    x = np.array(x0, dtype=float)
    f_current = objective(x)
    x_best = x.copy()
    f_best = f_current
    
    T = T0
    cooling_factor = (T1 / T0) ** (1.0 / max_iter)
    
    for i in range(max_iter):
        if time.time() - start_time > time_limit:
            break
            
        # Generate a new candidate by perturbing one or more dimensions
        # Randomly select a few dimensions to perturb
        n_perturb = max(1, int(n * 0.1))
        indices = np.random.choice(n, n_perturb, replace=False)
        
        x_new = x.copy()
        for idx in indices:
            # Perturbation size proportional to range
            range_size = bounds[idx][1] - bounds[idx][0]
            step = range_size * 0.1 * np.random.randn()
            x_new[idx] += step
            # Clip to bounds
            x_new[idx] = np.clip(x_new[idx], bounds[idx][0], bounds[idx][1])
        
        f_new = objective(x_new)
        
        # Acceptance criterion
        if f_new < f_current:
            x = x_new
            f_current = f_new
            if f_new < f_best:
                x_best = x_new.copy()
                f_best = f_new
        else:
            # Accept with probability exp(-(f_new - f_current) / T)
            delta_f = f_new - f_current
            if delta_f < 0:
                accept_prob = 1.0
            else:
                accept_prob = np.exp(-delta_f / T)
            
            if np.random.rand() < accept_prob:
                x = x_new
                f_current = f_new
        
        T *= cooling_factor
    
    return x_best, f_best

def design(problem):
    """Design antireflection coating using Simulated Annealing."""
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
    
    # Define some material patterns to try
    # Pattern 1: Alternating L, H, L, H...
    # Pattern 2: Alternating H, L, H, L...
    # We will try these for various numbers of layers.
    
    # Try different numbers of layers
    # For broadband AR, 4-12 layers are often optimal.
    # We will prioritize higher layer counts as they offer more freedom, but start lower to get a baseline.
    
    layer_counts = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    
    # Sort by priority: maybe start with mid-range, then expand?
    # Or just iterate. Let's iterate through them, but allocate more time to promising ones?
    # For simplicity, we'll just loop and let SA run for a fixed budget per config, 
    # but we can stop early if we find a very good solution.
    
    # Actually, a better strategy is to run SA for a fixed number of iterations or time 
    # for each configuration, and keep the best.
    
    # To make it more robust, we can also randomize the initial guess.
    
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
            
            # Initial guess: quarter-wave thicknesses
            thicknesses = []
            for m in materials:
                n_idx = problem["materials"][m]
                # Quarter wave at center wavelength
                t_qw = center_wl / (4.0 * n_idx)
                thicknesses.append(t_qw)
            
            # Check constraints
            if sum(thicknesses) > max_total_nm:
                scale = max_total_nm / sum(thicknesses)
                thicknesses = [t * scale for t in thicknesses]
            
            valid = True
            for t in thicknesses:
                if t < 0 or t > max_layer_nm:
                    valid = False
                    break
            if not valid:
                continue
            
            # Define objective
            bounds = [(0.0, max_layer_nm)] * n_layers
            
            def objective(x):
                # Enforce total thickness constraint
                total = sum(x)
                if total > max_total_nm:
                    scale = max_total_nm / total
                    x_scaled = [xi * scale for xi in x]
                else:
                    x_scaled = list(x)
                
                # Ensure non-negative
                x_scaled = [max(0.0, xi) for xi in x_scaled]
                
                return _evaluate(materials, x_scaled, problem)
            
            # Run SA
            # Time budget per SA run
            remaining_time = time_limit - (time.time() - start_time)
            # Allocate a portion of remaining time, but cap it to avoid one config taking too long
            # If many configs, we need to be fast.
            # Let's say max 10s per SA run, or 10% of remaining time.
            sa_time_limit = min(10.0, remaining_time * 0.5)
            
            if sa_time_limit < 1.0:
                break
                
            x0 = np.array(thicknesses, dtype=float)
            
            # Run SA with a fixed number of iterations, but bounded by time
            # 5000 iterations is a reasonable default for SA with vectorized eval.
            x_opt, f_opt = _simulated_annealing(
                objective, x0, bounds,
                max_iter=5000, T0=1.0, T1=0.01,
                time_limit=sa_time_limit,
                start_time=start_time
            )
            
            # Post-process: Clip and scale
            x_opt = np.clip(x_opt, 0, max_layer_nm)
            total = sum(x_opt)
            if total > max_total_nm:
                scale = max_total_nm / total
                x_opt = x_opt * scale
            
            thicknesses_opt = [float(xi) for xi in x_opt]
            
            # Validate
            if _is_valid(materials, thicknesses_opt, problem):
                score = _evaluate(materials, thicknesses_opt, problem)
                if score < best_overall_score:
                    best_overall_score = score
                    best_overall_materials = list(materials)
                    best_overall_thicknesses = list(thicknesses_opt)
                    
                    # If we find a very good solution, we might want to refine it further?
                    # Or just keep searching other configs.
    
    # If no valid result found, return a simple 1-layer design
    if not _is_valid(best_overall_materials, best_overall_thicknesses, problem):
        best_overall_materials = ["L"]
        best_overall_thicknesses = [center_wl / (4.0 * L_index)]
    
    # Final validation
    if not _is_valid(best_overall_materials, best_overall_thicknesses, problem):
        # Fallback: single quarter-wave L layer
        best_overall_materials = ["L"]
        best_overall_thicknesses = [center_wl / (4.0 * L_index)]
    
    return best_overall_materials, best_overall_thicknesses
# EVOLVE-BLOCK-END
