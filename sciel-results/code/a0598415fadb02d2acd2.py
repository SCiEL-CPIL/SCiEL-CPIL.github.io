"""Broadband antireflection coating design with Simulated Annealing and Nelder-Mead polishing."""
# EVOLVE-BLOCK-START
import numpy as np
import time

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

def _simulated_annealing(objective, x0, bounds, time_limit, start_time, T0=1.0, T1=0.01):
    """Simulated Annealing optimization with time limit."""
    n = len(x0)
    x = np.array(x0, dtype=float)
    f_current = objective(x)
    x_best = x.copy()
    f_best = f_current
    
    # Estimate max iterations based on time limit
    # Assume ~0.001s per iteration for small n
    max_iter = int(time_limit * 1000)
    if max_iter < 100:
        max_iter = 100
        
    T = T0
    cooling_factor = (T1 / T0) ** (1.0 / max_iter)
    
    for i in range(max_iter):
        if time.time() - start_time > time_limit:
            break
            
        # Generate a new candidate by perturbing one or more dimensions
        # Randomly select a few dimensions to perturb
        n_perturb = max(1, int(n * 0.2))
        indices = np.random.choice(n, n_perturb, replace=False)
        
        x_new = x.copy()
        for idx in indices:
            # Perturbation size proportional to range
            range_size = bounds[idx][1] - bounds[idx][0]
            # Use a Gaussian step that shrinks with temperature
            step = range_size * 0.1 * (T / T0) * np.random.randn()
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
                # Avoid overflow
                if T < 1e-10:
                    accept_prob = 0.0
                else:
                    accept_prob = np.exp(-delta_f / T)
            
            if np.random.rand() < accept_prob:
                x = x_new
                f_current = f_new
        
        T *= cooling_factor
    
    return x_best, f_best

def _nelder_mead_polish(f, x0, bounds, max_iter=50, tol=1e-6, time_limit=2.0, start_time=None):
    """Fast Nelder-Mead polishing step."""
    if start_time is None:
        start_time = time.time()
        
    n = len(x0)
    # Create initial simplex
    simplex = [np.array(x0, dtype=float)]
    for i in range(n):
        perturbation = np.zeros(n)
        step = 0.05 * (bounds[i][1] - bounds[i][0])
        if step < 1e-6: step = 1e-6
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

def design(problem):
    """Design antireflection coating using Simulated Annealing and Nelder-Mead polishing."""
    start_time = time.time()
    time_limit = 140.0  # Leave 10s safety margin
    
    L_index = problem["materials"]["L"]
    H_index = problem["materials"]["H"]
    center_wl = 0.5 * (problem["band_nm"][0] + problem["band_nm"][1])
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    best_overall_score = float('inf')
    best_overall_materials = ["L"]
    best_overall_thicknesses = [center_wl / (4.0 * L_index)]
    
    # Define layer counts to try
    # We prioritize mid-range counts which often provide the best trade-off
    layer_counts = [4, 6, 8, 10, 12, 14, 16]
    
    # Filter out invalid counts
    layer_counts = [n for n in layer_counts if n <= max_layers]
    
    # If no valid counts, fall back to 1
    if not layer_counts:
        layer_counts = [1]
        
    # We will allocate time to each configuration.
    # Total time available for search: time_limit - 5 (buffer)
    total_search_time = time_limit - 5
    time_per_config = total_search_time / len(layer_counts)
    
    for n_layers in layer_counts:
        if time.time() - start_time > time_limit - 5:
            break
            
        # Try two main patterns: L-H-L... and H-L-H...
        patterns = []
        if n_layers > 0:
            pat1 = ["L" if i % 2 == 0 else "H" for i in range(n_layers)]
            pat2 = ["H" if i % 2 == 0 else "L" for i in range(n_layers)]
            patterns.append(pat1)
            if n_layers > 1: # Only add second pattern if it's different
                if pat1 != pat2:
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
            
            # Define objective function
            def objective(x):
                # Enforce total thickness constraint by scaling if needed
                total = sum(x)
                if total > max_total_nm:
                    scale = max_total_nm / total
                    x_scaled = [xi * scale for xi in x]
                else:
                    x_scaled = list(x)
                
                # Ensure non-negative
                x_scaled = [max(0.0, xi) for xi in x_scaled]
                
                return _evaluate(materials, x_scaled, problem)
            
            bounds = [(0.0, max_layer_nm)] * n_layers
            x0 = np.array(thicknesses, dtype=float)
            
            # Run SA
            sa_time_limit = time_per_config * 0.7
            x_sa, f_sa = _simulated_annealing(objective, x0, bounds, sa_time_limit, start_time)
            
            # Clip and validate SA result
            x_sa = np.clip(x_sa, 0, max_layer_nm)
            total = sum(x_sa)
            if total > max_total_nm:
                scale = max_total_nm / total
                x_sa = x_sa * scale
            
            thicknesses_sa = [float(xi) for xi in x_sa]
            
            if _is_valid(materials, thicknesses_sa, problem):
                score_sa = _evaluate(materials, thicknesses_sa, problem)
                
                # Run NM Polish
                nm_time_limit = min(time_per_config * 0.3, 5.0) # Cap polish time
                if time.time() - start_time < time_limit - 5:
                    x_nm, f_nm = _nelder_mead_polish(objective, x_sa, bounds, 
                                                      max_iter=100, 
                                                      time_limit=nm_time_limit,
                                                      start_time=start_time)
                    
                    # Clip and validate NM result
                    x_nm = np.clip(x_nm, 0, max_layer_nm)
                    total = sum(x_nm)
                    if total > max_total_nm:
                        scale = max_total_nm / total
                        x_nm = x_nm * scale
                    
                    thicknesses_nm = [float(xi) for xi in x_nm]
                    
                    if _is_valid(materials, thicknesses_nm, problem):
                        score_nm = _evaluate(materials, thicknesses_nm, problem)
                        
                        # Choose the better of SA and NM
                        if score_nm < score_sa:
                            final_thicknesses = thicknesses_nm
                            final_score = score_nm
                        else:
                            final_thicknesses = thicknesses_sa
                            final_score = score_sa
                    else:
                        final_thicknesses = thicknesses_sa
                        final_score = score_sa
                else:
                    final_thicknesses = thicknesses_sa
                    final_score = score_sa
                
                if final_score < best_overall_score:
                    best_overall_score = final_score
                    best_overall_materials = list(materials)
                    best_overall_thicknesses = list(final_thicknesses)
    
    # Final validation and fallback
    if not _is_valid(best_overall_materials, best_overall_thicknesses, problem):
        best_overall_materials = ["L"]
        best_overall_thicknesses = [center_wl / (4.0 * L_index)]
    
    return best_overall_materials, best_overall_thicknesses
# EVOLVE-BLOCK-END
