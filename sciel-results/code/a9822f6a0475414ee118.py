"""Broadband antireflection coating design with vectorized characteristic-matrix method and Nelder-Mead optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _reflectance_vectorized(wavelengths, thicknesses, indices, ambient_n, substrate_n):
    """Compute reflectance spectrum using vectorized characteristic matrix method.
    
    Parameters:
    - wavelengths: array of wavelengths in nm
    - thicknesses: array of physical thicknesses in nm
    - indices: array of refractive indices for each layer
    - ambient_n: ambient refractive index
    - substrate_n: substrate refractive index
    
    Returns:
    - reflectance: array of reflectance values (0 to 1)
    """
    n_layers = len(thicknesses)
    if n_layers == 0:
        r0 = (ambient_n - substrate_n) / (ambient_n + substrate_n)
        return np.full(len(wavelengths), np.abs(r0) ** 2)
    
    # Precompute phase factors for all layers and wavelengths
    # delta_j = 2*pi*n_j*d_j/lambda
    # For each layer j, delta_j is a vector over wavelengths
    
    # M_j = [[cos(delta_j), i*sin(delta_j)/n_j], [i*n_j*sin(delta_j), cos(delta_j)]]
    # We can compute this vectorized over wavelengths
    
    # Start with identity matrix (vectorized over wavelengths)
    M00 = np.ones(len(wavelengths), dtype=complex)
    M01 = np.zeros(len(wavelengths), dtype=complex)
    M10 = np.zeros(len(wavelengths), dtype=complex)
    M11 = np.ones(len(wavelengths), dtype=complex)
    
    for j in range(n_layers):
        n_j = indices[j]
        d_j = thicknesses[j]
        
        if d_j < 1e-10:
            # Zero-thickness layer: identity
            continue
        
        delta_j = 2.0 * np.pi * n_j * d_j / wavelengths
        cos_d = np.cos(delta_j)
        sin_d = np.sin(delta_j)
        
        # Handle sin_d near zero
        sin_d_safe = np.where(np.abs(sin_d) < 1e-15, 1e-15, sin_d)
        sin_d_safe = np.where(sin_d < 0, -np.abs(sin_d_safe), np.abs(sin_d_safe))
        
        # M_j elements
        m00 = cos_d
        m01 = 1j * sin_d_safe / n_j
        m10 = 1j * n_j * sin_d_safe
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

def _nelder_mead(f, x0, bounds, max_iter=200, tol=1e-8, time_limit=None, start_time=None):
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
        step = 0.1 * bounds[i][1] if bounds[i][1] - bounds[i][0] > 1e-6 else 1.0
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
    """Design antireflection coating using Nelder-Mead optimization on layer thicknesses."""
    start_time = time.time()
    time_limit = 145.0  # Leave 5s safety margin
    
    L_index = problem["materials"]["L"]
    H_index = problem["materials"]["H"]
    center_wl = 0.5 * (problem["band_nm"][0] + problem["band_nm"][1])
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    # Try different numbers of layers and find the best
    best_overall_score = float('inf')
    best_overall_materials = ["L"]
    best_overall_thicknesses = [center_wl / (4.0 * L_index)]
    
    # Start with a reasonable number of layers
    # For broadband AR, 4-8 layers typically work well
    for n_layers in [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20]:
        if time.time() - start_time > time_limit - 10:
            break
        if n_layers > max_layers:
            continue
        
        # Create initial design: alternating L and H layers
        materials = []
        for i in range(n_layers):
            if i % 2 == 0:
                materials.append("L")
            else:
                materials.append("H")
        
        # Initial thicknesses: quarter-wave at center wavelength
        thicknesses = []
        for m in materials:
            n_idx = problem["materials"][m]
            thicknesses.append(center_wl / (4.0 * n_idx))
        
        # Check total thickness
        if sum(thicknesses) > max_total_nm:
            # Scale down
            scale = max_total_nm / sum(thicknesses)
            thicknesses = [t * scale for t in thicknesses]
        
        # Check individual layer constraints
        valid = True
        for i, t in enumerate(thicknesses):
            if t < 0 or t > max_layer_nm:
                valid = False
                break
        if not valid:
            continue
        
        # Define objective function with bounds
        bounds = [(0.0, max_layer_nm)] * n_layers
        
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
        
        # Run Nelder-Mead
        x0 = np.array(thicknesses, dtype=float)
        
        # Time budget for this configuration
        remaining_time = time_limit - (time.time() - start_time)
        nm_time_limit = min(remaining_time - 5, 30.0)
        
        if nm_time_limit < 5:
            break
        
        x_opt, f_opt = _nelder_mead(
            objective, x0, bounds,
            max_iter=300, tol=1e-7,
            time_limit=nm_time_limit,
            start_time=start_time
        )
        
        # Clip and validate
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
