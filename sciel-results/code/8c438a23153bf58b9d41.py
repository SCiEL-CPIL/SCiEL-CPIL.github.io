"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _reflectance_spectrum(materials, thicknesses, n_layers, ambient_index, substrate_index, wavelengths):
    """
    Compute reflectance spectrum using characteristic-matrix method (optimized).
    Returns array of reflectance values (0 to 1) for each wavelength.
    """
    n_w = len(wavelengths)
    
    if n_layers == 0:
        r = (ambient_index - substrate_index) / (ambient_index + substrate_index)
        return np.full(n_w, r ** 2)
    
    # Precompute optical thicknesses: n_j * d_j for each layer
    optical_thick = np.array(materials, dtype=float) * np.array(thicknesses, dtype=float)
    
    # delta[j, i] = 2*pi*optical_thick[j]/wavelengths[i]  shape (n_layers, n_w)
    delta = 2.0 * np.pi * optical_thick[:, None] / wavelengths[None, :]
    
    cos_d = np.cos(delta)  # (n_layers, n_w)
    sin_d = np.sin(delta)  # (n_layers, n_w)
    
    # Initialize the running product as identity for each wavelength
    M00 = np.ones(n_w, dtype=complex)
    M01 = np.zeros(n_w, dtype=complex)
    M10 = np.zeros(n_w, dtype=complex)
    M11 = np.ones(n_w, dtype=complex)
    
    for j in range(n_layers):
        c = cos_d[j, :]
        s = sin_d[j, :]
        n_j = materials[j]
        
        # New M = M_old @ M_j
        # M_j = [[c, 1j*s/n_j], [1j*n_j*s, c]]
        new_M00 = M00 * c + M01 * (1j * s / n_j)
        new_M01 = M00 * (1j * s / n_j) + M01 * c
        new_M10 = M10 * c + M11 * (1j * n_j * s)
        new_M11 = M10 * (1j * n_j * s) + M11 * c
        
        M00, M01, M10, M11 = new_M00, new_M01, new_M10, new_M11
    
    # Now compute B and C:
    # [B, C]^T = M @ [1, n_s]^T
    B = M00 * 1.0 + M01 * substrate_index
    C = M10 * 1.0 + M11 * substrate_index
    
    # Reflectance:
    # r = (n_0 * B - C) / (n_0 * B + C)
    # R = |r|^2
    r = (ambient_index * B - C) / (ambient_index * B + C)
    R = np.abs(r) ** 2
    
    return R

def _nelder_mead(objective, x0, bounds, max_iter=500, tol=1e-8, deadline=None):
    """
    Nelder-Mead optimization with bounds.
    Returns (best_x, best_f).
    """
    n = len(x0)
    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    sigma = 0.5
    
    # Initialize simplex
    simplex = [np.array(x0, dtype=float)]
    for i in range(n):
        perturbation = np.zeros(n)
        perturbation[i] = 0.1 * (bounds[i][1] - bounds[i][0]) if bounds[i][1] > bounds[i][0] else 0.1
        simplex.append(np.array(x0 + perturbation, dtype=float))
    
    # Evaluate initial simplex
    f_values = []
    for i in range(n + 1):
        simplex[i] = np.clip(simplex[i], bounds[:, 0], bounds[:, 1])
        f_values.append(objective(simplex[i]))
    
    best_x = simplex[0].copy()
    best_f = f_values[0]
    
    for iteration in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Sort simplex by function value
        order = np.argsort(f_values)
        simplex = [simplex[i].copy() for i in order]
        f_values = [f_values[i] for i in order]
        
        if f_values[0] < best_f:
            best_f = f_values[0]
            best_x = simplex[0].copy()
        
        # Check convergence
        if np.max(f_values) - np.min(f_values) < tol:
            break
        
        # Centroid of all but worst point
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        x_r = centroid + alpha * (centroid - simplex[-1])
        x_r = np.clip(x_r, bounds[:, 0], bounds[:, 1])
        f_r = objective(x_r)
        
        if f_r < f_values[0]:
            # Expansion
            x_e = centroid + gamma * (x_r - centroid)
            x_e = np.clip(x_e, bounds[:, 0], bounds[:, 1])
            f_e = objective(x_e)
            if f_e < f_r:
                simplex[-1] = x_e
                f_values[-1] = f_e
            else:
                simplex[-1] = x_r
                f_values[-1] = f_r
        elif f_r < f_values[-2]:
            simplex[-1] = x_r
            f_values[-1] = f_r
        else:
            # Contraction
            x_c = centroid + rho * (simplex[-1] - centroid)
            x_c = np.clip(x_c, bounds[:, 0], bounds[:, 1])
            f_c = objective(x_c)
            if f_c < f_values[-1]:
                simplex[-1] = x_c
                f_values[-1] = f_c
            else:
                # Shrink
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    simplex[i] = np.clip(simplex[i], bounds[:, 0], bounds[:, 1])
                    f_values[i] = objective(simplex[i])
    
    return best_x, best_f

def _simulated_annealing(objective, x0, bounds, max_iter, deadline=None, T_start=10.0, T_end=0.01, cooling_rate=0.95, noise_scale=1.0):
    """
    Simulated Annealing optimization.
    """
    n = len(x0)
    current_x = np.array(x0, dtype=float)
    current_f = objective(current_x)
    best_x = current_x.copy()
    best_f = current_f
    
    T = T_start
    # Adjust cooling rate to reach T_end in max_iter steps if possible
    # T_end = T_start * cooling_rate^max_iter => cooling_rate = (T_end/T_start)^(1/max_iter)
    if max_iter > 0:
        cooling_rate = (T_end / T_start) ** (1.0 / max_iter)
    
    for i in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
            
        # Generate neighbor
        # Perturb each dimension with Gaussian noise scaled by T
        noise = np.random.normal(0, T * noise_scale, size=n)
        neighbor_x = current_x + noise
        neighbor_x = np.clip(neighbor_x, bounds[:, 0], bounds[:, 1])
        
        neighbor_f = objective(neighbor_x)
        
        # Acceptance probability
        if neighbor_f < current_f:
            current_x = neighbor_x
            current_f = neighbor_f
            if current_f < best_f:
                best_f = current_f
                best_x = current_x.copy()
        else:
            delta_f = neighbor_f - current_f
            # Avoid log(0)
            if T > 1e-9:
                probability = np.exp(-delta_f / T)
            else:
                probability = 0.0
            if random.random() < probability:
                current_x = neighbor_x
                current_f = neighbor_f
        
        # Cool
        T *= cooling_rate
        
    return best_x, best_f

def design(problem):
    """
    Design a broadband antireflection coating using Simulated Annealing 
    for global search and Nelder-Mead for local polishing.
    """
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s limit
    
    materials_dict = problem["materials"]
    n_L = materials_dict["L"]
    n_H = materials_dict["H"]
    ambient_index = problem["ambient_index"]
    substrate_index = problem["substrate_index"]
    max_layers = problem.get("max_layers", 20)
    max_total_nm = problem.get("max_total_nm", 1000)
    max_layer_nm = problem.get("max_layer_nm", 400)
    band = problem["band_nm"]
    
    # Precompute wavelength grid for fast evaluation
    wavelengths = np.linspace(band[0], band[1], 401)
    
    def fast_reflectance(materials, thicknesses):
        """Fast reflectance computation using precomputed wavelengths."""
        n_layers = len(materials)
        if n_layers == 0:
            R = ((ambient_index - substrate_index) / 
                 (ambient_index + substrate_index)) ** 2
            return R * 100.0
        R = _reflectance_spectrum(materials, thicknesses, n_layers, 
                                   ambient_index, substrate_index, wavelengths)
        return np.mean(R) * 100.0
    
    # Track best solution
    best_materials = None
    best_thicknesses = None
    best_score = float('inf')
    
    centre = 0.5 * (band[0] + band[1])
    
    # We will focus on high layer counts (16-20) as they offer the most degrees of freedom
    # for broadband performance. We'll try a few counts if time permits.
    layer_counts = [20, 18, 16]
    
    for n_layers in layer_counts:
        if time.time() > deadline - 10.0: # Leave 10s for final polish
            break
            
        # Try both starting with L and H
        for start_with_L in [True, False]:
            if time.time() > deadline - 10.0:
                break
                
            init_materials = []
            init_thicknesses = []
            for i in range(n_layers):
                if start_with_L:
                    if i % 2 == 0:
                        init_materials.append(n_L)
                        init_thicknesses.append(centre / (4.0 * n_L))
                    else:
                        init_materials.append(n_H)
                        init_thicknesses.append(centre / (4.0 * n_H))
                else:
                    if i % 2 == 0:
                        init_materials.append(n_H)
                        init_thicknesses.append(centre / (4.0 * n_H))
                    else:
                        init_materials.append(n_L)
                        init_thicknesses.append(centre / (4.0 * n_L))
            
            # Scale thicknesses to respect total constraint
            total = sum(init_thicknesses)
            if total > max_total_nm:
                scale = max_total_nm / total
                init_thicknesses = [t * scale for t in init_thicknesses]
            
            # Ensure each thickness is within [0, max_layer_nm]
            init_thicknesses = [min(max(t, 0.0), max_layer_nm) for t in init_thicknesses]
            
            # Define objective function with penalty for constraint violation
            def objective(x, mat=init_materials):
                th = np.clip(x, 0.0, max_layer_nm)
                if np.sum(th) > max_total_nm:
                    return 100.0 + (np.sum(th) - max_total_nm) * 0.1 # Penalty
                return fast_reflectance(mat, th)
            
            bounds = np.array([[0.0, max_layer_nm]] * n_layers)
            
            # Run Simulated Annealing
            # Time allocation: use remaining time minus buffer for polish
            remaining_time = deadline - time.time() - 5.0
            if remaining_time < 1.0:
                continue
                
            # Estimate iterations based on time. 
            # Each iteration is fast (vectorized numpy). 
            # Let's aim for ~10000 iterations per second? No, maybe less due to overhead.
            # Let's just run until time runs out or max_iter.
            max_iter_sa = 50000
            try:
                x_sa, f_sa = _simulated_annealing(objective, np.array(init_thicknesses), bounds, 
                                                  max_iter=max_iter_sa, deadline=deadline - 5.0,
                                                  T_start=50.0, T_end=0.1, noise_scale=5.0)
            except Exception:
                continue
            
            if f_sa < best_score:
                best_score = f_sa
                best_materials = init_materials[:]
                best_thicknesses = list(np.clip(x_sa, 0.0, max_layer_nm))
                
                # Local Polish with Nelder-Mead
                if time.time() < deadline - 2.0:
                    def objective_polish(x):
                        th = np.clip(x, 0.0, max_layer_nm)
                        if np.sum(th) > max_total_nm:
                            return 100.0
                        return fast_reflectance(best_materials, th)
                    
                    bounds_polish = np.array([[0.0, max_layer_nm]] * n_layers)
                    try:
                        x_nm, f_nm = _nelder_mead(objective_polish, np.array(best_thicknesses), bounds_polish, 
                                                   max_iter=200, deadline=deadline - 1.0)
                        if f_nm < best_score:
                            best_score = f_nm
                            best_thicknesses = list(np.clip(x_nm, 0.0, max_layer_nm))
                    except Exception:
                        pass

    # Final constraint check and cleanup
    if best_materials is None:
        # Fallback: single quarter-wave L layer
        return ["L"], [centre / (4.0 * n_L)]
    
    material_labels = []
    for n in best_materials:
        if abs(n - n_L) < 1e-6:
            material_labels.append("L")
        else:
            material_labels.append("H")
    
    # Final constraint check
    total_thickness = sum(best_thicknesses)
    if total_thickness > max_total_nm:
        scale = max_total_nm / total_thickness
        best_thicknesses = [t * scale for t in best_thicknesses]
    
    best_thicknesses = [min(max(t, 0.0), max_layer_nm) for t in best_thicknesses]
    
    return material_labels, best_thicknesses
# EVOLVE-BLOCK-END
