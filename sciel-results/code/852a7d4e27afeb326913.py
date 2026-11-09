"""Antireflection coating design via Nelder-Mead on fixed graded-index sequence."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _reflectance_spectrum(thicknesses, n_L, n_H, n_amb, n_sub, wavelengths):
    """Compute normal-incidence reflectance using characteristic matrix method.
    
    thicknesses: list of physical thicknesses in nm (same order as layers)
    n_L, n_H: indices of low and high index materials
    n_amb, n_sub: ambient and substrate indices
    wavelengths: array of wavelengths in nm
    Returns: array of reflectance values (0 to 1)
    """
    N = len(thicknesses)
    if N == 0:
        # Bare substrate
        r = (n_amb - n_sub) / (n_amb + n_sub)
        return np.full_like(wavelengths, r**2)
    
    # Determine which layers are L and which are H based on the pattern
    # We'll use a fixed alternating pattern starting with L
    # Actually, we need to pass the layer types too. Let me restructure.
    
    # For now, this function assumes we know the layer sequence.
    # Let me create a separate function that takes the full design.
    pass

def _compute_reflectance(layer_types, thicknesses, n_L, n_H, n_amb, n_sub, wavelengths):
    """Compute mean reflectance over the wavelength band.
    
    layer_types: list of 'L' or 'H'
    thicknesses: list of physical thicknesses in nm
    Returns: mean reflectance in percent
    """
    N = len(layer_types)
    n_layers = []
    for lt in layer_types:
        n_layers.append(n_L if lt == 'L' else n_H)
    
    # Convert to arrays
    n_arr = np.array(n_layers, dtype=float)
    d_arr = np.array(thicknesses, dtype=float)
    wl_arr = np.array(wavelengths, dtype=float)
    
    # Optical thicknesses: n * d / wavelength (in radians for phase)
    # Phase delta = 2*pi*n*d/lambda
    # We need to compute the characteristic matrix for each wavelength
    
    # For efficiency, we'll compute for all wavelengths at once
    # M_j = [[cos(delta_j), i*sin(delta_j)/n_j], [i*n_j*sin(delta_j), cos(delta_j)]]
    # where delta_j = 2*pi*n_j*d_j/lambda
    
    # Total matrix M = M_1 * M_2 * ... * M_N
    # Then Y = (m11 + m12*n_sub) / (m21 + m22*n_sub)
    # R = |Y - n_amb|^2 / |Y + n_amb|^2
    
    num_wl = len(wl_arr)
    
    # Initialize total matrix as identity
    m11 = np.ones(num_wl, dtype=complex)
    m12 = np.zeros(num_wl, dtype=complex)
    m21 = np.zeros(num_wl, dtype=complex)
    m22 = np.ones(num_wl, dtype=complex)
    
    for j in range(N):
        n_j = n_arr[j]
        d_j = d_arr[j]
        
        # delta_j = 2*pi*n_j*d_j/lambda
        delta = 2.0 * np.pi * n_j * d_j / wl_arr
        
        cos_d = np.cos(delta)
        sin_d = np.sin(delta)
        
        # Layer matrix elements
        M11 = cos_d
        M12 = 1j * sin_d / n_j
        M21 = 1j * n_j * sin_d
        M22 = cos_d
        
        # Multiply: new_M = old_M * layer_M
        new_m11 = m11 * M11 + m12 * M21
        new_m12 = m11 * M12 + m12 * M22
        new_m21 = m21 * M11 + m22 * M21
        new_m22 = m21 * M12 + m22 * M22
        
        m11 = new_m11
        m12 = new_m12
        m21 = new_m21
        m22 = new_m22
    
    # Admittance
    Y = (m11 + m12 * n_sub) / (m21 + m22 * n_sub)
    
    # Reflectance
    R = np.abs((Y - n_amb) / (Y + n_amb))**2
    
    return np.mean(R) * 100.0

def _nelder_mead(objective, x0, bounds, max_iters=500, tol=1e-8):
    """Simple Nelder-Mead implementation with bounds."""
    n = len(x0)
    # Create initial simplex
    simplex = [x0.copy()]
    for i in range(n):
        point = x0.copy()
        step = 0.1 * (bounds[i][1] - bounds[i][0]) if bounds[i][1] > bounds[i][0] else 0.1
        point[i] += step
        point[i] = np.clip(point[i], bounds[i][0], bounds[i][1])
        simplex.append(point)
    
    # Evaluate objective
    values = [objective(x) for x in simplex]
    
    for it in range(max_iters):
        # Sort by objective value
        order = np.argsort(values)
        simplex = [simplex[i] for i in order]
        values = [values[i] for i in order]
        
        # Check convergence
        if abs(values[-1] - values[0]) < tol:
            break
        
        # Centroid of all but worst
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        alpha = 1.0
        x_r = centroid + alpha * (centroid - simplex[-1])
        x_r = np.clip(x_r, [b[0] for b in bounds], [b[1] for b in bounds])
        f_r = objective(x_r)
        
        if f_r < values[0]:
            # Expansion
            gamma = 2.0
            x_e = centroid + gamma * (x_r - centroid)
            x_e = np.clip(x_e, [b[0] for b in bounds], [b[1] for b in bounds])
            f_e = objective(x_e)
            if f_e < f_r:
                simplex[-1] = x_e
                values[-1] = f_e
            else:
                simplex[-1] = x_r
                values[-1] = f_r
        elif f_r < values[-2]:
            simplex[-1] = x_r
            values[-1] = f_r
        else:
            # Contraction
            rho = 0.5
            x_c = centroid + rho * (simplex[-1] - centroid)
            x_c = np.clip(x_c, [b[0] for b in bounds], [b[1] for b in bounds])
            f_c = objective(x_c)
            if f_c < values[-1]:
                simplex[-1] = x_c
                values[-1] = f_c
            else:
                # Shrink
                for i in range(1, n+1):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    values[i] = objective(simplex[i])
    
    # Return best point
    best_idx = np.argmin(values)
    return simplex[best_idx], values[best_idx]

def design(problem):
    """Design antireflection coating using Nelder-Mead on fixed graded-index sequence."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    n_amb = problem["ambient_index"]
    n_sub = problem["substrate_index"]
    band = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    # Create wavelength grid (401 points from 400 to 800 nm)
    wavelengths = np.linspace(band[0], band[1], 401)
    
    # Try different numbers of layers with alternating L/H pattern
    # Start with a reasonable number, e.g., 8 layers
    best_materials = None
    best_thicknesses = None
    best_score = float('inf')
    
    # Try several layer counts
    for n_layers in [4, 6, 8, 10, 12]:
        if n_layers > max_layers:
            break
        
        if time.time() > deadline - 10.0:
            break
        
        # Create alternating pattern starting with L
        layer_types = ['L' if i % 2 == 0 else 'H' for i in range(n_layers)]
        
        # Initial guess: quarter-wave at center wavelength
        center_wl = 0.5 * (band[0] + band[1])
        initial_thicknesses = []
        for lt in layer_types:
            n_mat = n_L if lt == 'L' else n_H
            d = center_wl / (4.0 * n_mat)
            d = min(d, max_layer_nm)
            initial_thicknesses.append(d)
        
        # Ensure total thickness constraint
        total = sum(initial_thicknesses)
        if total > max_total_nm:
            scale = max_total_nm / total
            initial_thicknesses = [d * scale for d in initial_thicknesses]
        
        # Define objective function
        def objective(thicknesses):
            # Enforce constraints
            thicknesses = np.clip(thicknesses, 0.0, max_layer_nm)
            if np.sum(thicknesses) > max_total_nm:
                scale = max_total_nm / np.sum(thicknesses)
                thicknesses = thicknesses * scale
            return _compute_reflectance(layer_types, thicknesses, n_L, n_H, n_amb, n_sub, wavelengths)
        
        # Set bounds
        bounds = [(0.0, max_layer_nm)] * n_layers
        
        # Run Nelder-Mead
        x0 = np.array(initial_thicknesses)
        
        # Limit iterations based on time
        max_iters = 200
        x_opt, f_opt = _nelder_mead(objective, x0, bounds, max_iters=max_iters)
        
        # Enforce constraints on final result
        x_opt = np.clip(x_opt, 0.0, max_layer_nm)
        if np.sum(x_opt) > max_total_nm:
            scale = max_total_nm / np.sum(x_opt)
            x_opt = x_opt * scale
        
        score = objective(x_opt)
        
        if score < best_score:
            best_score = score
            best_materials = layer_types
            best_thicknesses = [float(d) for d in x_opt]
    
    # If no valid result found, return a simple design
    if best_materials is None:
        center_wl = 0.5 * (band[0] + band[1])
        d = center_wl / (4.0 * n_L)
        return ["L"], [min(d, max_layer_nm)]
    
    return best_materials, best_thicknesses
# EVOLVE-BLOCK-END
