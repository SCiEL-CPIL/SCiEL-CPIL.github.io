import numpy as np
import scipy.optimize
import time

def _calc_R_vec(thicknesses, labels, materials, n_sub, n_amb, wavelengths):
    """
    Vectorized calculation of reflectance for a stack of layers.
    thicknesses: list of floats
    labels: list of 'L' or 'H'
    materials: dict {'L': nL, 'H': nH}
    n_sub: substrate index
    n_amb: ambient index
    wavelengths: array of floats
    Returns: array of reflectance values
    """
    n_layers = len(thicknesses)
    n_w = len(wavelengths)
    
    # Initialize M_total to identity matrix for all wavelengths
    # Shape: (2, 2, n_w)
    M_total = np.zeros((2, 2, n_w), dtype=np.complex128)
    M_total[0, 0, :] = 1.0
    M_total[1, 1, :] = 1.0
    
    # Iterate through layers
    for k in range(n_layers):
        n_k = materials[labels[k]]
        d_k = thicknesses[k]
        
        # delta = 2 * pi * n_k * d_k / lambda
        # Handle division by zero if any wavelength is 0 (not expected)
        delta = 2.0 * np.pi * n_k * d_k / wavelengths
        
        cos_d = np.cos(delta)
        sin_d = np.sin(delta)
        
        # M_k = [[cos_d, i/n_k * sin_d], [i*n_k*sin_d, cos_d]]
        # Shape: (2, 2, n_w)
        M_k = np.zeros((2, 2, n_w), dtype=np.complex128)
        M_k[0, 0, :] = cos_d
        M_k[0, 1, :] = (1.0j / n_k) * sin_d
        M_k[1, 0, :] = (1.0j * n_k) * sin_d
        M_k[1, 1, :] = cos_d
        
        # M_total = M_k @ M_total
        # Broadcasting: (2, 2, n_w) @ (2, 2, n_w)
        # We need to sum over the middle index.
        # M_total_new[i, j, w] = sum_l M_k[i, l, w] * M_total[l, j, w]
        M_total = np.einsum('ilw,ljw->ijw', M_k, M_total)
        
    # Calculate Y_in
    # Y_in = (M21 + M22 * n_sub) / (M11 + M12 * n_sub)
    M11 = M_total[0, 0, :]
    M12 = M_total[0, 1, :]
    M21 = M_total[1, 0, :]
    M22 = M_total[1, 1, :]
    
    Y_in = (M21 + M22 * n_sub) / (M11 + M12 * n_sub)
    
    # R = | (Y_in - n_amb) / (Y_in + n_amb) |^2
    r = (Y_in - n_amb) / (Y_in + n_amb)
    R = np.abs(r)**2
    
    return R

def _objective(thicknesses, labels, materials, n_sub, n_amb, wavelengths):
    """
    Objective function: Mean Reflectance + Penalty for total thickness.
    """
    R = _calc_R_vec(thicknesses, labels, materials, n_sub, n_amb, wavelengths)
    mean_R = np.mean(R)
    
    # Penalty for total thickness > 1000 nm
    total_t = np.sum(thicknesses)
    if total_t > 1000.0:
        penalty = (total_t - 1000.0) * 0.1
    else:
        penalty = 0.0
        
    return mean_R + penalty

def design(problem):
    """
    Design a broadband antireflection coating.
    """
    start_time = time.time()
    deadline = start_time + 120.0 # 120s limit, leave margin
    
    n_amb = problem['ambient_index']
    n_sub = problem['substrate_index']
    materials = problem['materials']
    band = problem['band_nm']
    max_layers = problem['max_layers']
    max_total_nm = problem['max_total_nm']
    max_layer_nm = problem['max_layer_nm']
    
    # Optimization wavelengths (fewer than evaluation for speed)
    # 400 to 800, step 10nm -> 41 points
    opt_wavelengths = np.arange(band[0], band[1] + 1, 10.0)
    
    # Candidate sequences (symmetric)
    # Lengths: 4, 6, 8, 10
    sequences = [
        ['L', 'H', 'H', 'L'],
        ['L', 'H', 'H', 'H', 'H', 'L'],
        ['L', 'H', 'H', 'L', 'L', 'H', 'H', 'L'],
        ['L', 'H', 'H', 'H', 'H', 'L', 'L', 'H', 'H', 'H', 'H', 'L'],
        ['L', 'H', 'H', 'L', 'L', 'H', 'H', 'H', 'H', 'L', 'L', 'H', 'H', 'L'],
        ['L', 'H', 'H', 'H', 'H', 'L', 'L', 'H', 'H', 'H', 'H', 'L', 'L', 'H', 'H', 'H', 'H', 'L']
    ]
    
    best_score = float('inf')
    best_materials = None
    best_thicknesses = None
    
    center_wl = 0.5 * (band[0] + band[1])
    
    for seq in sequences:
        if time.time() > deadline - 10: # Check time
            break
            
        n_layers = len(seq)
        if n_layers > max_layers:
            continue
            
        # Initial guess: Quarter-wave at center wavelength
        init_thicknesses = []
        for label in seq:
            n = materials[label]
            d = center_wl / (4.0 * n)
            # Clamp to [0, max_layer_nm]
            d = np.clip(d, 0.0, max_layer_nm)
            init_thicknesses.append(d)
            
        # Bounds for optimization
        bounds = [(0.0, max_layer_nm) for _ in range(n_layers)]
        
        # Optimize
        try:
            result = scipy.optimize.minimize(
                _objective,
                init_thicknesses,
                args=(seq, materials, n_sub, n_amb, opt_wavelengths),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-8}
            )
            
            if result.success or result.fun < best_score:
                # Verify constraints
                t_opt = result.x
                if np.sum(t_opt) <= max_total_nm + 1e-6 and all(0 <= t <= max_layer_nm for t in t_opt):
                    # Final check with full resolution (401 points)
                    eval_wavelengths = np.linspace(band[0], band[1], 401)
                    R_final = _calc_R_vec(t_opt, seq, materials, n_sub, n_amb, eval_wavelengths)
                    final_score = np.mean(R_final)
                    
                    if final_score < best_score:
                        best_score = final_score
                        best_materials = seq
                        best_thicknesses = t_opt
                        
        except Exception:
            pass

    # If no valid solution found, return a simple valid one
    if best_materials is None:
        best_materials = ['L']
        best_thicknesses = [center_wl / (4.0 * materials['L'])]
        
    # Ensure return types are lists of floats/strings
    return best_materials, [float(t) for t in best_thicknesses]

