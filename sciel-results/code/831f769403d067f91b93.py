"""Design broadband antireflection coating using vectorized reflectance + Nelder-Mead."""
import numpy as np
import time

def _reflectance_vectorized(thicknesses, materials, n_layers, ambient_index, substrate_index, wavelengths):
    """
    Compute normal-incidence reflectance for all wavelengths simultaneously.
    thicknesses: array of length n_layers (nm)
    materials: array of refractive indices for each layer
    wavelengths: array of wavelengths (nm)
    Returns: reflectance array (fraction, not percent)
    """
    # Optical thickness in radians: delta = 2*pi*n*d/lambda
    # For each layer i and wavelength j: delta[i,j]
    d = thicknesses[np.newaxis, :]  # (1, n_layers)
    n = materials[np.newaxis, :]     # (1, n_layers)
    wl = wavelengths[:, np.newaxis]  # (n_wl, 1)
    
    delta = 2.0 * np.pi * n * d / wl  # (n_wl, n_layers)
    
    # Build characteristic matrices for each layer at each wavelength
    # M_i = [[cos(delta_i), i*sin(delta_i)/n_i],
    #        [i*n_i*sin(delta_i), cos(delta_i)]]
    cos_d = np.cos(delta)   # (n_wl, n_layers)
    sin_d = np.sin(delta)   # (n_wl, n_layers)
    
    # For each wavelength, multiply all layer matrices
    # We'll do this iteratively to save memory
    # Start with identity for each wavelength
    # M_total = M_1 * M_2 * ... * M_n
    # M[i] = [[a, b], [c, d]]
    
    a = np.ones((len(wavelengths), 1))  # (n_wl, 1)
    b = np.zeros((len(wavelengths), 1))
    c = np.zeros((len(wavelengths), 1))
    d_mat = np.ones((len(wavelengths), 1))
    
    for i in range(n_layers):
        cos_i = cos_d[:, i]  # (n_wl,)
        sin_i = sin_d[:, i]
        n_i = materials[i]
        
        # New a = a*cos_i + b*i*n_i*sin_i  (since b term: b * (i*sin_i/n_i) * ... wait let me redo)
        # M_i = [[cos_i, i*sin_i/n_i], [i*n_i*sin_i, cos_i]]
        # M_new = M_old * M_i
        # a_new = a*cos_i + b*(i*n_i*sin_i)
        # b_new = a*(i*sin_i/n_i) + b*cos_i
        # c_new = c*cos_i + d_mat*(i*n_i*sin_i)
        # d_new = c*(i*sin_i/n_i) + d_mat*cos_i
        
        a_new = a * cos_i + b * (1j * n_i * sin_i)
        b_new = a * (1j * sin_i / n_i) + b * cos_i
        c_new = c * cos_i + d_mat * (1j * n_i * sin_i)
        d_new = c * (1j * sin_i / n_i) + d_mat * cos_i
        
        a = a_new
        b = b_new
        c = c_new
        d_mat = d_new
    
    # Admittance: Y = (c + d_mat) / (a + b)
    # Reflectance: R = |1 - Y| / |1 + Y|  ... wait, need to be careful with the admittance
    # Actually, the characteristic matrix method:
    # [E, H] at air side = M_total * [E, H] at substrate
    # The admittance Y = H/E at the front surface
    # Y = (M21 + M22) / (M11 + M12)  where M = [[M11, M12], [M21, M22]]
    # In our notation: a=M11, b=M12, c=M21, d=M22
    # Y = (c + d_mat) / (a + b)
    # Reflectance: R = |n_ambient - Y| / |n_ambient + Y|  ... 
    # Actually R = |(n_ambient*Y - 1)/(n_ambient*Y + 1)|^2? Let me think again.
    # The standard formula: r = (Y - n_sub) / (Y + n_sub) where Y is the input admittance
    # Wait, Y is normalized. Let me use: r = (n_amb - Y) / (n_amb + Y) where Y is the admittance
    # Actually the admittance Y = H/E, and the reflection coefficient is:
    # r = (n_amb - Y) / (n_amb + Y)  if we normalize by the ambient
    # Hmm, let me be more careful. The characteristic matrix relates:
    # [E0, H0] = M * [Es, Hs]
    # At the substrate, Hs = n_sub * Es (for normal incidence, traveling wave)
    # So [E0, H0] = M * [Es, n_sub*Es] = Es * [M11 + M12*n_sub, M21 + M22*n_sub]
    # The input admittance Y0 = H0/E0 = (M21 + M22*n_sub) / (M11 + M12*n_sub)
    # The reflection coefficient at the air interface: r = (n_amb - Y0) / (n_amb + Y0)
    # Wait, that's not right either. Let me look this up mentally.
    # The reflection coefficient is: r = (n_amb * Y0 - 1) / (n_amb * Y0 + 1)
    # No... Let me think differently.
    # Y0 = H0/E0 is the effective admittance. The reflection coefficient for a wave incident from medium n_amb onto an interface with admittance Y0 is:
    # r = (n_amb - Y0) / (n_amb + Y0)
    # Hmm, actually I think it's r = (1 - Y0/n_amb) / (1 + Y0/n_amb) = (n_amb - Y0)/(n_amb + Y0)
    # Let me just use: r = (n_amb - Y0) / (n_amb + Y0)
    
    Y0 = (c + d_mat * substrate_index) / (a + b * substrate_index)
    r = (ambient_index - Y0) / (ambient_index + Y0)
    R = np.abs(r) ** 2
    return R

def _mean_reflectance_percent(thicknesses, materials_arr, n_layers, ambient_index, substrate_index, wavelengths):
    """Compute mean reflectance in percent."""
    R = _reflectance_vectorized(thicknesses, materials_arr, n_layers, ambient_index, substrate_index, wavelengths)
    return np.mean(R) * 100.0

def design(problem):
    """Design broadband antireflection coating using Nelder-Mead optimization."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    ambient_index = problem["ambient_index"]
    substrate_index = problem["substrate_index"]
    materials_dict = problem["materials"]
    band_nm = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    n_L = materials_dict["L"]
    n_H = materials_dict["H"]
    
    # Wavelengths for evaluation
    wavelengths = np.linspace(band_nm[0], band_nm[1], 401)
    
    # We'll search over a fixed number of layers. Let's try different numbers.
    # Start with a reasonable number, say 10 layers, and optimize thicknesses.
    
    best_score = float('inf')
    best_result = None
    
    # Try several layer counts
    for n_layers in [4, 6, 8, 10, 12, 16, 20]:
        if time.time() > deadline - 30.0:
            break
        
        # Initial guess: quarter-wave stack
        centre = 0.5 * (band_nm[0] + band_nm[1])
        
        # Alternate L and H, starting with L
        thicknesses = np.zeros(n_layers)
        materials_arr = np.zeros(n_layers)
        for i in range(n_layers):
            if i % 2 == 0:
                materials_arr[i] = n_L
                thicknesses[i] = centre / (4.0 * n_L)
            else:
                materials_arr[i] = n_H
                thicknesses[i] = centre / (4.0 * n_H)
        
        # Clamp to valid range
        thicknesses = np.clip(thicknesses, 0, max_layer_nm)
        
        # Nelder-Mead optimization
        from scipy.optimize import minimize
        
        def objective(x):
            x = np.clip(x, 0, max_layer_nm)
            if np.sum(x) > max_total_nm:
                return 100.0  # Penalty
            return _mean_reflectance_percent(x, materials_arr, n_layers, ambient_index, substrate_index, wavelengths)
        
        # Try a few random starts
        rng = np.random.RandomState(42 + n_layers)
        for start_trial in range(3):
            if time.time() > deadline - 20.0:
                break
            
            if start_trial == 0:
                x0 = thicknesses.copy()
            else:
                x0 = rng.uniform(0, max_layer_nm, n_layers)
            
            # Scale variables to [0,1] for better Nelder-Mead behavior
            x0_scaled = x0 / max_layer_nm
            
            def objective_scaled(x):
                x = np.clip(x * max_layer_nm, 0, max_layer_nm)
                if np.sum(x) > max_total_nm:
                    return 100.0
                return _mean_reflectance_percent(x, materials_arr, n_layers, ambient_index, substrate_index, wavelengths)
            
            result = minimize(
                objective_scaled,
                x0_scaled,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
            )
            
            x_opt = np.clip(result.x * max_layer_nm, 0, max_layer_nm)
            if np.sum(x_opt) > max_total_nm:
                # Scale down
                x_opt = x_opt * (max_total_nm / np.sum(x_opt))
            
            score = _mean_reflectance_percent(x_opt, materials_arr, n_layers, ambient_index, substrate_index, wavelengths)
            
            if score < best_score:
                best_score = score
                best_result = (n_layers, materials_arr.copy(), x_opt.copy())
    
    if best_result is None:
        # Fallback: single quarter-wave L layer
        centre = 0.5 * (band_nm[0] + band_nm[1])
        return ["L"], [centre / (4.0 * n_L)]
    
    n_layers, materials_arr, thicknesses = best_result
    labels = []
    for i in range(n_layers):
        if materials_arr[i] == n_L:
            labels.append("L")
        else:
            labels.append("H")
    
    return labels, thicknesses.tolist()
