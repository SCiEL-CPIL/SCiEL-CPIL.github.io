"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def _compute_reflectance(materials, thicknesses, n_ambient, n_substrate, wavelengths):
    """
    Compute reflectance (fraction, 0-1) using characteristic matrix method.
    materials: list of refractive indices
    thicknesses: list of physical thicknesses in nm
    n_ambient: ambient index
    n_substrate: substrate index
    wavelengths: array of wavelengths in nm
    Returns: array of reflectance values
    """
    n_layers = len(materials)
    if n_layers == 0:
        r = (n_ambient - n_substrate) / (n_ambient + n_substrate)
        return np.full_like(wavelengths, abs(r)**2)
    
    # Characteristic matrix for each layer
    # M_j = [[cos(delta_j), i*sin(delta_j)/n_j], [i*n_j*sin(delta_j), cos(delta_j)]]
    # where delta_j = 2*pi*n_j*d_j / lambda
    
    # We compute the product M = M_1 @ M_2 @ ... @ M_N
    # Then the admittance Y = [M21, M22] @ [1, n_substrate]^T ... wait.
    # Standard formulation:
    # [B, C]^T = M * [1, n_s]^T
    # Y = C/B
    # R = |(n_0 - Y)/(n_0 + Y)|^2
    
    # Let's build the total matrix M_total
    # M_total is 2x2 complex array for each wavelength
    
    # Initialize M_total = I
    M_total = np.zeros((2, 2, len(wavelengths)), dtype=complex)
    M_total[0, 0, :] = 1.0
    M_total[1, 1, :] = 1.0
    
    for i in range(n_layers):
        n_i = materials[i]
        d_i = thicknesses[i]
        
        # delta = 2*pi*n*d/lambda
        delta = 2.0 * np.pi * n_i * d_i / wavelengths
        
        c = np.cos(delta)
        s = np.sin(delta)
        
        # M_i = [[c, i*s/n_i], [i*n_i*s, c]]
        M_i = np.zeros((2, 2, len(wavelengths)), dtype=complex)
        M_i[0, 0, :] = c
        M_i[0, 1, :] = 1j * s / n_i
        M_i[1, 0, :] = 1j * n_i * s
        M_i[1, 1, :] = c
        
        # M_total = M_total @ M_i
        # Since these are 2x2 matrices, we can do it element-wise for each wavelength
        # M_new[0,0] = M[0,0]*M_i[0,0] + M[0,1]*M_i[1,0]
        # M_new[0,1] = M[0,0]*M_i[0,1] + M[0,1]*M_i[1,1]
        # M_new[1,0] = M[1,0]*M_i[0,0] + M[1,1]*M_i[1,0]
        # M_new[1,1] = M[1,0]*M_i[0,1] + M[1,1]*M_i[1,1]
        
        M00 = M_total[0, 0, :] * M_i[0, 0, :] + M_total[0, 1, :] * M_i[1, 0, :]
        M01 = M_total[0, 0, :] * M_i[0, 1, :] + M_total[0, 1, :] * M_i[1, 1, :]
        M10 = M_total[1, 0, :] * M_i[0, 0, :] + M_total[1, 1, :] * M_i[1, 0, :]
        M11 = M_total[1, 0, :] * M_i[0, 1, :] + M_total[1, 1, :] * M_i[1, 1, :]
        
        M_total[0, 0, :] = M00
        M_total[0, 1, :] = M01
        M_total[1, 0, :] = M10
        M_total[1, 1, :] = M11
        
    # Now compute B and C
    # [B, C]^T = M_total * [1, n_s]^T
    # B = M[0,0] * 1 + M[0,1] * n_s
    # C = M[1,0] * 1 + M[1,1] * n_s
    B = M_total[0, 0, :] + M_total[0, 1, :] * n_substrate
    C = M_total[1, 0, :] + M_total[1, 1, :] * n_substrate
    
    # Y = C / B
    Y = C / B
    
    # R = |(n_0 - Y)/(n_0 + Y)|^2
    R = np.abs((n_ambient - Y) / (n_ambient + Y))**2
    
    return R

def _fast_reflectance(materials, thicknesses, n_ambient, n_substrate, wavelengths):
    """Optimized version for speed."""
    n_layers = len(materials)
    if n_layers == 0:
        r = (n_ambient - n_substrate) / (n_ambient + n_substrate)
        return np.full_like(wavelengths, abs(r)**2)
    
    M00 = np.ones(len(wavelengths), dtype=complex)
    M01 = np.zeros(len(wavelengths), dtype=complex)
    M10 = np.zeros(len(wavelengths), dtype=complex)
    M11 = np.ones(len(wavelengths), dtype=complex)
    
    for i in range(n_layers):
        n_i = materials[i]
        d_i = thicknesses[i]
        
        if d_i == 0.0:
            continue
            
        delta = 2.0 * np.pi * n_i * d_i / wavelengths
        c = np.cos(delta)
        s = np.sin(delta)
        
        # M_i components
        m00 = c
        m01 = 1j * s / n_i
        m10 = 1j * n_i * s
        m11 = c
        
        # Matrix multiplication
        n00 = M00 * m00 + M01 * m10
        n01 = M00 * m01 + M01 * m11
        n10 = M10 * m00 + M11 * m10
        n11 = M10 * m01 + M11 * m11
        
        M00 = n00
        M01 = n01
        M10 = n10
        M11 = n11
        
    B = M00 + M01 * n_substrate
    C = M10 + M11 * n_substrate
    Y = C / B
    R = np.abs((n_ambient - Y) / (n_ambient + Y))**2
    return R

def design(problem):
    n_ambient = problem["ambient_index"]
    n_substrate = problem["substrate_index"]
    materials_dict = problem["materials"]
    n_L = materials_dict["L"]
    n_H = materials_dict["H"]
    band = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    # Wavelengths for evaluation (401 points)
    wavelengths = np.linspace(band[0], band[1], 401)
    
    # Objective function
    def objective(materials_labels, thicknesses):
        mats = [materials_dict[l] for l in materials_labels]
        R = _fast_reflectance(mats, thicknesses, n_ambient, n_substrate, wavelengths)
        return np.mean(R) * 100.0 # in percent
    
    # Start with a simple multi-layer stack
    # A common broadband AR coating structure is L-H-L or H-L-H etc.
    # Let's try to optimize a small stack, e.g., 3 or 4 layers.
    
    # We will use a local optimization (Nelder-Mead or similar) on thicknesses for a fixed structure.
    # First, let's pick a structure.
    # Structure: L H L H L (5 layers) or just L H L (3 layers)
    # Let's try a 5-layer stack: L H L H L
    # Or maybe a 4-layer: H L H L
    
    # Let's start with a 3-layer stack: L H L
    # Initial guess: quarter-wave at center wavelength
    center_wl = 0.5 * (band[0] + band[1])
    t_L_qw = center_wl / (4.0 * n_L)
    t_H_qw = center_wl / (4.0 * n_H)
    
    # Structure: L H L
    init_labels = ["L", "H", "L"]
    init_thick = [t_L_qw, t_H_qw, t_L_qw]
    
    # Check constraints
    if sum(init_thick) > max_total_nm:
        # Scale down? Or just pick a simpler structure
        pass
        
    # Let's try to optimize this structure
    # We can use scipy.optimize.minimize if available, or a simple gradient-free method.
    # Since we have a 60s budget, we can afford a few hundred iterations.
    
    # Let's try a 5-layer stack for better broadband performance.
    # Structure: L H L H L
    init_labels_5 = ["L", "H", "L", "H", "L"]
    init_thick_5 = [t_L_qw, t_H_qw, t_L_qw, t_H_qw, t_L_qw]
    
    if sum(init_thick_5) <= max_total_nm:
        labels = init_labels_5
        init_thick = init_thick_5
    else:
        labels = init_labels
        init_thick = init_thick
    
    # Optimization
    # We'll use a simple Nelder-Mead implementation or just use scipy if available.
    # Let's assume scipy is available.
    try:
        from scipy.optimize import minimize
    except ImportError:
        # Fallback to a simple coordinate descent or just return initial
        return labels, init_thick
        
    def obj_func(x):
        # x is the vector of thicknesses
        if np.any(x < 0) or np.any(x > max_layer_nm) or np.sum(x) > max_total_nm:
            return 1e6
        return objective(labels, x)
        
    # Bounds
    bounds = [(0, max_layer_nm)] * len(labels)
    
    # Initial point
    x0 = np.array(init_thick)
    
    # Check if x0 is valid
    if np.sum(x0) > max_total_nm:
        # Scale down
        scale = max_total_nm / np.sum(x0)
        x0 = x0 * scale * 0.99
        
    # Run optimization
    result = minimize(obj_func, x0, method='Nelder-Mead', bounds=bounds, options={'maxiter': 1000, 'xatol': 0.1, 'fatol': 1e-5})
    
    best_thick = result.x
    best_score = result.fun
    
    # Ensure validity
    if np.any(best_thick < 0) or np.any(best_thick > max_layer_nm) or np.sum(best_thick) > max_total_nm:
        # Fallback to initial
        best_thick = x0
        best_score = obj_func(x0)
        
    # Clip to bounds
    best_thick = np.clip(best_thick, 0, max_layer_nm)
    if np.sum(best_thick) > max_total_nm:
        best_thick = best_thick * (max_total_nm / np.sum(best_thick))
        
    return labels, list(best_thick)

# EVOLVE-BLOCK-END
