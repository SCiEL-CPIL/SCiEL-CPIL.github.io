"""Broadband antireflection coating design using differential evolution + Nelder-Mead polish."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

def _reflectance_spectrum(materials, thicknesses, band_nm, ambient_index, substrate_index, n_points=401):
    """Compute mean normal-incidence reflectance (percent) using characteristic-matrix method."""
    wavelengths = np.linspace(band_nm[0], band_nm[1], n_points)
    k0 = 2.0 * np.pi / wavelengths  # 1/nm
    
    n_L = materials['L']
    n_H = materials['H']
    
    n_layers = len(materials)
    # Build the index array for each layer
    n_arr = np.array([materials[m] for m in materials])
    d_arr = np.array(thicknesses)
    
    # Characteristic matrices for each layer
    # For lossless media: M_i = [[cos(delta_i), i*sin(delta_i)/n_i], [i*n_i*sin(delta_i), cos(delta_i)]]
    # where delta_i = 2*pi*n_i*d_i/lambda
    
    # We'll compute the product matrix M = M_1 * M_2 * ... * M_N
    # Then Y = (m11 + m12*n_s) / (m21 + m22*n_s)  where n_s is substrate index
    # R = |(n_0 - Y)/(n_0 + Y)|^2  where n_0 is ambient index
    
    # Precompute for all wavelengths
    # delta_i = k0 * n_i * d_i  (since k0 = 2*pi/lambda, so delta = 2*pi*n*d/lambda = k0*n*d)
    
    # For efficiency, compute layer by layer
    # M_total starts as identity
    M = np.eye(2, dtype=complex)
    
    for i in range(n_layers):
        n_i = n_arr[i]
        d_i = d_arr[i]
        delta = k0 * n_i * d_i  # shape: (n_points,)
        
        cos_d = np.cos(delta)
        sin_d = np.sin(delta)
        
        # M_i = [[cos_d, 1j*sin_d/n_i], [1j*n_i*sin_d, cos_d]]
        m11 = cos_d
        m12 = 1j * sin_d / n_i
        m21 = 1j * n_i * sin_d
        m22 = cos_d
        
        # M = M * M_i  (M is 2x2 with elements being arrays of shape n_points)
        # We need to handle this carefully. M is a 2x2 array where each element is a 1D array.
        m11_new = m11 * M[0,0] + m12 * M[1,0]
        m12_new = m11 * M[0,1] + m12 * M[1,1]
        m21_new = m21 * M[0,0] + m22 * M[1,0]
        m22_new = m21 * M[0,1] + m22 * M[1,1]
        
        M[0,0] = m11_new
        M[0,1] = m12_new
        M[1,0] = m21_new
        M[1,1] = m22_new
    
    # Now compute Y (input admittance)
    n_s = substrate_index
    Y = (M[0,0] + M[0,1] * n_s) / (M[1,0] + M[1,1] * n_s)
    
    n_0 = ambient_index
    R = np.abs((n_0 - Y) / (n_0 + Y)) ** 2
    
    return 100.0 * np.mean(R)

def design(problem):
    """Design a broadband antireflection coating using DE + Nelder-Mead."""
    t_start = time.time()
    deadline = t_start + 140.0  # safety margin before 150s kill
    
    ambient_index = problem["ambient_index"]
    substrate_index = problem["substrate_index"]
    materials = problem["materials"]
    band_nm = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    n_L = materials['L']
    n_H = materials['H']
    
    # We'll use a fixed number of layers for the optimization.
    # Let's try 10 layers first, which is a good balance.
    n_layers = 10
    
    # Define the objective function for optimization
    def objective(x):
        """x is a flat array of thicknesses. Returns mean reflectance."""
        # Check constraints
        if np.any(x < -1e-9) or np.any(x > max_layer_nm + 1e-9):
            return 100.0
        x_clipped = np.clip(x, 0.0, max_layer_nm)
        if np.sum(x_clipped) > max_total_nm + 1e-9:
            return 100.0
        
        # Create materials list
        mat_list = []
        # Alternating L, H starting with L
        for i in range(n_layers):
            if i % 2 == 0:
                mat_list.append('L')
            else:
                mat_list.append('H')
        
        return _reflectance_spectrum(mat_list, x_clipped, band_nm, ambient_index, substrate_index)
    
    # Initial guess: quarter-wave stack
    centre_wl = 0.5 * (band_nm[0] + band_nm[1])
    x0 = np.zeros(n_layers)
    for i in range(n_layers):
        if i % 2 == 0:
            x0[i] = centre_wl / (4.0 * n_L)
        else:
            x0[i] = centre_wl / (4.0 * n_H)
    
    # Ensure total thickness constraint
    total = np.sum(x0)
    if total > max_total_nm:
        x0 *= max_total_nm / total
    
    # Define bounds for DE
    bounds = [(0.0, max_layer_nm) for _ in range(n_layers)]
    
    # Run differential evolution
    best_result = None
    best_score = 100.0
    
    try:
        de_result = differential_evolution(
            objective, bounds,
            maxiter=50,
            popsize=15,
            tol=1e-6,
            seed=42,
            polish=False,
            init='latinhypercube'
        )
        
        if de_result.fun < best_score:
            best_score = de_result.fun
            best_result = de_result.x.copy()
        
        # Check time
        if time.time() > deadline - 20.0:
            # Not enough time for polish, return DE result
            x_final = np.clip(best_result, 0.0, max_layer_nm)
            if np.sum(x_final) > max_total_nm:
                x_final *= max_total_nm / np.sum(x_final)
            mat_list = []
            for i in range(n_layers):
                if i % 2 == 0:
                    mat_list.append('L')
                else:
                    mat_list.append('H')
            return mat_list, list(x_final)
        
        # Nelder-Mead polish
        nm_result = minimize(
            objective, best_result,
            method='Nelder-Mead',
            bounds=bounds,
            options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8}
        )
        
        if nm_result.fun < best_score:
            best_score = nm_result.fun
            best_result = nm_result.x.copy()
    
    except Exception:
        pass
    
    # Final result
    x_final = np.clip(best_result, 0.0, max_layer_nm)
    if np.sum(x_final) > max_total_nm:
        x_final *= max_total_nm / np.sum(x_final)
    
    mat_list = []
    for i in range(n_layers):
        if i % 2 == 0:
            mat_list.append('L')
        else:
            mat_list.append('H')
    
    return mat_list, list(x_final)
# EVOLVE-BLOCK-END
