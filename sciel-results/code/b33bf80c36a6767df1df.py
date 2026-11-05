import time
import numpy as np
from scipy.optimize import minimize

def _reflectance(thicknesses, materials, n_ambient, n_substrate, wavelengths):
    """
    Calculate reflectance using the characteristic matrix method.
    thicknesses: list of physical thicknesses in nm
    materials: list of refractive indices
    n_ambient: ambient refractive index
    n_substrate: substrate refractive index
    wavelengths: array of wavelengths in nm
    Returns: array of reflectance values (0 to 1)
    """
    n_layers = len(thicknesses)
    if n_layers == 0:
        # Bare substrate
        r = (n_ambient - n_substrate) / (n_ambient + n_substrate)
        return np.full_like(wavelengths, np.abs(r)**2)
    
    # Pre-calculate phase factors for each layer and wavelength
    # M_i = [[cos(delta_i), i*sin(delta_i)/n_i], [i*n_i*sin(delta_i), cos(delta_i)]]
    # For normal incidence, characteristic matrix is 2x2.
    
    # We can compute the total characteristic matrix product.
    # M_total = M_1 * M_2 * ... * M_N
    # Y = M_total[0,1] / M_total[1,1]
    # R = |(n_ambient - Y) / (n_ambient + Y)|^2
    
    # To be efficient, we can compute M_total for all wavelengths at once.
    # M_i is a 2x2 matrix with elements depending on wavelength.
    
    # Let's build the matrix product.
    # Initialize M_total as identity for each wavelength.
    # M_total shape: (num_wavelengths, 2, 2)
    
    num_w = len(wavelengths)
    # Identity matrix
    m_total = np.zeros((num_w, 2, 2), dtype=np.complex128)
    m_total[:, 0, 0] = 1.0
    m_total[:, 1, 1] = 1.0
    
    for i in range(n_layers):
        n_i = materials[i]
        d_i = thicknesses[i]
        
        # delta_i = 2 * pi * n_i * d_i / lambda
        delta = 2.0 * np.pi * n_i * d_i / wavelengths
        
        c = np.cos(delta)
        s = np.sin(delta)
        
        # M_i = [[c, i*s/n_i], [i*n_i*s, c]]
        m_i = np.zeros((num_w, 2, 2), dtype=np.complex128)
        m_i[:, 0, 0] = c
        m_i[:, 0, 1] = 1j * s / n_i
        m_i[:, 1, 0] = 1j * n_i * s
        m_i[:, 1, 1] = c
        
        # M_total = M_total @ M_i
        # Using einsum for batched matrix multiplication
        # m_total is (N, 2, 2), m_i is (N, 2, 2)
        m_total = np.einsum('nij,njk->nik', m_total, m_i)
        
    # Y = M_total[0,1] / M_total[1,1]
    y = m_total[:, 0, 1] / m_total[:, 1, 1]
    
    # R = |(n_ambient - y) / (n_ambient + y)|^2
    r = (n_ambient - y) / (n_ambient + y)
    reflectance = np.abs(r)**2
    
    return reflectance

def design(problem):
    start_time = time.time()
    deadline = start_time + 50.0  # 50s CPU budget, leave margin
    
    n_ambient = problem["ambient_index"]
    n_substrate = problem["substrate_index"]
    mat_L = problem["materials"]["L"]
    mat_H = problem["materials"]["H"]
    band = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    wavelengths = np.linspace(band[0], band[1], 401)
    
    # Define candidate sequences
    # A common effective structure is alternating L and H, starting with L.
    # Let's try a few sequences with different numbers of layers.
    sequences = [
        ['L', 'H', 'L', 'H', 'L', 'H'],
        ['L', 'H', 'L', 'H', 'L', 'H', 'L', 'H'],
        ['L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H'],
        ['H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L']
    ]
    
    best_score = float('inf')
    best_materials = None
    best_thicknesses = None
    
    # Initial guess: quarter-wave stack centered at 600 nm
    center_wl = 0.5 * (band[0] + band[1])
    
    for seq in sequences:
        if len(seq) > max_layers:
            continue
            
        n_layers = len(seq)
        materials = [mat_L if m == 'L' else mat_H for m in seq]
        
        # Initial thicknesses: quarter-wave at center wavelength
        x0 = np.array([center_wl / (4.0 * n) for n in materials])
        
        # Objective function
        def objective(x):
            if time.time() > deadline:
                return best_score # Return current best to stop optimization
            
            # Check constraints
            if np.any(x < 0) or np.any(x > max_layer_nm):
                return 1.0 # Penalty
            if np.sum(x) > max_total_nm:
                return 1.0 # Penalty
            
            # Compute reflectance
            R = _reflectance(x, materials, n_ambient, n_substrate, wavelengths)
            mean_R = np.mean(R)
            
            return mean_R * 100.0 # Score is in percent
        
        # Run optimization
        try:
            # Nelder-Mead is robust for small dimensions
            # maxiter and maxfev to control time
            result = minimize(
                objective, 
                x0, 
                method='Nelder-Mead', 
                options={
                    'maxiter': 200, 
                    'maxfev': 300, 
                    'xatol': 1e-4, 
                    'fatol': 1e-6
                }
            )
            
            if result.success or result.fun < best_score:
                x_opt = result.x
                # Ensure validity
                if np.all(x_opt >= 0) and np.all(x_opt <= max_layer_nm) and np.sum(x_opt) <= max_total_nm:
                    score = result.fun
                    if score < best_score:
                        best_score = score
                        best_materials = seq
                        best_thicknesses = x_opt
        except Exception:
            pass
        
        if time.time() > deadline:
            break
    
    # If no valid result found, fallback to seed
    if best_materials is None:
        return ["L"], [center_wl / (4.0 * mat_L)]
    
    # Round thicknesses to reasonable precision
    best_thicknesses = [float(t) for t in best_thicknesses]
    
    return best_materials, best_thicknesses
