import numpy as np
from scipy.optimize import minimize
import time

def _mean_reflectance_vec(thicknesses, materials_seq, n_amb, n_sub, n_L, n_H, wavelengths):
    """
    Calculate mean reflectance in percent for a given set of thicknesses.
    Vectorized over wavelengths.
    """
    N = len(thicknesses)
    if N == 0:
        r = (n_amb - n_sub) / (n_amb + n_sub)
        return (abs(r)**2 * 100.0)

    indices = np.array([n_L if m == 'L' else n_H for m in materials_seq])
    
    # wavelengths shape: (M,)
    # indices shape: (N,)
    # thicknesses shape: (N,)
    
    # Calculate deltas: (N, M)
    deltas = 2.0 * np.pi * indices[:, None] * thicknesses[:, None] / wavelengths[None, :]
    
    cos_d = np.cos(deltas) # (N, M)
    sin_d = np.sin(deltas) # (N, M)
    
    # Initialize M for all wavelengths
    # Shape: (M, 2, 2)
    M = np.zeros((len(wavelengths), 2, 2), dtype=complex)
    M[:, 0, 0] = 1.0
    M[:, 1, 1] = 1.0
    
    for i in range(N):
        n_i = indices[i]
        
        # Construct M_i for all wavelengths
        M_i = np.zeros((len(wavelengths), 2, 2), dtype=complex)
        M_i[:, 0, 0] = cos_d[i]
        M_i[:, 0, 1] = 1j * sin_d[i] / n_i
        M_i[:, 1, 0] = 1j * n_i * sin_d[i]
        M_i[:, 1, 1] = cos_d[i]
        
        # Batched matrix multiplication
        M = np.matmul(M, M_i)
        
    M11 = M[:, 0, 0]
    M12 = M[:, 0, 1]
    M21 = M[:, 1, 0]
    M22 = M[:, 1, 1]
    
    denom = M11 + n_sub * M12
    # Avoid division by zero
    denom[np.abs(denom) < 1e-12] = 1e-12
    
    Y_in = (M21 + n_sub * M22) / denom
    r = (n_amb - Y_in) / (n_amb + Y_in)
    R = np.abs(r)**2
    
    return np.mean(R) * 100.0

def design(problem):
    start_time = time.time()
    timeout = 140.0  # Safety margin for 150s wall clock
    
    n_amb = problem["ambient_index"]
    n_sub = problem["substrate_index"]
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    band = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    wavelengths = np.linspace(band[0], band[1], 401)
    
    best_score = float('inf')
    best_materials = ["L"]
    best_thicknesses = [0.5 * (band[0] + band[1]) / (4.0 * n_L)]
    
    # Try different stack lengths
    candidate_lengths = [3, 5, 7, 9, 11]
    
    for N in candidate_lengths:
        if time.time() - start_time > timeout:
            break
            
        if N > max_layers:
            continue
            
        # Generate sequence: L H L H ... L
        seq = []
        for i in range(N):
            if i % 2 == 0:
                seq.append('L')
            else:
                seq.append('H')
        
        # Initial guess: Quarter-wave at center wavelength
        center_wl = 0.5 * (band[0] + band[1])
        indices = np.array([n_L if m == 'L' else n_H for m in seq])
        init_thick = center_wl / (4.0 * indices)
        
        # Ensure initial guess is within bounds
        init_thick = np.clip(init_thick, 0, max_layer_nm)
        
        def objective(thick):
            # Penalty for constraints
            penalty = 0.0
            # Check individual bounds
            # Using quadratic penalty for smoothness
            neg_viol = np.minimum(0, thick)
            pen1 = np.sum(neg_viol**2) * 10.0
            pos_viol = np.maximum(0, thick - max_layer_nm)
            pen2 = np.sum(pos_viol**2) * 10.0
            
            total_thick = np.sum(thick)
            if total_thick > max_total_nm:
                pen3 = ((total_thick - max_total_nm)**2) * 10.0
            else:
                pen3 = 0.0
            
            # Calculate reflectance
            safe_thick = np.clip(thick, 0, max_layer_nm)
            R = _mean_reflectance_vec(safe_thick, seq, n_amb, n_sub, n_L, n_H, wavelengths)
            
            return R + pen1 + pen2 + pen3
        
        try:
            res = minimize(objective, init_thick, method='Nelder-Mead', 
                           options={'maxiter': 1000, 'xatol': 1e-5, 'fatol': 1e-8, 'adaptive': True})
            
            res_thick = res.x
            res_thick = np.clip(res_thick, 0, max_layer_nm)
            
            total_t = np.sum(res_thick)
            if total_t > max_total_nm:
                scale = max_total_nm / total_t
                res_thick = res_thick * scale
                res_thick = np.clip(res_thick, 0, max_layer_nm)
            
            final_R = _mean_reflectance_vec(res_thick, seq, n_amb, n_sub, n_L, n_H, wavelengths)
            
            if final_R < best_score:
                best_score = final_R
                best_materials = seq
                best_thicknesses = list(res_thick)
                
        except Exception:
            pass
        
    if best_score == float('inf'):
        center = 0.5 * (band[0] + band[1])
        return ["L"], [center / (4.0 * n_L)]
        
    return best_materials, best_thicknesses
