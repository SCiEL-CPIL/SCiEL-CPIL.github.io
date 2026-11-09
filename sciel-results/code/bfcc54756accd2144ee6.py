import numpy as np
from scipy.optimize import minimize
import time

def _reflectance_spectrum(thicknesses, indices, n_substrate, n_ambient, wavelengths):
    N = len(thicknesses)
    R = np.zeros(len(wavelengths))
    for i, lam in enumerate(wavelengths):
        deltas = 2.0 * np.pi * indices * thicknesses / lam
        M_total = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)
        
        for j in range(N):
            n_j = indices[j]
            d_j = deltas[j]
            c = np.cos(d_j)
            s = np.sin(d_j)
            M_j = np.array([[c, 1j * s / n_j], [1j * n_j * s, c]], dtype=complex)
            M_total = M_total @ M_j
            
        Y_in = (M_total[1, 0] + M_total[1, 1] / n_substrate) / (M_total[0, 0] + M_total[0, 1] / n_substrate)
        r = (Y_in - n_ambient) / (Y_in + n_ambient)
        R[i] = np.abs(r) ** 2
    return R

def design(problem):
    n_ambient = problem["ambient_index"]
    n_substrate = problem["substrate_index"]
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    band_nm = problem["band_nm"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    wavelengths = np.linspace(band_nm[0], band_nm[1], 401)
    
    # Fallback solution: Single L layer
    center_wl = 0.5 * (band_nm[0] + band_nm[1])
    t_single = center_wl / (4.0 * n_L)
    if t_single > max_layer_nm:
        t_single = max_layer_nm
    fallback_labels = ["L"]
    fallback_thick = [t_single]
    
    # Primary solution: 10-layer stack
    N_layers = 10
    layer_labels = []
    for i in range(N_layers):
        if i % 2 == 0:
            layer_labels.append('L')
        else:
            layer_labels.append('H')
            
    indices = np.array([n_L if l == 'L' else n_H for l in layer_labels])
    
    t_init = np.array([center_wl / (4.0 * n) for n in indices])
    t_init = np.clip(t_init, 0, max_layer_nm)
    
    if np.sum(t_init) > max_total_nm:
        scale = max_total_nm / np.sum(t_init)
        t_init = t_init * scale * 0.95
        
    def objective(t):
        penalty = 0.0
        for ti in t:
            if ti < 0:
                penalty += 100 * (0 - ti) ** 2
            elif ti > max_layer_nm:
                penalty += 100 * (ti - max_layer_nm) ** 2
        
        total_t = np.sum(t)
        if total_t > max_total_nm:
            penalty += 100 * (total_t - max_total_nm) ** 2
            
        if penalty > 1e-6:
            return 1000.0 + penalty
            
        R = _reflectance_spectrum(t, indices, n_substrate, n_ambient, wavelengths)
        return np.mean(R) * 100.0
        
    bounds = [(0, max_layer_nm) for _ in range(N_layers)]
    
    start_time = time.time()
    timeout = 40.0
    
    best_t = t_init
    best_obj = objective(t_init)
    
    try:
        res = minimize(objective, t_init, method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8})
        
        if res.fun < best_obj:
            if np.all(res.x >= 0) and np.all(res.x <= max_layer_nm) and np.sum(res.x) <= max_total_nm + 1e-6:
                best_t = res.x
                best_obj = res.fun
    except Exception:
        pass
        
    elapsed = time.time() - start_time
    if elapsed > timeout:
        return fallback_labels, fallback_thick
        
    if np.any(best_t < 0) or np.any(best_t > max_layer_nm) or np.sum(best_t) > max_total_nm:
        return fallback_labels, fallback_thick
        
    return layer_labels, list(best_t)
