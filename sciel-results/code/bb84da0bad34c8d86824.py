import numpy as np
import time
from scipy.optimize import minimize

def _reflectance_spectrum(labels, thicknesses, wavelengths, materials, n_ambient, n_substrate):
    N = len(thicknesses)
    reflectance = np.zeros(len(wavelengths))
    indices = np.array([materials[label] for label in labels])
    
    for i_w, lam in enumerate(wavelengths):
        M = np.array([[1.0+0.0j, 0.0+0.0j], [0.0+0.0j, 1.0+0.0j]])
        for i in range(N):
            n_i = indices[i]
            t_i = thicknesses[i]
            delta = 2.0 * np.pi * n_i * t_i / lam
            cos_d = np.cos(delta)
            sin_d = np.sin(delta)
            M_i = np.array([
                [cos_d, 1.0j * sin_d / n_i],
                [1.0j * n_i * sin_d, cos_d]
            ])
            M = M_i @ M
        
        B = M[0, 0] + M[0, 1] / n_substrate
        C = M[1, 0] + M[1, 1] / n_substrate
        r = (n_ambient * B - C) / (n_ambient * B + C)
        reflectance[i_w] = np.abs(r) ** 2
    return reflectance

def design(problem):
    start_time = time.time()
    deadline = start_time + 50.0
    
    n_ambient = problem["ambient_index"]
    n_substrate = problem["substrate_index"]
    materials = problem["materials"]
    band_nm = problem["band_nm"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    wavelengths = np.linspace(band_nm[0], band_nm[1], 401)
    n_L = materials['L']
    n_H = materials['H']
    lambda_c = np.sqrt(band_nm[0] * band_nm[1])
    opt_thick_c = lambda_c / 4.0
    
    best_labels = None
    best_thick = None
    best_score = float('inf')
    
    # Try 5 layers and 9 layers
    for num_layers in [5, 9]:
        labels = []
        for i in range(num_layers):
            if i % 2 == 0:
                labels.append('L')
            else:
                labels.append('H')
        
        init_thick = []
        for lab in labels:
            n_i = materials[lab]
            init_thick.append(opt_thick_c / n_i)
        
        def objective(thicknesses, labels=labels):
            if np.any(thicknesses < 0) or np.any(thicknesses > max_layer_nm):
                return 1.0
            if np.sum(thicknesses) > max_total_nm:
                return 1.0
            refl = _reflectance_spectrum(labels, thicknesses, wavelengths, materials, n_ambient, n_substrate)
            return np.mean(refl)
        
        x0 = np.array(init_thick)
        bounds = [(0, max_layer_nm) for _ in range(num_layers)]
        
        try:
            res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 1000})
            score = res.fun
            if score < best_score:
                best_score = score
                best_labels = labels
                best_thick = res.x
        except Exception:
            pass
            
        if time.time() > deadline:
            break
            
    if best_labels is None:
        # Fallback to initial 5-layer
        labels = ['L', 'H', 'L', 'H', 'L']
        thick = [opt_thick_c / n_L, opt_thick_c / n_H, opt_thick_c / n_L, opt_thick_c / n_H, opt_thick_c / n_L]
        return labels, thick
        
    # Final constraint check
    best_thick = np.clip(best_thick, 0, max_layer_nm)
    if np.sum(best_thick) > max_total_nm:
        scale = max_total_nm / np.sum(best_thick)
        best_thick = best_thick * scale
        best_thick = np.clip(best_thick, 0, max_layer_nm)
        
    return best_labels, list(best_thick)
