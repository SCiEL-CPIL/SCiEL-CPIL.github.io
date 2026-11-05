import time
import math
import numpy as np
from scipy.optimize import differential_evolution, minimize

def _reflectance_array(materials, thicknesses, n_ambient, n_substrate, wls, n_mat):
    """
    Compute reflectance array for given materials and thicknesses using
    the thin-film characteristic-matrix method.
    Returns array of reflectance values (0 to 1).
    """
    if len(materials) == 0:
        r = (n_ambient - n_substrate) / (n_ambient + n_substrate)
        return np.full(len(wls), r * r)

    n_layers = len(materials)
    n_wls = len(wls)
    
    # Precompute beta = 2*pi*n*d/wl for all layers and wavelengths
    # beta shape: (n_layers, n_wls)
    beta = np.zeros((n_layers, n_wls))
    for i, m in enumerate(materials):
        n = n_mat[m]
        d = thicknesses[i]
        if d != 0:
            beta[i, :] = 2.0 * math.pi * n * d / wls

    cosb = np.cos(beta)
    sinb = np.sin(beta)

    # Build the total characteristic matrix
    # M = M_1 * M_2 * ... * M_N
    # Each M_i = [[cos(beta_i), i*sin(beta_i)/n_i], [i*n_i*sin(beta_i), cos(beta_i)]]
    
    M11 = np.ones(n_wls, dtype=complex)
    M12 = np.zeros(n_wls, dtype=complex)
    M21 = np.zeros(n_wls, dtype=complex)
    M22 = np.ones(n_wls, dtype=complex)

    for i, m in enumerate(materials):
        n = n_mat[m]
        c = cosb[i]
        s = sinb[i]
        
        # Matrix elements for layer i
        A = c
        B = 1j * s / n
        C = 1j * n * s
        D = c

        # Matrix multiplication: M_new = M_old * M_i
        nM11 = M11 * A + M12 * C
        nM12 = M11 * B + M12 * D
        nM21 = M21 * A + M22 * C
        nM22 = M21 * B + M22 * D
        
        M11, M12, M21, M22 = nM11, nM12, nM21, nM22

    # Admittance Y
    Y = (M21 + n_substrate * M22) / (M11 + n_substrate * M12)
    
    # Reflectance r
    r = (n_ambient - Y) / (n_ambient + Y)
    
    # Reflectance R = |r|^2
    R = np.abs(r) ** 2
    return R

def _mean_reflectance_percent(materials, thicknesses, problem, wls, n_mat):
    """
    Compute mean reflectance in percent.
    Returns penalty if constraints are violated.
    """
    if len(materials) != len(thicknesses):
        return 1e6
    if len(materials) > problem["max_layers"]:
        return 1e6
        
    max_layer = problem["max_layer_nm"]
    max_total = problem["max_total_nm"]
    
    # Check individual thickness bounds
    for t in thicknesses:
        if t < -1e-9 or t > max_layer + 1e-9:
            return 1e6
            
    total_th = sum(thicknesses)
    if total_th > max_total + 1e-6:
        return 1e6 + (total_th - max_total) * 100.0

    if len(materials) == 0:
        r = (problem["ambient_index"] - problem["substrate_index"]) / (
            problem["ambient_index"] + problem["substrate_index"])
        return r * r * 100.0

    R = _reflectance_array(materials, thicknesses, problem["ambient_index"],
                           problem["substrate_index"], wls, n_mat)
    return float(np.mean(R)) * 100.0

def design(problem):
    """
    Design a broadband antireflection coating.
    """
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin

    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    n_ambient = problem["ambient_index"]
    n_substrate = problem["substrate_index"]
    max_layers = problem["max_layers"]
    max_total = problem["max_total_nm"]
    max_layer = problem["max_layer_nm"]
    
    band_start, band_end = problem["band_nm"]
    wls = np.linspace(band_start, band_end, 401)
    n_mat = problem["materials"]
    
    best_val = float('inf')
    best_mats = None
    best_ths = None
    
    # Candidate sequences
    # We test alternating sequences of different lengths.
    # Starting with L is generally better for AR on high-index substrate.
    sequences = []
    for length in [10, 12, 14, 16]:
        if length > max_layers:
            break
        seq = []
        for i in range(length):
            if i % 2 == 0:
                seq.append('L')
            else:
                seq.append('H')
        sequences.append(seq)
        
    for mats in sequences:
        if time.time() > deadline - 30.0:
            break
            
        n_layers = len(mats)
        
        # Define bounds for differential evolution
        # Each thickness is between 0 and max_layer
        bounds = [(0.0, max_layer) for _ in range(n_layers)]
        
        def objective_de(x):
            # x is a numpy array of thicknesses
            # Check total thickness constraint
            total_th = np.sum(x)
            if total_th > max_total:
                # Penalty proportional to violation
                return 1e6 + (total_th - max_total) * 10.0
            
            # Compute reflectance
            val = _mean_reflectance_percent(mats, list(x), problem, wls, n_mat)
            return val

        # Run differential evolution
        # popsize=15, maxiter=50 is a good balance for speed and quality
        # tol=1e-5
        try:
            de_result = differential_evolution(
                objective_de, 
                bounds, 
                seed=42, 
                maxiter=50, 
                popsize=15, 
                tol=1e-5,
                polish=False # We will polish separately
            )
            
            x_opt = de_result.x
            f_opt = de_result.fun
            
            # Refine with L-BFGS-B
            # L-BFGS-B handles bounds well
            def objective_lbfgsb(x):
                total_th = np.sum(x)
                if total_th > max_total:
                    return 1e6 + (total_th - max_total) * 10.0
                return _mean_reflectance_percent(mats, list(x), problem, wls, n_mat)

            res_lbfgsb = minimize(
                objective_lbfgsb, 
                x_opt, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-10}
            )
            
            x_final = res_lbfgsb.x
            f_final = res_lbfgsb.fun
            
            # Post-process: Ensure constraints strictly
            x_final = np.clip(x_final, 0.0, max_layer)
            total_final = np.sum(x_final)
            if total_final > max_total:
                factor = max_total / total_final
                x_final = x_final * factor
                x_final = np.clip(x_final, 0.0, max_layer)
                
            # Recalculate final score to be sure
            val_final = _mean_reflectance_percent(mats, list(x_final), problem, wls, n_mat)
            
            if val_final < best_val:
                best_val = val_final
                best_mats = mats
                best_ths = list(x_final)
                
        except Exception:
            # If DE fails, try a simple quarter-wave stack
            lambda_gm = math.sqrt(band_start * band_end)
            x0 = []
            for m in mats:
                n = n_mat[m]
                d = lambda_gm / (4.0 * n)
                x0.append(d)
            
            total = sum(x0)
            if total > max_total:
                factor = max_total / total
                x0 = [d * factor for d in x0]
            
            val_qw = _mean_reflectance_percent(mats, x0, problem, wls, n_mat)
            if val_qw < best_val:
                best_val = val_qw
                best_mats = mats
                best_ths = x0

    if best_mats is None:
        # Fallback: single layer quarter-wave at center
        best_mats = ['L']
        best_ths = [600.0 / (4.0 * n_L)]
        best_val = _mean_reflectance_percent(best_mats, best_ths, problem, wls, n_mat)
        
    # Final constraint check and scaling
    best_ths = [min(max(t, 0.0), max_layer) for t in best_ths]
    if sum(best_ths) > max_total + 1e-6:
        f = max_total / sum(best_ths)
        best_ths = [t * f for t in best_ths]
        
    return list(best_mats), list(best_ths)
