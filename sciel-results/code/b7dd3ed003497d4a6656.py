"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def _mean_reflectance(materials, thicknesses, problem):
    """Compute mean reflectance (percent) over the band using characteristic matrix method."""
    n_amb = problem["ambient_index"]
    n_sub = problem["substrate_index"]
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    band = problem["band_nm"]
    wls = np.linspace(band[0], band[1], 401)
    n_mat = np.array([n_L if m == 'L' else n_H for m in materials])
    d = np.array(thicknesses, dtype=float)
    total_R = 0.0
    for wl in wls:
        # Build characteristic matrix product
        M = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)
        for i in range(len(materials)):
            n_i = n_mat[i]
            phase = 2.0 * np.pi * n_i * d[i] / wl
            cos_p = np.cos(phase)
            sin_p = np.sin(phase)
            B = np.array([[cos_p, 1j * sin_p / n_i],
                          [1j * n_i * sin_p, cos_p]], dtype=complex)
            M = M @ B
        # Reflectance at normal incidence
        Y = M[0, 1] / M[1, 1]
        r = (n_amb - Y) / (n_amb + Y)
        R = np.abs(r) ** 2
        total_R += R * 100.0
    return total_R / len(wls)

def design(problem):
    """Multi-layer quarter-wave stack with local optimization for broadband AR coating."""
    t0 = time.time()
    deadline = t0 + 130.0  # safety margin before 150s kill

    n_amb = problem["ambient_index"]
    n_sub = problem["substrate_index"]
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    band = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    centre = 0.5 * (band[0] + band[1])

    # Fallback: single quarter-wave layer
    best_mats = ["L"]
    best_thick = [centre / (4.0 * n_L)]
    best_score = _mean_reflectance(best_mats, best_thick, problem)

    # Try multi-layer quarter-wave stacks: (H/L)^N pattern
    # Classic AR: start with H on substrate side, L on air side
    # For N pairs: L H L H ... L H (air to substrate)
    # Actually for AR on high-index substrate: first layer (air side) should be L
    # Pattern: L, H, L, H, ... alternating, starting with L
    for n_pairs in range(1, min(10, max_layers // 2 + 1)):
        n_layers = 2 * n_pairs
        if n_layers > max_layers:
            break
        # Quarter-wave thicknesses at centre wavelength
        mats = []
        thick = []
        for i in range(n_layers):
            if i % 2 == 0:
                mats.append('L')
                thick.append(centre / (4.0 * n_L))
            else:
                mats.append('H')
                thick.append(centre / (4.0 * n_H))
        
        # Check constraints
        if sum(thick) > max_total_nm or any(t > max_layer_nm for t in thick):
            continue
        
        score = _mean_reflectance(mats, thick, problem)
        if score < best_score:
            best_score = score
            best_mats = mats[:]
            best_thick = thick[:]
        
        if time.time() > deadline - 20:
            break
    
    # Try odd number of layers: L H L H ... L (ends with L on air side, H on substrate side)
    for n_layers in range(3, min(19, max_layers), 2):
        if time.time() > deadline - 20:
            break
        mats = []
        thick = []
        for i in range(n_layers):
            if i % 2 == 0:
                mats.append('L')
                thick.append(centre / (4.0 * n_L))
            else:
                mats.append('H')
                thick.append(centre / (4.0 * n_H))
        
        if sum(thick) > max_total_nm or any(t > max_layer_nm for t in thick):
            continue
        
        score = _mean_reflectance(mats, thick, problem)
        if score < best_score:
            best_score = score
            best_mats = mats[:]
            best_thick = thick[:]
    
    # Local optimization: Nelder-Mead style on the best design
    if time.time() < deadline - 10 and len(best_mats) >= 2:
        try:
            from scipy.optimize import minimize
            n_params = len(best_thick)
            x0 = np.array(best_thick, dtype=float)
            
            def obj(x):
                if np.any(x < 0) or np.any(x > max_layer_nm) or np.sum(x) > max_total_nm:
                    return 100.0
                return _mean_reflectance(best_mats, list(x), problem)
            
            # Use Nelder-Mead with reasonable bounds via penalty
            res = minimize(obj, x0, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-8})
            
            if res.success and res.fun < best_score:
                # Clamp to valid range
                x_opt = np.clip(res.x, 0, max_layer_nm)
                if np.sum(x_opt) <= max_total_nm:
                    best_score = float(res.fun)
                    best_thick = list(x_opt)
        except Exception:
            pass
    
    return best_mats, best_thick
# EVOLVE-BLOCK-END
