"""Design broadband antireflection coating using numerical optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize as opt

def _reflectance_spectrum(thicknesses, indices, n_layers, wavelengths, ambient, substrate):
    """
    Compute normal-incidence reflectance using characteristic matrix method.
    Returns reflectance as a fraction (0 to 1).
    Vectorized over wavelengths.
    """
    # thicknesses: list or array of length n_layers
    # indices: list or array of length n_layers
    # wavelengths: array of length N_wl
    
    # Initialize M as identity for all wavelengths
    # M is 2x2 matrix for each wavelength
    # m11, m12, m21, m22 are arrays of shape (N_wl,)
    
    m11 = np.ones_like(wavelengths, dtype=complex)
    m12 = np.zeros_like(wavelengths, dtype=complex)
    m21 = np.zeros_like(wavelengths, dtype=complex)
    m22 = np.ones_like(wavelengths, dtype=complex)
    
    for i in range(n_layers):
        ni = indices[i]
        di = thicknesses[i]
        
        if di == 0:
            continue
            
        delta = 2.0 * np.pi * ni * di / wavelengths
        cd = np.cos(delta)
        sd = np.sin(delta)
        
        # Layer matrix
        li11 = cd
        li12 = 1j * sd / ni
        li21 = 1j * ni * sd
        li22 = cd
        
        # M_new = M_old * M_layer
        nm11 = m11 * li11 + m12 * li21
        nm12 = m11 * li12 + m12 * li22
        nm21 = m21 * li11 + m22 * li21
        nm22 = m21 * li12 + m22 * li22
        
        m11, m12, m21, m22 = nm11, nm12, nm21, nm22
    
    # Effective admittance
    y0 = (m21 + m22 * substrate) / (m11 + m12 * substrate)
    r = (y0 - ambient) / (y0 + ambient)
    R = np.abs(r) ** 2
    return R

def design(problem):
    ambient = problem["ambient_index"]
    substrate = problem["substrate_index"]
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    band = problem["band_nm"]
    wl_min, wl_max = band[0], band[1]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    wavelengths = np.linspace(wl_min, wl_max, 401)
    wl_center = 0.5 * (wl_min + wl_max)
    
    best_R = np.inf
    best_design = None
    
    # Try different numbers of layers
    # We limit to reasonable numbers to fit in 60s CPU time
    # Optimization is expensive, so we try a few specific counts
    candidate_layer_counts = [5, 7, 9, 11, 13, 15]
    
    for n_layers in candidate_layer_counts:
        if n_layers > max_layers:
            continue
            
        # Try starting with L and H
        for start_char in ['L', 'H']:
            labels = []
            indices = []
            for i in range(n_layers):
                if i % 2 == 0:
                    labels.append(start_char)
                    indices.append(n_L if start_char == 'L' else n_H)
                else:
                    other = 'H' if start_char == 'L' else 'L'
                    labels.append(other)
                    indices.append(n_H if other == 'H' else n_L)
            
            # Initial guess: Quarter-wave at center wavelength
            d_init = []
            for i in range(n_layers):
                ni = indices[i]
                d_i = wl_center / (4.0 * ni)
                d_init.append(d_i)
                
            # Check total thickness constraint and scale if necessary
            total_init = sum(d_init)
            if total_init > max_total_nm:
                scale = max_total_nm / total_init
                d_init = [d * scale for d in d_init]
            
            # Ensure individual layers are within bounds
            d_init = [min(max(d, 0), max_layer_nm) for d in d_init]
            
            # Objective function
            def objective(thicknesses):
                # thicknesses is a 1D array
                R = _reflectance_spectrum(thicknesses, indices, n_layers, wavelengths, ambient, substrate)
                mean_R = np.mean(R)
                
                # Penalty for exceeding total thickness
                total_t = np.sum(thicknesses)
                penalty = 10.0 * max(0.0, total_t - max_total_nm)
                
                return mean_R + penalty
            
            # Bounds for L-BFGS-B
            bounds = [(0, max_layer_nm)] * n_layers
            
            # Run optimization
            # Use L-BFGS-B as it is efficient for smooth problems
            try:
                result = opt.minimize(objective, d_init, method='L-BFGS-B', bounds=bounds, options={'maxiter': 200, 'ftol': 1e-6})
                
                # Check if the result is valid (feasible)
                if result.success or result.status == 0:
                    opt_thick = result.x
                    # Clamp to bounds just in case
                    opt_thick = np.clip(opt_thick, 0, max_layer_nm)
                    
                    # Final check on total thickness
                    if np.sum(opt_thick) > max_total_nm + 1e-5:
                        # Scale down if slightly over
                        scale = max_total_nm / np.sum(opt_thick)
                        opt_thick = opt_thick * scale
                        opt_thick = np.clip(opt_thick, 0, max_layer_nm)
                    
                    # Calculate final score without penalty
                    R = _reflectance_spectrum(opt_thick, indices, n_layers, wavelengths, ambient, substrate)
                    mean_R = np.mean(R)
                    
                    if mean_R < best_R:
                        best_R = mean_R
                        best_design = (labels, list(opt_thick))
                        
            except Exception:
                pass
    
    # Fallback if no optimization succeeded
    if best_design is None:
        # Simple quarter wave L layer
        d = wl_center / (4.0 * n_L)
        if d > max_layer_nm:
            d = max_layer_nm
        return ["L"], [d]
        
    return best_design

# EVOLVE-BLOCK-END
