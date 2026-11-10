"""Broadband antireflection coating design via global + local optimization."""
import numpy as np
import time
import sys
from scipy.optimize import differential_evolution, minimize

def design(problem):
    """Design a broadband AR coating stack minimizing mean reflectance 400-800 nm."""
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s wall-clock kill
    
    n_ambient = problem["ambient_index"]
    n_substrate = problem["substrate_index"]
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    band = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    # Wavelengths for evaluation (401 points)
    wavelengths = np.linspace(band[0], band[1], 401)
    
    def compute_reflectance(labels, thicknesses, wls=wavelengths):
        """Compute reflectance spectrum using characteristic matrix method."""
        n_w = len(wls)
        R = np.zeros(n_w)
        for i, wl in enumerate(wls):
            # Build characteristic matrices
            # For each layer: M_j = [[cos(delta_j), (i/sin(delta_j))/n_j],
            #                    [i*n_j*sin(delta_j), cos(delta_j)]]
            # where delta_j = 2*pi*n_j*d_j / wl (for normal incidence, non-magnetic)
            
            # We'll build the product of all matrices
            # Start with identity
            M_total = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)
            
            for j in range(len(labels)):
                n_j = n_L if labels[j] == 'L' else n_H
                d_j = thicknesses[j]
                delta_j = 2.0 * np.pi * n_j * d_j / wl
                cos_d = np.cos(delta_j)
                sin_d = np.sin(delta_j)
                
                if abs(sin_d) < 1e-15:
                    # Layer is effectively zero thickness or integer quarter waves
                    M_j = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)
                else:
                    M_j = np.array([
                        [cos_d, 1j * cos_d / (n_j * sin_d)],
                        [1j * n_j * sin_d, cos_d]
                    ], dtype=complex)
                
                M_total = M_total @ M_j
            
            # Admittance at input plane
            # Y = (M21 + M22 * n_s) / (M11 + M12 * n_s)  -- wait, let me be careful
            # Actually: B = M11 + M12 * n_s, C = M21 + M22 * n_s
            # Y = C / B
            # R = |(Y - n_0)/(Y + n_0)|^2
            
            B = M_total[0, 0] + M_total[0, 1] * n_substrate
            C = M_total[1, 0] + M_total[1, 1] * n_substrate
            
            Y = C / B
            R[i] = np.abs((Y - n_ambient) / (Y + n_ambient)) ** 2
        
        return R
    
    def objective(params):
        """Objective function: mean reflectance. params = [n_layers, t1, t2, ..., tn, m1, m2, ..., mn]
        where m_i = 0 for L, 1 for H. We'll encode material choice as part of params for DE."""
        n_l = int(round(params[0]))
        n_l = max(1, min(n_l, max_layers))
        thicknesses = params[1:1+n_l]
        materials_codes = params[1+n_l:1+2*n_l]
        
        # Clamp thicknesses
        thicknesses = np.clip(thicknesses, 0.0, max_layer_nm)
        
        # Check total thickness
        if np.sum(thicknesses) > max_total_nm:
            return 100.0  # penalty
        
        labels = ['L' if mc < 0.5 else 'H' for mc in materials_codes]
        
        R = compute_reflectance(labels, thicknesses)
        return np.mean(R) * 100.0  # in percent
    
    def objective_fixed_n(n_layers, thicknesses, materials_codes):
        """Objective for fixed number of layers."""
        if n_layers < 1 or n_layers > max_layers:
            return 100.0
        thicknesses = np.clip(thicknesses, 0.0, max_layer_nm)
        if np.sum(thicknesses) > max_total_nm:
            return 100.0
        labels = ['L' if mc < 0.5 else 'H' for mc in materials_codes]
        R = compute_reflectance(labels, thicknesses)
        return np.mean(R) * 100.0
    
    # Start with a reasonable initial guess
    # Quarter-wave stack at center wavelength
    centre_wl = 0.5 * (band[0] + band[1])
    
    # Try a few fixed layer counts with differential evolution
    best_score = 100.0
    best_labels = None
    best_thicknesses = None
    
    # Start with smaller stacks for speed, then larger
    layer_counts_to_try = [2, 3, 4, 5, 6, 8, 10, 12]
    
    for n_l in layer_counts_to_try:
        if time.time() > deadline - 10.0:
            break
        
        # Bounds for DE: thicknesses in [0, max_layer_nm], materials in [0, 1]
        bounds = []
        for i in range(n_l):
            bounds.append((0.0, max_layer_nm))  # thickness
        for i in range(n_l):
            bounds.append((0.0, 1.0))  # material code
        
        # Initial population: include quarter-wave guess
        def de_objective(x):
            thicknesses = x[:n_l]
            materials_codes = x[n_l:]
            return objective_fixed_n(n_l, thicknesses, materials_codes)
        
        # Use DE with moderate settings
        try:
            result = differential_evolution(
                de_objective,
                bounds=bounds,
                maxiter=30,
                popsize=15,
                tol=0.01,
                seed=42,
                init='latinhypercube',
                polish=False,  # We'll do local refinement separately
                update_interval=5
            )
            
            if time.time() > deadline - 5.0:
                break
            
            if result.fun < best_score:
                best_score = result.fun
                thicknesses = result.x[:n_l]
                materials_codes = result.x[n_l:]
                labels = ['L' if mc < 0.5 else 'H' for mc in materials_codes]
                best_labels = labels
                best_thicknesses = thicknesses.copy()
                
        except Exception:
            continue
    
    # Local refinement with Nelder-Mead on the best solution found
    if best_labels is not None and time.time() < deadline - 5.0:
        n_l = len(best_labels)
        
        def local_objective(x):
            thicknesses = np.clip(x[:n_l], 0.0, max_layer_nm)
            if np.sum(thicknesses) > max_total_nm:
                return 100.0
            labels = best_labels  # Keep materials fixed during local refinement
            R = compute_reflectance(labels, thicknesses)
            return np.mean(R) * 100.0
        
        try:
            result = minimize(
                local_objective,
                best_thicknesses,
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 0.01, 'fatol': 0.001}
            )
            
            if result.fun < best_score:
                best_score = result.fun
                best_thicknesses = np.clip(result.x, 0.0, max_layer_nm)
                
        except Exception:
            pass
    
    # Also try a simple heuristic: quarter-wave stack
    # For broadband, try alternating L/H quarter-wave layers
    if time.time() < deadline - 5.0:
        for n_qw in [3, 5, 7]:
            labels = []
            thicknesses = []
            for i in range(n_qw):
                if i % 2 == 0:
                    labels.append('L')
                    thicknesses.append(centre_wl / (4.0 * n_L))
                else:
                    labels.append('H')
                    thicknesses.append(centre_wl / (4.0 * n_H))
            
            if len(labels) <= max_layers and np.sum(thicknesses) <= max_total_nm:
                R = compute_reflectance(labels, thicknesses)
                score = np.mean(R) * 100.0
                if score < best_score:
                    best_score = score
                    best_labels = labels
                    best_thicknesses = np.array(thicknesses)
    
    # Final validation and return
    if best_labels is None:
        # Fallback: single quarter-wave layer
        return ["L"], [centre_wl / (4.0 * n_L)]
    
    # Ensure constraints are met
    final_thicknesses = np.clip(best_thicknesses, 0.0, max_layer_nm)
    if np.sum(final_thicknesses) > max_total_nm:
        # Scale down
        scale = max_total_nm / np.sum(final_thicknesses)
        final_thicknesses = final_thicknesses * scale
    
    return best_labels, list(final_thicknesses)
