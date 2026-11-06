"""Broadband antireflection coating design via differential evolution with adaptive wavelength strategy."""
import numpy as np
import time
from scipy.optimize import differential_evolution

def _compute_reflectance(wavelengths_nm, indices, thicknesses_nm, ambient_idx, substrate_idx):
    """
    Compute reflectance (fraction, 0-1) at each wavelength using the characteristic matrix method.
    """
    n_layers = len(indices)
    if n_layers == 0:
        r0 = (ambient_idx - substrate_idx) / (ambient_idx + substrate_idx)
        return np.full(len(wavelengths_nm), r0 * r0)
    
    k = 2.0 * np.pi / wavelengths_nm
    
    M00 = np.ones(len(wavelengths_nm), dtype=complex)
    M01 = np.zeros(len(wavelengths_nm), dtype=complex)
    M10 = np.zeros(len(wavelengths_nm), dtype=complex)
    M11 = np.ones(len(wavelengths_nm), dtype=complex)
    
    for i in range(n_layers):
        n_i = indices[i]
        d_i = thicknesses_nm[i]
        delta = k * n_i * d_i
        
        cos_d = np.cos(delta)
        sin_d = np.sin(delta)
        
        B00 = cos_d
        B01 = 1j * sin_d / n_i
        B10 = 1j * n_i * sin_d
        B11 = cos_d
        
        new_M00 = M00 * B00 + M01 * B10
        new_M01 = M00 * B01 + M01 * B11
        new_M10 = M10 * B00 + M11 * B10
        new_M11 = M10 * B01 + M11 * B11
        
        M00, M01, M10, M11 = new_M00, new_M01, new_M10, new_M11
    
    Y_out = substrate_idx
    Y_in = (M10 + M11 * Y_out) / (M00 + M01 * Y_out)
    
    r = (ambient_idx - Y_in) / (ambient_idx + Y_in)
    R = np.abs(r) ** 2
    
    return R

def _objective(de_params, indices, wavelengths_nm, ambient_idx, substrate_idx, 
               max_layer_nm, max_total_nm):
    """
    Objective function for DE: takes flat array of thicknesses.
    Returns mean reflectance in percent with penalty for constraint violations.
    """
    thicknesses = np.asarray(de_params, dtype=float)
    thicknesses = np.clip(thicknesses, 0.0, max_layer_nm)
    
    if np.sum(thicknesses) > max_total_nm:
        # Scale down to satisfy total thickness constraint
        thicknesses = thicknesses * (max_total_nm / np.sum(thicknesses))
    
    R = _compute_reflectance(wavelengths_nm, indices, thicknesses, 
                             ambient_idx, substrate_idx)
    return np.mean(R) * 100.0

def _de_wrapper(args):
    """Wrapper for DE that unpacks arguments."""
    de_params, indices, wavelengths_nm, ambient_idx, substrate_idx, max_layer_nm, max_total_nm = args
    return _objective(de_params, indices, wavelengths_nm, ambient_idx, substrate_idx, 
                      max_layer_nm, max_total_nm)

def _generate_label_patterns(n_layers):
    """Generate diverse label patterns for a given number of layers."""
    patterns = []
    
    # Alternating starting with L
    labels_l = ['L' if i % 2 == 0 else 'H' for i in range(n_layers)]
    patterns.append(labels_l)
    
    # Alternating starting with H
    labels_h = ['H' if i % 2 == 0 else 'L' for i in range(n_layers)]
    patterns.append(labels_h)
    
    # If n_layers >= 4, add period-4 patterns
    if n_layers >= 4:
        # L H H L repeating
        labels_lhh = ['L' if i % 4 == 0 else ('H' if i % 4 in [1, 2] else 'L') for i in range(n_layers)]
        patterns.append(labels_lhh)
        
        # H L L H repeating
        labels_hll = ['H' if i % 4 == 0 else ('L' if i % 4 in [1, 2] else 'H') for i in range(n_layers)]
        patterns.append(labels_hll)
    
    # For larger n, add some random patterns
    if n_layers >= 8:
        rng = np.random.RandomState(42)
        for _ in range(3):
            labels = ['L' if rng.random() < 0.5 else 'H' for _ in range(n_layers)]
            # Ensure not all same
            if len(set(labels)) > 1:
                patterns.append(labels)
    
    return patterns

def design(problem):
    """
    Design a broadband antireflection coating using differential evolution.
    Returns: (labels, thicknesses)
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    ambient_idx = problem["ambient_index"]
    substrate_idx = problem["substrate_index"]
    materials = problem["materials"]
    band_nm = problem["band_nm"]
    max_layers = problem.get("max_layers", 20)
    max_total_nm = problem.get("max_total_nm", 1000)
    max_layer_nm = problem.get("max_layer_nm", 400)
    
    # Coarse wavelength grid for fast evaluation during optimization
    wavelengths_coarse = np.linspace(band_nm[0], band_nm[1], 51)
    # Fine wavelength grid for final evaluation
    wavelengths_fine = np.linspace(band_nm[0], band_nm[1], 401)
    
    # Best result tracking
    best_labels = None
    best_thicknesses = None
    best_score = np.inf
    
    # Try different layer counts, prioritizing those known to work well
    layer_counts = [5, 7, 9, 11, 13, 15, 17, 19]
    layer_counts = [n for n in layer_counts if n <= max_layers]
    
    # Sort by expected performance: mid-range layers often best for broadband
    layer_counts.sort(key=lambda n: abs(n - 11))
    
    for n_layers in layer_counts:
        if time.time() > deadline - 30:
            break
        
        # Generate label patterns for this layer count
        label_patterns = _generate_label_patterns(n_layers)
        
        for labels in label_patterns:
            if time.time() > deadline - 25:
                break
            
            indices = [materials[label] for label in labels]
            bounds = [(0.0, float(max_layer_nm))] * n_layers
            
            # Create DE wrapper with fixed indices
            de_args = (None, indices, wavelengths_coarse, ambient_idx, substrate_idx, 
                       max_layer_nm, max_total_nm)
            
            def de_objective(x, _args=de_args):
                _args = list(_args)
                _args[0] = x
                return _objective(x, _args[1], _args[2], _args[3], _args[4], _args[5], _args[6])
            
            # Run differential evolution with time-limited budget
            time_per_de = max(2.0, min(8.0, (deadline - 20 - time.time()) / max(1, len(label_patterns))))
            de_deadline = time.time() + time_per_de
            
            try:
                result = differential_evolution(
                    de_objective,
                    bounds,
                    maxiter=50,
                    popsize=15,
                    tol=1e-8,
                    seed=42,
                    polish=False,  # We'll do our own polish
                    callback=None
                )
                
                if result.success or result.fun < best_score:
                    x_opt = result.x
                    f_coarse = result.fun
                    
                    # Evaluate on fine grid
                    f_fine = _objective(x_opt, indices, wavelengths_fine, ambient_idx, substrate_idx, 
                                        max_layer_nm, max_total_nm)
                    
                    if f_fine < best_score:
                        best_score = f_fine
                        best_labels = labels.copy()
                        best_thicknesses = x_opt.copy()
                        
                        # Early exit if very good
                        if best_score < 0.5:
                            break
                
            except Exception as e:
                pass
        
        # Early exit if very good
        if best_score < 0.3:
            break
    
    # If we haven't found a good solution, try a few more patterns with more DE iterations
    if best_score > 1.0 and time.time() < deadline - 40:
        for n_layers in [9, 11, 13]:
            if n_layers > max_layers or time.time() > deadline - 35:
                continue
            
            indices = [materials['L' if i % 2 == 0 else 'H'] for i in range(n_layers)]
            bounds = [(0.0, float(max_layer_nm))] * n_layers
            
            def de_objective2(x):
                return _objective(x, indices, wavelengths_coarse, ambient_idx, substrate_idx, 
                                  max_layer_nm, max_total_nm)
            
            try:
                result = differential_evolution(
                    de_objective2,
                    bounds,
                    maxiter=100,
                    popsize=20,
                    tol=1e-10,
                    seed=123,
                    polish=False
                )
                
                if result.fun < best_score:
                    x_opt = result.x
                    f_fine = _objective(x_opt, indices, wavelengths_fine, ambient_idx, substrate_idx, 
                                        max_layer_nm, max_total_nm)
                    
                    if f_fine < best_score:
                        best_score = f_fine
                        best_labels = ['L' if i % 2 == 0 else 'H' for i in range(n_layers)]
                        best_thicknesses = x_opt.copy()
                        
            except Exception:
                pass
    
    # Final fallback
    if best_labels is None:
        center_wl = 0.5 * (band_nm[0] + band_nm[1])
        return ["L"], [center_wl / (4.0 * materials["L"])]
    
    # Ensure constraints on final result
    thicknesses = np.clip(best_thicknesses, 0, max_layer_nm)
    if np.sum(thicknesses) > max_total_nm:
        thicknesses = thicknesses * (max_total_nm / np.sum(thicknesses))
    
    return best_labels, thicknesses.tolist()
