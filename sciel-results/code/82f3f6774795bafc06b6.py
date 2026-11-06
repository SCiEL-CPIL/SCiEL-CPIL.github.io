"""Broadband antireflection coating design via gradient-based L-BFGS-B optimization."""
import numpy as np
import time
from scipy.optimize import minimize

def _compute_reflectance(wavelengths_nm, indices, thicknesses_nm, ambient_idx, substrate_idx):
    """Compute reflectance (fraction, 0-1) at each wavelength using characteristic matrix method."""
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

def _objective_and_gradient(x, indices, wavelengths_nm, ambient_idx, substrate_idx,
                            max_layer_nm, max_total_nm):
    """Objective and gradient: mean reflectance in percent."""
    x = np.asarray(x, dtype=float)
    n_layers = len(x)
    
    # Clip to valid range
    x_clipped = np.clip(x, 0.0, max_layer_nm)
    
    # Check total thickness constraint
    total = np.sum(x_clipped)
    if total > max_total_nm:
        # Scale down to satisfy constraint
        x_clipped = x_clipped * (max_total_nm / total)
    
    # Compute reflectance
    R = _compute_reflectance(wavelengths_nm, indices, x_clipped, ambient_idx, substrate_idx)
    obj = np.mean(R) * 100.0
    
    # Compute gradient using finite differences (more robust than manual chain rule)
    eps = 1e-6
    grad = np.zeros(n_layers)
    for i in range(n_layers):
        x_plus = x_clipped.copy()
        x_minus = x_clipped.copy()
        x_plus[i] += eps
        x_minus[i] -= eps
        
        # Re-apply total constraint
        total_plus = np.sum(x_plus)
        if total_plus > max_total_nm:
            x_plus = x_plus * (max_total_nm / total_plus)
        total_minus = np.sum(x_minus)
        if total_minus > max_total_nm:
            x_minus = x_minus * (max_total_nm / total_minus)
        
        R_plus = _compute_reflectance(wavelengths_nm, indices, x_plus, ambient_idx, substrate_idx)
        R_minus = _compute_reflectance(wavelengths_nm, indices, x_minus, ambient_idx, substrate_idx)
        
        grad[i] = (np.mean(R_plus) - np.mean(R_minus)) / (2 * eps) * 100.0
    
    return obj, grad

def _l_bfgsb_optimize(f, x0, bounds, max_iter=200, time_deadline=None):
    """L-BFGS-B optimization with time deadline."""
    def wrapped_f(x):
        if time_deadline is not None and time.time() > time_deadline:
            raise Exception("Time deadline exceeded")
        obj, grad = f(x)
        return obj, grad
    
    try:
        result = minimize(wrapped_f, x0, method='L-BFGS-B', bounds=bounds,
                         options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8})
        return result.x, result.fun
    except Exception:
        return x0, f(x0)[0]

def design(problem):
    """Design a broadband antireflection coating."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    ambient_idx = problem["ambient_index"]
    substrate_idx = problem["substrate_index"]
    materials = problem["materials"]
    band_nm = problem["band_nm"]
    max_layers = problem.get("max_layers", 20)
    max_total_nm = problem.get("max_total_nm", 1000)
    max_layer_nm = problem.get("max_layer_nm", 400)
    
    wavelengths_nm = np.linspace(band_nm[0], band_nm[1], 401)
    center_wl = 0.5 * (band_nm[0] + band_nm[1])
    
    best_labels = None
    best_thicknesses = None
    best_score = np.inf
    
    # Define patterns: alternating L/H starting with L or H
    # Focus on odd numbers of layers for symmetric AR designs
    patterns = []
    
    for n in [5, 7, 9, 11, 13, 15, 17, 19]:
        if n > max_layers:
            continue
        # Alternating L/H
        labels_lh = ['L' if i % 2 == 0 else 'H' for i in range(n)]
        patterns.append((n, labels_lh))
        
        # Alternating H/L
        labels_hl = ['H' if i % 2 == 0 else 'L' for i in range(n)]
        patterns.append((n, labels_hl))
    
    # Sort by number of layers (fewer layers first for faster convergence)
    patterns.sort(key=lambda x: x[0])
    
    # Time budget allocation
    # L-BFGS-B is much faster than DE/NM, so we can try more patterns
    # Allocate ~10s per pattern for L-BFGS-B
    time_per_pattern = 8.0
    
    for n_layers, labels in patterns:
        if time.time() > deadline - 5:
            break
        
        indices = [materials[label] for label in labels]
        bounds = [(0.0, float(max_layer_nm))] * n_layers
        
        def make_f(indices):
            def f(x):
                return _objective_and_gradient(x, indices, wavelengths_nm, ambient_idx, 
                                               substrate_idx, max_layer_nm, max_total_nm)
            return f
        
        f = make_f(indices)
        
        # Multiple diverse starting points
        start_points = []
        
        # 1. Quarter-wave stack
        x_qw = np.array([center_wl / (4.0 * n) for n in indices])
        x_qw = np.clip(x_qw, 0, max_layer_nm)
        if np.sum(x_qw) > max_total_nm:
            x_qw = x_qw * (max_total_nm / np.sum(x_qw))
        start_points.append(x_qw)
        
        # 2. Uniform thickness
        x_uniform = np.full(n_layers, max_total_nm / n_layers)
        x_uniform = np.clip(x_uniform, 0, max_layer_nm)
        start_points.append(x_uniform)
        
        # 3. Random starting points (2-3 of them)
        np.random.seed(42)
        for _ in range(3):
            x_rand = np.random.uniform(0, max_layer_nm, size=n_layers)
            if np.sum(x_rand) > max_total_nm:
                x_rand = x_rand * (max_total_nm / np.sum(x_rand))
            start_points.append(x_rand)
        
        # 4. Asymmetric quarter-wave (thinner on one side)
        x_asym = np.array([center_wl / (4.0 * n) * (1.0 if i % 2 == 0 else 0.7) 
                          for i, n in enumerate(indices)])
        x_asym = np.clip(x_asym, 0, max_layer_nm)
        if np.sum(x_asym) > max_total_nm:
            x_asym = x_asym * (max_total_nm / np.sum(x_asym))
        start_points.append(x_asym)
        
        # Try each starting point with L-BFGS-B
        for x0 in start_points:
            if time.time() > deadline - 3:
                break
            
            # Quick evaluation of starting point
            f0 = f(x0)
            obj0 = f0[0]
            if obj0 < best_score:
                best_score = obj0
                best_labels = labels.copy()
                best_thicknesses = x0.copy()
            
            if obj0 >= 50.0:  # Skip if starting point is very bad
                continue
            
            # Run L-BFGS-B
            lbfgsb_deadline = time.time() + time_per_pattern
            try:
                x_opt, f_opt = _l_bfgsb_optimize(f, x0, bounds, max_iter=100, 
                                                 time_deadline=lbfgsb_deadline)
            except Exception:
                x_opt, f_opt = x0, obj0
            
            if f_opt < best_score:
                best_score = f_opt
                best_labels = labels.copy()
                best_thicknesses = x_opt.copy()
            
            # Early exit if very good
            if best_score < 0.05:
                break
        
        # Early exit if very good
        if best_score < 0.05:
            break
    
    # Final fallback
    if best_labels is None:
        return ["L"], [center_wl / (4.0 * materials["L"])]
    
    # Ensure constraints on final result
    thicknesses = np.clip(best_thicknesses, 0, max_layer_nm)
    if np.sum(thicknesses) > max_total_nm:
        thicknesses = thicknesses * (max_total_nm / np.sum(thicknesses))
    
    return best_labels, thicknesses.tolist()
