"""Broadband antireflection coating design via multi-start Nelder-Mead with gradient polish."""
import numpy as np
import time

def _compute_reflectance(wavelengths_nm, indices, thicknesses_nm, ambient_idx, substrate_idx):
    """
    Compute reflectance (fraction, 0-1) at each wavelength using the characteristic matrix method.
    Vectorized over wavelengths.
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

def _objective(thicknesses, indices, wavelengths_nm, ambient_idx, substrate_idx, 
               max_layer_nm, max_total_nm):
    """
    Objective: mean reflectance in percent. Returns large penalty if constraints violated.
    """
    thicknesses = np.asarray(thicknesses, dtype=float)
    if np.any(thicknesses < -1e-6) or np.any(thicknesses > max_layer_nm + 1e-6):
        return 100.0
    if np.sum(thicknesses) > max_total_nm + 1e-6:
        return 100.0
    
    thicknesses_clipped = np.clip(thicknesses, 0.0, max_layer_nm)
    if np.sum(thicknesses_clipped) > max_total_nm:
        return 100.0
    
    R = _compute_reflectance(wavelengths_nm, indices, thicknesses_clipped, 
                             ambient_idx, substrate_idx)
    return np.mean(R) * 100.0

def _nelder_mead(f, x0, bounds, max_iter=5000, tol=1e-12, time_deadline=None):
    """
    Nelder-Mead simplex with bounds and adaptive step size.
    """
    n = len(x0)
    lo_bounds = np.array([b[0] for b in bounds])
    hi_bounds = np.array([b[1] for b in bounds])
    x0 = np.clip(np.asarray(x0, dtype=float), lo_bounds, hi_bounds)
    
    simplex = np.zeros((n + 1, n))
    simplex[0] = x0
    
    for i in range(n):
        simplex[i + 1] = x0.copy()
        lo, hi = bounds[i]
        step = 0.1 * (hi - lo) if hi > lo else 0.1
        simplex[i + 1][i] += step
    
    fvals = np.array([f(simplex[i]) for i in range(n + 1)])
    
    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    sigma = 0.5
    
    for it in range(max_iter):
        if time_deadline is not None and time.time() > time_deadline:
            break
        
        idx = np.argsort(fvals)
        simplex = simplex[idx]
        fvals = fvals[idx]
        
        if np.max(fvals) - np.min(fvals) < tol:
            break
        
        centroid = np.mean(simplex[:-1], axis=0)
        
        x_r = centroid + alpha * (centroid - simplex[-1])
        x_r = np.clip(x_r, lo_bounds, hi_bounds)
        f_r = f(x_r)
        
        if f_r < fvals[0]:
            x_e = centroid + gamma * (x_r - centroid)
            x_e = np.clip(x_e, lo_bounds, hi_bounds)
            f_e = f(x_e)
            
            if f_e < f_r:
                simplex[-1] = x_e
                fvals[-1] = f_e
            else:
                simplex[-1] = x_r
                fvals[-1] = f_r
        elif f_r < fvals[-2]:
            simplex[-1] = x_r
            fvals[-1] = f_r
        else:
            x_c = centroid + rho * (simplex[-1] - centroid)
            x_c = np.clip(x_c, lo_bounds, hi_bounds)
            f_c = f(x_c)
            
            if f_c < fvals[-1]:
                simplex[-1] = x_c
                fvals[-1] = f_c
            else:
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    simplex[i] = np.clip(simplex[i], lo_bounds, hi_bounds)
                    fvals[i] = f(simplex[i])
    
    best_idx = np.argmin(fvals)
    return simplex[best_idx], fvals[best_idx]

def _polish_with_gradient_descent(f, x0, bounds, max_iter=2000, time_deadline=None):
    """
    Gradient descent with numerical gradient and line search for final polish.
    """
    n = len(x0)
    lo_bounds = np.array([b[0] for b in bounds])
    hi_bounds = np.array([b[1] for b in bounds])
    x = np.clip(np.asarray(x0, dtype=float), lo_bounds, hi_bounds)
    fval = f(x)
    
    def numerical_gradient(x):
        eps = 1e-6
        grad = np.zeros(n)
        for i in range(n):
            x_plus = x.copy()
            x_plus[i] += eps
            x_plus = np.clip(x_plus, lo_bounds, hi_bounds)
            x_minus = x.copy()
            x_minus[i] -= eps
            x_minus = np.clip(x_minus, lo_bounds, hi_bounds)
            grad[i] = (f(x_plus) - f(x_minus)) / (2 * eps)
        return grad
    
    lr = 1.0
    for it in range(max_iter):
        if time_deadline is not None and time.time() > time_deadline:
            break
        
        grad = numerical_gradient(x)
        if np.max(np.abs(grad)) < 1e-8:
            break
        
        direction = -grad
        norm_dir = np.linalg.norm(direction)
        if norm_dir < 1e-10:
            break
        direction = direction / norm_dir
        
        step = lr
        f_old = fval
        for ls_iter in range(50):
            x_trial = x + step * direction
            x_trial = np.clip(x_trial, lo_bounds, hi_bounds)
            f_trial = f(x_trial)
            
            if f_trial < f_old - 1e-4 * step * np.dot(grad, direction):
                x = x_trial
                fval = f_trial
                break
            else:
                step *= 0.5
        
        if fval < f_old - 1e-3:
            lr = min(lr * 1.5, 10.0)
        elif fval > f_old + 1e-4:
            lr = max(lr * 0.5, 0.01)
    
    return x, fval

def design(problem):
    """
    Design a broadband antireflection coating.
    Returns: (labels, thicknesses)
    """
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
    
    # Generate layer patterns: alternating L/H starting with either
    patterns = []
    for n in range(1, min(max_layers + 1, 21), 2):
        labels_L = ['L' if i % 2 == 0 else 'H' for i in range(n)]
        patterns.append((n, labels_L))
        if n > 1:
            labels_H = ['H' if i % 2 == 0 else 'L' for i in range(n)]
            patterns.append((n, labels_H))
    
    # Also add some non-alternating patterns for variety
    if max_layers >= 5:
        patterns.append((5, ['L', 'L', 'H', 'H', 'L']))
        patterns.append((5, ['H', 'H', 'L', 'L', 'H']))
    if max_layers >= 7:
        patterns.append((7, ['L', 'L', 'H', 'H', 'L', 'L', 'H']))
        patterns.append((7, ['H', 'H', 'L', 'L', 'H', 'H', 'L']))
    
    # Sort patterns by number of layers (try smaller first for quick wins)
    patterns.sort(key=lambda x: x[0])
    
    # Time budget per pattern
    total_patterns = len(patterns)
    base_time_per_pattern = 10.0  # seconds
    
    for n_layers, labels in patterns:
        elapsed = time.time() - start_time
        if elapsed > deadline - 5:
            break
        
        remaining_patterns = total_patterns - patterns.index((n_layers, labels))
        remaining_time = deadline - 5 - elapsed
        time_budget = max(2.0, remaining_time / max(1, remaining_patterns))
        
        try:
            indices = [materials[label] for label in labels]
            bounds = [(0.0, max_layer_nm)] * n_layers
            
            def obj(x):
                return _objective(x, indices, wavelengths_nm, ambient_idx, substrate_idx, 
                                  max_layer_nm, max_total_nm)
            
            # Multiple starting points
            start_points = []
            
            # Quarter-wave starting point
            x0_qw = np.array([center_wl / (4.0 * n) for n in indices])
            start_points.append(x0_qw)
            
            # Uniform thickness starting point
            x0_uniform = np.full(n_layers, max_total_nm / n_layers * 0.5)
            start_points.append(x0_uniform)
            
            # Random starting points
            np.random.seed(n_layers * 100 + len(labels))
            for _ in range(3):
                x0_rand = np.random.uniform(0, max_layer_nm, size=n_layers)
                if np.sum(x0_rand) > max_total_nm:
                    x0_rand *= max_total_nm / np.sum(x0_rand)
                start_points.append(x0_rand)
            
            best_x_for_pattern = None
            best_f_for_pattern = np.inf
            
            for sp_idx, x0 in enumerate(start_points):
                sp_elapsed = time.time() - start_time
                if sp_elapsed > deadline - 5:
                    break
                
                # Time allocation: Nelder-Mead gets 70%, gradient polish gets 30%
                nm_deadline = time.time() + time_budget * 0.7
                gd_deadline = deadline - 2
                
                # Nelder-Mead optimization
                x_nm, f_nm = _nelder_mead(obj, x0, bounds, max_iter=3000, tol=1e-10, 
                                          time_deadline=nm_deadline)
                
                if f_nm < best_f_for_pattern:
                    best_f_for_pattern = f_nm
                    best_x_for_pattern = x_nm.copy()
                
                # Gradient descent polish
                if time.time() < gd_deadline:
                    x_gd, f_gd = _polish_with_gradient_descent(obj, x_nm, bounds, 
                                                               max_iter=1000, 
                                                               time_deadline=gd_deadline)
                    if f_gd < best_f_for_pattern:
                        best_f_for_pattern = f_gd
                        best_x_for_pattern = x_gd.copy()
            
            # Update global best
            if best_x_for_pattern is not None and best_f_for_pattern < best_score:
                best_score = best_f_for_pattern
                best_labels = labels
                best_thicknesses = best_x_for_pattern.tolist()
                
                # Early termination if excellent
                if best_score < 0.3:
                    break
                    
        except Exception:
            continue
    
    # Fallback to single quarter-wave layer if nothing found
    if best_labels is None:
        return ["L"], [center_wl / (4.0 * materials["L"])]
    
    # Final constraint enforcement
    thicknesses = [max(0, min(t, max_layer_nm)) for t in best_thicknesses]
    total = sum(thicknesses)
    if total > max_total_nm:
        scale = max_total_nm / total
        thicknesses = [t * scale for t in thicknesses]
    
    return best_labels, thicknesses
