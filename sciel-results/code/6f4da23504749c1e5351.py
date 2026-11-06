"""Broadband antireflection coating design via multi-start coarse-to-fine search with gradient polish."""
import numpy as np
import time

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

def _nelder_mead(f, x0, bounds, max_iter=2000, tol=1e-8, time_deadline=None):
    """
    Nelder-Mead simplex with bounds.
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
        step = 0.05 * (hi - lo) if hi > lo else 0.05
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

def _polish_gradient_descent(f, x0, bounds, max_iter=500, time_deadline=None, tol=1e-10):
    """
    Gradient descent with numerical gradient and backtracking line search.
    This is faster than NM for final precision in smooth landscapes.
    """
    n = len(x0)
    lo_bounds = np.array([b[0] for b in bounds])
    hi_bounds = np.array([b[1] for b in bounds])
    x = np.clip(np.asarray(x0, dtype=float), lo_bounds, hi_bounds)
    fval = f(x)
    
    def numerical_gradient(x):
        eps = 1e-5
        grad = np.zeros(n)
        for i in range(n):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += eps
            x_minus[i] -= eps
            x_plus = np.clip(x_plus, lo_bounds, hi_bounds)
            x_minus = np.clip(x_minus, lo_bounds, hi_bounds)
            grad[i] = (f(x_plus) - f(x_minus)) / (2 * eps)
        return grad
    
    lr = 1.0
    min_lr = 1e-6
    
    for it in range(max_iter):
        if time_deadline is not None and time.time() > time_deadline:
            break
        
        grad = numerical_gradient(x)
        grad_norm = np.linalg.norm(grad)
        if grad_norm < tol:
            break
        
        direction = -grad / grad_norm
        
        # Backtracking line search
        step = lr
        f_old = fval
        success = False
        for ls_iter in range(30):
            x_trial = x + step * direction
            x_trial = np.clip(x_trial, lo_bounds, hi_bounds)
            f_trial = f(x_trial)
            
            # Armijo condition
            if f_trial < f_old - 1e-4 * step * np.dot(grad, direction):
                x = x_trial
                fval = f_trial
                success = True
                break
            else:
                step *= 0.5
        
        if not success:
            # If line search fails, try a smaller step or break
            step = min_lr
            x_trial = x + step * direction
            x_trial = np.clip(x_trial, lo_bounds, hi_bounds)
            f_trial = f(x_trial)
            if f_trial < fval:
                x = x_trial
                fval = f_trial
            else:
                break
        
        # Adaptive learning rate
        if fval < f_old - 1e-3:
            lr = min(lr * 1.5, 10.0)
        elif fval > f_old + 1e-4:
            lr = max(lr * 0.5, min_lr)
    
    return x, fval

def design(problem):
    """
    Design a broadband antireflection coating.
    Returns: (labels, thicknesses)
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    ambient_idx = problem["ambient_index"]
    substrate_idx = problem["substrate_index"]
    materials = problem["materials"]
    band_nm = problem["band_nm"]
    max_layers = problem.get("max_layers", 20)
    max_total_nm = problem.get("max_total_nm", 1000)
    max_layer_nm = problem.get("max_layer_nm", 400)
    
    # Fine grid for final evaluation
    wavelengths_fine = np.linspace(band_nm[0], band_nm[1], 401)
    # Coarse grid for fast multi-start search
    wavelengths_coarse = np.linspace(band_nm[0], band_nm[1], 50)
    
    center_wl = 0.5 * (band_nm[0] + band_nm[1])
    
    best_labels = None
    best_thicknesses = None
    best_score = np.inf
    
    # Generate layer patterns: Alternating L/H is usually optimal for broadband AR
    patterns = []
    for n in range(3, min(max_layers + 1, 21), 2):
        labels_L = ['L' if i % 2 == 0 else 'H' for i in range(n)]
        patterns.append((n, labels_L))
        labels_H = ['H' if i % 2 == 0 else 'L' for i in range(n)]
        patterns.append((n, labels_H))
    
    # Sort by number of layers, prioritizing 5-15 layers which are often optimal
    def pattern_priority(p):
        n = p[0]
        if 5 <= n <= 15:
            return 0
        elif 3 <= n < 5 or 15 < n <= 19:
            return 1
        else:
            return 2
            
    patterns.sort(key=pattern_priority)
    
    # Define objective functions for coarse and fine grids
    def make_f_coarse(indices):
        def f(x):
            return _objective(x, indices, wavelengths_coarse, ambient_idx, substrate_idx,
                              max_layer_nm, max_total_nm)
        return f
    
    def make_f_fine(indices):
        def f(x):
            return _objective(x, indices, wavelengths_fine, ambient_idx, substrate_idx,
                              max_layer_nm, max_total_nm)
        return f
    
    # Multi-start search strategy
    # We will test each pattern with a few random starts, using coarse grid for speed.
    # Then, we polish the best few candidates with fine grid and GD.
    
    candidate_pool = [] # List of (score_coarse, labels, thicknesses)
    
    for n_layers, labels in patterns:
        if time.time() > deadline - 30:
            break
            
        indices = [materials[label] for label in labels]
        bounds = [(0.0, float(max_layer_nm))] * n_layers
        f_coarse = make_f_coarse(indices)
        
        # Try 3-5 random starts for each pattern
        num_starts = 5
        for start_iter in range(num_starts):
            if time.time() > deadline - 30:
                break
                
            # Generate initial guess
            if start_iter == 0:
                # Quarter-wave initial guess
                x0 = np.array([center_wl / (4.0 * n) for n in indices])
                x0 = np.clip(x0, 0, max_layer_nm)
                if np.sum(x0) > max_total_nm:
                    x0 = x0 * (max_total_nm / np.sum(x0))
            else:
                # Random start with some structure
                rng = np.random.RandomState(42 + n_layers + start_iter)
                x0 = rng.uniform(0, max_layer_nm, n_layers)
                # Ensure total thickness constraint
                if np.sum(x0) > max_total_nm:
                    x0 = x0 * (max_total_nm / np.sum(x0))
            
            # Quick NM polish on coarse grid to find local minimum
            nm_deadline = time.time() + max(0.5, min(2.0, (deadline - 30 - time.time()) / num_starts))
            try:
                x_nm, f_nm = _nelder_mead(f_coarse, x0, bounds, max_iter=500, tol=1e-6, time_deadline=nm_deadline)
                if f_nm < 100.0:
                    candidate_pool.append((f_nm, labels, x_nm, indices))
            except Exception:
                pass
        
        # Early exit if we have a very good solution
        if best_score < 0.5 and len(candidate_pool) > 10:
            break
            
    # Sort candidates by coarse score
    if not candidate_pool:
        # Fallback: simple 1-layer
        return ["L"], [center_wl / (4.0 * materials["L"])]
        
    candidate_pool.sort(key=lambda x: x[0])
    
    # Take top K candidates and polish with fine grid
    top_k = min(10, len(candidate_pool))
    polished_candidates = []
    
    for i in range(top_k):
        if time.time() > deadline - 10:
            break
            
        coarse_score, labels, x_coarse, indices = candidate_pool[i]
        f_fine = make_f_fine(indices)
        bounds = [(0.0, float(max_layer_nm))] * len(indices)
        
        # Evaluate on fine grid first
        f_fine_val = f_fine(x_coarse)
        if f_fine_val >= 100.0:
            continue
            
        # Stage 1: NM on fine grid
        nm_deadline = time.time() + max(1.0, min(5.0, (deadline - 10 - time.time()) / max(1, top_k - i)))
        x_nm_fine, f_nm_fine = _nelder_mead(f_fine, x_coarse, bounds, max_iter=1000, tol=1e-8, time_deadline=nm_deadline)
        
        # Stage 2: Gradient Descent on fine grid for final polish
        if time.time() < deadline - 2:
            gd_deadline = time.time() + max(0.5, min(2.0, (deadline - 2 - time.time())))
            x_gd, f_gd = _polish_gradient_descent(f_fine, x_nm_fine, bounds, max_iter=200, time_deadline=gd_deadline, tol=1e-10)
            if f_gd < f_nm_fine:
                x_final = x_gd
                f_final = f_gd
            else:
                x_final = x_nm_fine
                f_final = f_nm_fine
        else:
            x_final = x_nm_fine
            f_final = f_nm_fine
            
        polished_candidates.append((f_final, labels, x_final))
        
        if f_final < best_score:
            best_score = f_final
            best_labels = labels
            best_thicknesses = x_final.copy()
            
        # Early exit if perfect
        if best_score < 0.1:
            break

    # Final fallback
    if best_labels is None:
        return ["L"], [center_wl / (4.0 * materials["L"])]
    
    # Ensure constraints on final result
    thicknesses = np.clip(best_thicknesses, 0, max_layer_nm)
    if np.sum(thicknesses) > max_total_nm:
        thicknesses = thicknesses * (max_total_nm / np.sum(thicknesses))
    
    return best_labels, thicknesses.tolist()
