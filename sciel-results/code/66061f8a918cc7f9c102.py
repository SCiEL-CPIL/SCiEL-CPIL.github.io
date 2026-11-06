"""Broadband antireflection coating design via multi-start Nelder-Mead and Gradient Descent polish."""
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

def _objective(params, indices, wavelengths_nm, ambient_idx, substrate_idx, max_layer_nm, max_total_nm):
    """
    Objective function: mean reflectance in percent with penalty for constraint violations.
    """
    thicknesses = np.clip(params, 0, max_layer_nm)
    
    total = np.sum(thicknesses)
    if total > max_total_nm:
        # Strong penalty for exceeding total thickness
        penalty = (total - max_total_nm) * 0.5
        return 100.0 + penalty
    
    R = _compute_reflectance(wavelengths_nm, indices, thicknesses, ambient_idx, substrate_idx)
    return np.mean(R) * 100.0

def _nelder_mead(f, x0, bounds, maxiter=1000, xatol=1e-8, fatol=1e-10, time_deadline=None):
    """
    Simplex-based Nelder-Mead optimizer with bounds enforcement.
    """
    n = len(x0)
    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    sigma = 0.5
    
    # Initialize simplex
    simplex = [np.array(x0, dtype=float)]
    for i in range(n):
        xi = x0.copy()
        # Small step in each direction
        step = 0.1 * (bounds[i][1] - bounds[i][0]) if bounds[i][1] > bounds[i][0] else 0.1
        xi[i] += step
        simplex.append(xi)
    
    # Enforce bounds
    for i in range(len(simplex)):
        for j in range(n):
            simplex[i][j] = np.clip(simplex[i][j], bounds[j][0], bounds[j][1])
    
    # Evaluate
    fvals = np.array([f(simplex[i]) for i in range(len(simplex))])
    
    for _ in range(maxiter):
        if time_deadline is not None and time.time() > time_deadline:
            break
            
        # Sort by function value
        idx = np.argsort(fvals)
        simplex = [simplex[i] for i in idx]
        fvals = fvals[idx]
        
        # Check convergence
        if abs(fvals[0] - fvals[-1]) < fatol:
            break
        if np.max(np.abs(simplex[0] - simplex[-1])) < xatol:
            break
        
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        xr = centroid + alpha * (centroid - simplex[-1])
        xr = np.clip(xr, [b[0] for b in bounds], [b[1] for b in bounds])
        fr = f(xr)
        
        if fr < fvals[0]:
            # Expansion
            xe = centroid + gamma * (xr - centroid)
            xe = np.clip(xe, [b[0] for b in bounds], [b[1] for b in bounds])
            fe = f(xe)
            if fe < fr:
                simplex[-1] = xe
                fvals[-1] = fe
            else:
                simplex[-1] = xr
                fvals[-1] = fr
        elif fr < fvals[-2]:
            simplex[-1] = xr
            fvals[-1] = fr
        else:
            # Contraction
            xc = centroid + rho * (simplex[-1] - centroid)
            xc = np.clip(xc, [b[0] for b in bounds], [b[1] for b in bounds])
            fc = f(xc)
            if fc < fvals[-1]:
                simplex[-1] = xc
                fvals[-1] = fc
            else:
                # Shrink
                for i in range(1, len(simplex)):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    simplex[i] = np.clip(simplex[i], [b[0] for b in bounds], [b[1] for b in bounds])
                    fvals[i] = f(simplex[i])
    
    best_idx = np.argmin(fvals)
    return simplex[best_idx], fvals[best_idx]

def _gradient_descent_polish(f, x0, bounds, max_iter=500, time_deadline=None):
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
    
    lr = 0.5 # Initial learning rate
    for it in range(max_iter):
        if time_deadline is not None and time.time() > time_deadline:
            break
            
        grad = numerical_gradient(x)
        if np.max(np.abs(grad)) < 1e-9:
            break
        
        direction = -grad
        norm_dir = np.linalg.norm(direction)
        if norm_dir < 1e-10:
            break
        direction = direction / norm_dir
        
        # Backtracking line search
        step = lr
        f_old = fval
        improved = False
        for ls_iter in range(30):
            x_trial = x + step * direction
            x_trial = np.clip(x_trial, lo_bounds, hi_bounds)
            f_trial = f(x_trial)
            
            # Armijo condition
            if f_trial < f_old - 1e-4 * step * np.dot(grad, direction):
                x = x_trial
                fval = f_trial
                improved = True
                break
            else:
                step *= 0.5
        
        if improved:
            # Increase learning rate slightly if we made good progress
            lr = min(lr * 1.2, 2.0)
        else:
            # Decrease learning rate if stuck
            lr = max(lr * 0.5, 0.01)
            
    return x, fval

def design(problem):
    """
    Design a broadband antireflection coating.
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
    
    wavelengths_nm = np.linspace(band_nm[0], band_nm[1], 401)
    center_wl = 0.5 * (band_nm[0] + band_nm[1])
    
    best_result = None
    best_score = np.inf
    
    # Generate candidate label patterns
    # Focus on odd numbers of layers, alternating L and H
    # 3, 5, 7, 9, 11, 13 layers are good candidates
    candidates = []
    for n in [3, 5, 7, 9, 11, 13]:
        if n <= max_layers:
            # Start with L
            labels_l = ['L' if i % 2 == 0 else 'H' for i in range(n)]
            candidates.append(labels_l)
            # Start with H
            labels_h = ['H' if i % 2 == 0 else 'L' for i in range(n)]
            candidates.append(labels_h)
            
    # Also add 20 layers if allowed, but maybe less effective due to constraints
    if max_layers >= 19:
        labels_19 = ['L' if i % 2 == 0 else 'H' for i in range(19)]
        candidates.append(labels_19)
        
    # Phase 1: Broad Search
    # Run a quick optimization for each candidate to find the most promising ones
    quick_results = []
    time_per_candidate_quick = 2.0 # 2 seconds per candidate in phase 1
    
    for labels in candidates:
        if time.time() > deadline - 30.0: # Keep 30s for phase 2
            break
            
        indices = [materials[label] for label in labels]
        n_layers = len(labels)
        bounds = [(0, max_layer_nm)] * n_layers
        
        def obj(x):
            return _objective(x, indices, wavelengths_nm, ambient_idx, substrate_idx, max_layer_nm, max_total_nm)
        
        # Initial guess: quarter-wave
        x0 = np.array([center_wl / (4.0 * n) for n in indices])
        
        # Run quick Nelder-Mead
        try:
            x_best, f_best = _nelder_mead(obj, x0, bounds, maxiter=200, time_deadline=start_time + time_per_candidate_quick)
            # Clip and check
            x_best = np.clip(x_best, 0, max_layer_nm)
            if np.sum(x_best) > max_total_nm:
                x_best = x_best * (max_total_nm / np.sum(x_best))
            f_best = obj(x_best)
            
            if f_best < best_score:
                best_score = f_best
                best_result = (labels, x_best.tolist())
                
            quick_results.append((f_best, labels, x_best))
        except Exception:
            continue
            
    # Phase 2: Deep Polish
    # Sort quick results by score and take the top 3
    quick_results.sort(key=lambda x: x[0])
    top_candidates = quick_results[:3]
    
    # Allocate remaining time to top candidates
    remaining_time = deadline - time.time() - 5.0 # 5s safety
    if remaining_time > 0 and top_candidates:
        time_per_candidate_deep = remaining_time / len(top_candidates)
        
        for score, labels, x_init in top_candidates:
            if time.time() > deadline - 5.0:
                break
                
            indices = [materials[label] for label in labels]
            n_layers = len(labels)
            bounds = [(0, max_layer_nm)] * n_layers
            
            def obj(x):
                return _objective(x, indices, wavelengths_nm, ambient_idx, substrate_idx, max_layer_nm, max_total_nm)
            
            # Run deep Nelder-Mead
            try:
                # Use a longer deadline for NM
                nm_deadline = time.time() + time_per_candidate_deep * 0.7
                x_nm, f_nm = _nelder_mead(obj, x_init, bounds, maxiter=1000, time_deadline=nm_deadline)
                
                # Polish with Gradient Descent
                gd_deadline = time.time() + time_per_candidate_deep * 0.3
                x_gd, f_gd = _gradient_descent_polish(obj, x_nm, bounds, max_iter=200, time_deadline=gd_deadline)
                
                # Final clip and check
                x_gd = np.clip(x_gd, 0, max_layer_nm)
                if np.sum(x_gd) > max_total_nm:
                    x_gd = x_gd * (max_total_nm / np.sum(x_gd))
                f_final = obj(x_gd)
                
                if f_final < best_score:
                    best_score = f_final
                    best_result = (labels, x_gd.tolist())
            except Exception:
                continue
                
    # If no result found, fall back to single quarter-wave layer
    if best_result is None:
        center_wl = 0.5 * (band_nm[0] + band_nm[1])
        return ["L"], [center_wl / (4.0 * materials["L"])]
    
    labels, thicknesses = best_result
    
    # Final constraint enforcement
    thicknesses = [max(0, min(t, max_layer_nm)) for t in thicknesses]
    total = sum(thicknesses)
    if total > max_total_nm:
        scale = max_total_nm / total
        thicknesses = [t * scale for t in thicknesses]
    
    return labels, thicknesses
