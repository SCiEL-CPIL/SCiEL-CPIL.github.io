"""Broadband antireflection coating design: smooth multi-start local optimization."""
import numpy as np
import time

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

def _objective(thicknesses, indices, wavelengths_nm, ambient_idx, substrate_idx, 
               max_layer_nm, max_total_nm):
    """Objective: mean reflectance in percent."""
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

def _nelder_mead(f, x0, bounds, max_iter=2000, tol=1e-12, time_deadline=None):
    """Nelder-Mead simplex with bounds and adaptive step size."""
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

def _polish_with_gradient_descent(f, x0, bounds, max_iter=1000, time_deadline=None):
    """Gradient descent with numerical gradient and line search for final polish."""
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
        improved = False
        for ls_iter in range(50):
            x_trial = x + step * direction
            x_trial = np.clip(x_trial, lo_bounds, hi_bounds)
            f_trial = f(x_trial)
            
            if f_trial < f_old - 1e-4 * step * np.dot(grad, direction):
                x = x_trial
                fval = f_trial
                improved = True
                break
            else:
                step *= 0.5
        
        if not improved:
            break
        
        if fval < f_old - 1e-3:
            lr = min(lr * 1.5, 10.0)
        elif fval > f_old + 1e-4:
            lr = max(lr * 0.5, 0.01)
    
    return x, fval

def _perturb_and_swap(f, x0, labels, materials, bounds, wavelengths_nm, ambient_idx, substrate_idx,
                       max_layer_nm, max_total_nm, time_deadline, max_iter=200):
    """Local search: perturb thicknesses, swap adjacent materials, add/remove layers."""
    n = len(x0)
    lo_bounds = np.array([b[0] for b in bounds])
    hi_bounds = np.array([b[1] for b in bounds])
    x = np.clip(np.asarray(x0, dtype=float), lo_bounds, hi_bounds)
    fval = f(x)
    current_labels = labels[:]
    
    for it in range(max_iter):
        if time_deadline is not None and time.time() > time_deadline:
            break
        
        best_x = x
        best_f = fval
        best_labels = current_labels[:]
        
        # 1. Perturb thicknesses
        for _ in range(3):
            x_trial = x.copy()
            idx = np.random.randint(n)
            x_trial[idx] += np.random.uniform(-20, 20)
            x_trial = np.clip(x_trial, lo_bounds, hi_bounds)
            if np.sum(x_trial) <= max_total_nm:
                f_trial = f(x_trial)
                if f_trial < best_f:
                    best_f = f_trial
                    best_x = x_trial
        
        # 2. Swap adjacent layers
        if n > 1:
            i = np.random.randint(n - 1)
            labels_trial = current_labels[:]
            labels_trial[i], labels_trial[i+1] = labels_trial[i+1], labels_trial[i]
            indices_trial = [materials[l] for l in labels_trial]
            f_trial = _objective(x, indices_trial, wavelengths_nm, ambient_idx, substrate_idx,
                                 max_layer_nm, max_total_nm)
            if f_trial < best_f:
                best_f = f_trial
                best_labels = labels_trial
        
        # 3. Add a layer (if under max)
        if n < 20 and np.sum(x) < max_total_nm - 10:
            new_label = 'L' if np.random.random() < 0.5 else 'H'
            pos = np.random.randint(n + 1)
            labels_trial = current_labels[:pos] + [new_label] + current_labels[pos:]
            indices_trial = [materials[l] for l in labels_trial]
            x_trial = np.insert(x, pos, 50.0)
            x_trial = np.clip(x_trial, 0, max_layer_nm)
            if np.sum(x_trial) <= max_total_nm:
                f_trial = _objective(x_trial, indices_trial, wavelengths_nm, ambient_idx, substrate_idx,
                                     max_layer_nm, max_total_nm)
                if f_trial < best_f:
                    best_f = f_trial
                    best_x = x_trial
                    best_labels = labels_trial
        
        # 4. Remove a layer (if > 1)
        if n > 1:
            pos = np.random.randint(n)
            labels_trial = current_labels[:pos] + current_labels[pos+1:]
            indices_trial = [materials[l] for l in labels_trial]
            x_trial = np.delete(x, pos)
            if np.sum(x_trial) <= max_total_nm:
                f_trial = _objective(x_trial, indices_trial, wavelengths_nm, ambient_idx, substrate_idx,
                                     max_layer_nm, max_total_nm)
                if f_trial < best_f:
                    best_f = f_trial
                    best_x = x_trial
                    best_labels = labels_trial
        
        if best_f < fval:
            x = best_x
            fval = best_f
            current_labels = best_labels
        else:
            # No improvement, try a larger perturbation
            x_trial = x + np.random.uniform(-10, 10, size=n)
            x_trial = np.clip(x_trial, lo_bounds, hi_bounds)
            if np.sum(x_trial) <= max_total_nm:
                f_trial = f(x_trial)
                if f_trial < fval:
                    x = x_trial
                    fval = f_trial
    
    return x, fval, current_labels

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
    
    # Generate layer patterns - focus on odd numbers of layers for symmetric designs
    patterns = []
    
    # Alternating starting with L (most common for AR coatings)
    for n in range(3, min(max_layers + 1, 21), 2):
        labels = ['L' if i % 2 == 0 else 'H' for i in range(n)]
        patterns.append((n, labels, "alt_L"))
    
    # Alternating starting with H
    for n in range(3, min(max_layers + 1, 21), 2):
        labels = ['H' if i % 2 == 0 else 'L' for i in range(n)]
        patterns.append((n, labels, "alt_H"))
    
    # Prioritize: 5, 7, 9 layers first
    def pattern_priority(p):
        n = p[0]
        if n in [5, 7, 9]:
            return 0
        elif n in [3, 11]:
            return 1
        elif n in [13, 15]:
            return 2
        else:
            return 3
    
    patterns.sort(key=pattern_priority)
    
    # For each pattern, do multi-start optimization
    for n_layers, labels, ptype in patterns:
        if time.time() > deadline - 20:
            break
        
        indices = [materials[label] for label in labels]
        bounds = [(0.0, float(max_layer_nm))] * n_layers
        
        def make_f(indices):
            def f(x):
                return _objective(x, indices, wavelengths_nm, ambient_idx, substrate_idx,
                                  max_layer_nm, max_total_nm)
            return f
        
        f = make_f(indices)
        
        # Multiple initial guesses - focus on physics-based starts
        n_starts = 6
        for start_i in range(n_starts):
            if time.time() > deadline - 15:
                break
            
            # Generate initial guess based on physics
            if start_i == 0:
                # Quarter-wave stack at center wavelength
                x0 = np.array([center_wl / (4.0 * n) for n in indices])
            elif start_i == 1:
                # Quarter-wave at shorter wavelength (toward blue)
                x0 = np.array([band_nm[0] / (4.0 * n) for n in indices])
            elif start_i == 2:
                # Quarter-wave at longer wavelength (toward red)
                x0 = np.array([band_nm[1] / (4.0 * n) for n in indices])
            elif start_i == 3:
                # Weighted average of quarter-wave at different wavelengths
                x0 = np.array([(0.3 * band_nm[0] + 0.4 * center_wl + 0.3 * band_nm[1]) / (4.0 * n) for n in indices])
            elif start_i == 4:
                # Quarter-wave with noise
                x0 = np.array([center_wl / (4.0 * n) for n in indices])
                x0 = x0 * np.random.uniform(0.7, 1.3, size=n_layers)
            else:
                # Random feasible point
                x0 = np.random.uniform(0, max_layer_nm * 0.5, size=n_layers)
                if np.sum(x0) > max_total_nm:
                    x0 = x0 * (max_total_nm / np.sum(x0))
            
            x0 = np.clip(x0, 0, max_layer_nm)
            if np.sum(x0) > max_total_nm:
                x0 = x0 * (max_total_nm / np.sum(x0))
            
            # Evaluate initial
            f0 = f(x0)
            if f0 < best_score:
                best_score = f0
                best_labels = labels.copy()
                best_thicknesses = x0.copy()
            
            if f0 >= 100.0:
                continue
            
            # Phase 1: Nelder-Mead for local refinement
            nm_deadline = time.time() + min(5.0, deadline - 15 - time.time())
            try:
                x_nm, f_nm = _nelder_mead(f, x0, bounds, max_iter=1000, tol=1e-12, 
                                           time_deadline=nm_deadline)
            except Exception:
                x_nm, f_nm = x0, f0
            
            if f_nm < best_score:
                best_score = f_nm
                best_labels = labels.copy()
                best_thicknesses = x_nm.copy()
            
            # Phase 2: Gradient descent for fine polish
            if f_nm < best_score + 0.5 and time.time() < deadline - 10:
                gd_deadline = time.time() + min(3.0, deadline - 10 - time.time())
                try:
                    x_gd, f_gd = _polish_with_gradient_descent(f, x_nm, bounds, max_iter=500, 
                                                               time_deadline=gd_deadline)
                except Exception:
                    x_gd, f_gd = x_nm, f_nm
                
                if f_gd < best_score:
                    best_score = f_gd
                    best_labels = labels.copy()
                    best_thicknesses = x_gd.copy()
            
            # Phase 3: Perturb and swap local search
            if f_gd < best_score + 0.5 and time.time() < deadline - 5:
                ps_deadline = time.time() + min(3.0, deadline - 5 - time.time())
                try:
                    x_ps, f_ps, labels_ps = _perturb_and_swap(
                        f, x_gd, labels, materials, bounds, wavelengths_nm, 
                        ambient_idx, substrate_idx, max_layer_nm, max_total_nm,
                        time_deadline=ps_deadline, max_iter=100)
                except Exception:
                    x_ps, f_ps, labels_ps = x_gd, f_gd, labels
                
                if f_ps < best_score:
                    best_score = f_ps
                    best_labels = labels_ps
                    best_thicknesses = x_ps.copy()
            
            # Early exit if very good
            if best_score < 0.1:
                break
        
        # Early exit if very good
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
