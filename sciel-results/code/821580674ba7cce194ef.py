"""Broadband antireflection coating design via Particle Swarm Optimization with transfer matrix method."""
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

def _objective(params, problem, wavelengths_nm, labels):
    """
    Objective function: mean reflectance in percent.
    params: flat array of thicknesses
    """
    max_layer_nm = problem.get("max_layer_nm", 400)
    max_total_nm = problem.get("max_total_nm", 1000)
    
    thicknesses = np.clip(params, 0, max_layer_nm)
    
    # Penalty for exceeding total thickness
    total_thick = np.sum(thicknesses)
    penalty = 0.0
    if total_thick > max_total_nm:
        penalty = 10.0 * (total_thick - max_total_nm)
    
    indices = [problem["materials"][label] for label in labels]
    
    R = _compute_reflectance(wavelengths_nm, indices, thicknesses, 
                             problem["ambient_index"], problem["substrate_index"])
    
    return np.mean(R) * 100.0 + penalty

def _ps_optimize(f, bounds, n_particles=30, max_iter=80, w=0.9, c1=1.5, c2=1.5,
                 time_limit=30.0, start_time=None, rng=None):
    """
    Particle Swarm Optimization.
    """
    if start_time is None:
        start_time = time.time()
    if rng is None:
        rng = np.random.default_rng(42)
    
    n = len(bounds)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    
    # Initialize particles
    particles = rng.uniform(lo, hi, size=(n_particles, n))
    velocities = rng.uniform(-0.1 * (hi - lo), 0.1 * (hi - lo), size=(n_particles, n))
    
    # Evaluate initial particles
    fvals = np.array([f(particles[i]) for i in range(n_particles)])
    
    personal_best_x = particles.copy()
    personal_best_f = fvals.copy()
    
    global_best_idx = np.argmin(fvals)
    global_best_x = particles[global_best_idx].copy()
    global_best_f = fvals[global_best_idx]
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        for i in range(n_particles):
            # Update velocity
            r1 = rng.random(n)
            r2 = rng.random(n)
            
            velocities[i] = w * velocities[i] + c1 * r1 * (personal_best_x[i] - particles[i]) + c2 * r2 * (global_best_x - particles[i])
            
            # Update position
            particles[i] = particles[i] + velocities[i]
            
            # Clamp to bounds
            particles[i] = np.clip(particles[i], lo, hi)
            
            # Evaluate
            f_val = f(particles[i])
            
            # Update personal best
            if f_val < personal_best_f[i]:
                personal_best_f[i] = f_val
                personal_best_x[i] = particles[i].copy()
                
                # Update global best
                if f_val < global_best_f:
                    global_best_f = f_val
                    global_best_x = particles[i].copy()
        
        # Convergence check
        if np.max(fvals) - np.min(fvals) < 1e-6:
            break
        
        # Re-evaluate all for convergence check
        fvals = np.array([f(particles[i]) for i in range(n_particles)])
    
    return global_best_x, global_best_f

def _nelder_mead(f, x0, bounds, max_iter=500, tol=1e-8, time_limit=10.0, start_time=None):
    """
    Custom Nelder-Mead simplex algorithm with bounds handling.
    """
    if start_time is None:
        start_time = time.time()
    
    n = len(x0)
    simplex = np.zeros((n + 1, n))
    simplex[0] = x0
    
    for i in range(n):
        simplex[i + 1] = x0.copy()
        step = 0.1 * (bounds[i][1] - bounds[i][0]) if bounds[i][1] > bounds[i][0] else 0.1
        simplex[i + 1][i] += step
    
    fvals = np.array([f(simplex[i]) for i in range(n + 1)])
    
    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    sigma = 0.5
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        idx = np.argsort(fvals)
        simplex = simplex[idx]
        fvals = fvals[idx]
        
        if np.max(fvals) - np.min(fvals) < tol:
            break
        
        centroid = np.mean(simplex[:-1], axis=0)
        
        x_r = centroid + alpha * (centroid - simplex[-1])
        x_r = np.clip(x_r, [b[0] for b in bounds], [b[1] for b in bounds])
        f_r = f(x_r)
        
        if f_r < fvals[0]:
            x_e = centroid + gamma * (x_r - centroid)
            x_e = np.clip(x_e, [b[0] for b in bounds], [b[1] for b in bounds])
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
            x_c = np.clip(x_c, [b[0] for b in bounds], [b[1] for b in bounds])
            f_c = f(x_c)
            
            if f_c < fvals[-1]:
                simplex[-1] = x_c
                fvals[-1] = f_c
            else:
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    simplex[i] = np.clip(simplex[i], [b[0] for b in bounds], [b[1] for b in bounds])
                    fvals[i] = f(simplex[i])
    
    best_idx = np.argmin(fvals)
    return simplex[best_idx], fvals[best_idx]

def design(problem):
    """
    Design a broadband antireflection coating.
    Returns: (labels, thicknesses)
    """
    start_time = time.time()
    time_limit = 145.0  # Safety margin for 150s limit
    
    ambient_idx = problem["ambient_index"]
    substrate_idx = problem["substrate_index"]
    materials = problem["materials"]
    band_nm = problem["band_nm"]
    max_layers = problem.get("max_layers", 20)
    max_total_nm = problem.get("max_total_nm", 1000)
    max_layer_nm = problem.get("max_layer_nm", 400)
    
    wavelengths_nm = np.linspace(band_nm[0], band_nm[1], 401)
    
    best_result = None
    best_score = np.inf
    
    # Strategy: Use PSO for global exploration, then Nelder-Mead for local polish.
    # Try a few different layer counts with alternating L/H structure.
    
    # For broadband AR, a common good starting structure is alternating L/H layers.
    # We'll try counts: 15, 12, 18, 10, 13, 17, 11, 19, 14, 20, 16
    
    counts_to_try = [15, 12, 18, 10, 13, 17, 11, 19, 14, 20, 16]
    
    for n_layers in counts_to_try:
        if n_layers > max_layers:
            continue
            
        elapsed = time.time() - start_time
        remaining = time_limit - elapsed - 5.0
        
        if remaining < 2.0:
            break
        
        # Define labels: L H L H ...
        labels = ['L' if i % 2 == 0 else 'H' for i in range(n_layers)]
        
        # Initial guess: Quarter wave at center wavelength
        center_wl = 0.5 * (band_nm[0] + band_nm[1])
        indices = [materials[label] for label in labels]
        init_thicknesses = np.array([center_wl / (4.0 * n) for n in indices])
        
        # Ensure initial guess satisfies total thickness constraint
        if np.sum(init_thicknesses) > max_total_nm:
            init_thicknesses = init_thicknesses * (max_total_nm / np.sum(init_thicknesses))
        
        bounds = [(0, max_layer_nm)] * n_layers
        
        def obj_func(x):
            return _objective(x, problem, wavelengths_nm, labels)
        
        # Allocate time: 70% to PSO, 30% to Nelder-Mead
        ps_time = remaining * 0.7
        nm_time = remaining * 0.3
        
        try:
            # 1. Particle Swarm Optimization
            x_ps, f_ps = _ps_optimize(obj_func, bounds, 
                                       n_particles=30, 
                                       max_iter=80, 
                                       time_limit=ps_time, 
                                       start_time=start_time)
            
            # 2. Nelder-Mead Polish
            x_nm, f_nm = _nelder_mead(obj_func, x_ps, bounds, 
                                       max_iter=500, 
                                       time_limit=nm_time, 
                                       start_time=start_time)
            
            # Final check and clip
            x_final = np.clip(x_nm, 0, max_layer_nm)
            if np.sum(x_final) > max_total_nm:
                x_final = x_final * (max_total_nm / np.sum(x_final))
            
            f_final = obj_func(x_final)
            
            if f_final < best_score:
                best_score = f_final
                best_result = (labels, x_final.tolist())
                
        except Exception:
            continue

    # Fallback if no result found
    if best_result is None:
        labels = ['L']
        thicknesses = [600.0 / (4.0 * materials["L"])]
        return labels, thicknesses
    
    labels, thicknesses = best_result
    
    # Final constraint enforcement
    thicknesses = np.clip(thicknesses, 0, max_layer_nm)
    if np.sum(thicknesses) > max_total_nm:
        thicknesses = thicknesses * (max_total_nm / np.sum(thicknesses))
    
    return labels, thicknesses.tolist()
