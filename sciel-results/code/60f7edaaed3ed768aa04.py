"""Broadband antireflection coating design via Differential Evolution and Nelder-Mead polish."""
import numpy as np
import time

def _compute_reflectance(wavelengths_nm, indices, thicknesses_nm, ambient_idx, substrate_idx):
    """
    Compute reflectance (fraction, 0-1) at each wavelength using the characteristic matrix method.
    wavelengths_nm: array of wavelengths in nm
    indices: list/array of refractive indices for each layer
    thicknesses_nm: list/array of physical thicknesses in nm
    ambient_idx: refractive index of ambient (air)
    substrate_idx: refractive index of substrate
    Returns: array of reflectance values (0 to 1)
    """
    n_layers = len(indices)
    if n_layers == 0:
        r0 = (ambient_idx - substrate_idx) / (ambient_idx + substrate_idx)
        return np.full(len(wavelengths_nm), r0 * r0)
    
    # Convert to optical path lengths: n * d
    optical_thick = np.array(thicknesses_nm) * np.array(indices)
    
    # k = 2*pi / lambda (in 1/nm)
    k = 2.0 * np.pi / wavelengths_nm  # shape (N,)
    
    # M is 2x2, represented as four arrays of shape (N,)
    M00 = np.ones(len(wavelengths_nm), dtype=complex)
    M01 = np.zeros(len(wavelengths_nm), dtype=complex)
    M10 = np.zeros(len(wavelengths_nm), dtype=complex)
    M11 = np.ones(len(wavelengths_nm), dtype=complex)
    
    for i in range(n_layers):
        n_i = indices[i]
        d_i = thicknesses_nm[i]
        delta = k * n_i * d_i  # shape (N,)
        
        cos_d = np.cos(delta)
        sin_d = np.sin(delta)
        
        # Layer matrix B
        B00 = cos_d
        B01 = 1j * sin_d / n_i
        B10 = 1j * n_i * sin_d
        B11 = cos_d
        
        # M_new = M * B
        new_M00 = M00 * B00 + M01 * B10
        new_M01 = M00 * B01 + M01 * B11
        new_M10 = M10 * B00 + M11 * B10
        new_M11 = M10 * B01 + M11 * B11
        
        M00, M01, M10, M11 = new_M00, new_M01, new_M10, new_M11
    
    # Admittance Y_in
    Y_out = substrate_idx
    Y_in = (M10 + M11 * Y_out) / (M00 + M01 * Y_out)
    
    # Reflectance
    r = (ambient_idx - Y_in) / (ambient_idx + Y_in)
    R = np.abs(r) ** 2
    
    return R

def _objective(params, problem, wavelengths_nm, max_layers, max_total_nm, max_layer_nm):
    """
    Objective function: mean reflectance in percent + penalties.
    params: flat array of physical thicknesses
    """
    # Clip to valid range
    thicknesses = np.clip(params, 0, max_layer_nm)
    
    # Check total thickness constraint
    total_thick = np.sum(thicknesses)
    if total_thick > max_total_nm:
        # Penalize heavily if total thickness is exceeded
        return 100.0 + (total_thick - max_total_nm) * 0.1
    
    indices = [problem["materials"][label] for label in problem["labels"]]
    
    R = _compute_reflectance(wavelengths_nm, indices, thicknesses, 
                             problem["ambient_index"], problem["substrate_index"])
    
    return np.mean(R) * 100.0

def _nelder_mead(f, x0, bounds, max_iter=500, tol=1e-12, time_limit=10.0, start_time=None):
    """
    Nelder-Mead simplex algorithm for local polishing.
    """
    if start_time is None:
        start_time = time.time()
    
    n = len(x0)
    simplex = np.zeros((n + 1, n))
    simplex[0] = x0
    
    for i in range(n):
        simplex[i + 1] = x0.copy()
        step = 0.05 * (bounds[i][1] - bounds[i][0]) if bounds[i][1] > bounds[i][0] else 0.05
        simplex[i + 1][i] += step
    
    fvals = np.array([f(simplex[i], bounds, start_time) for i in range(n + 1)])
    
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
        f_r = f(x_r, bounds, start_time)
        
        if f_r < fvals[0]:
            x_e = centroid + gamma * (x_r - centroid)
            x_e = np.clip(x_e, [b[0] for b in bounds], [b[1] for b in bounds])
            f_e = f(x_e, bounds, start_time)
            
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
            f_c = f(x_c, bounds, start_time)
            
            if f_c < fvals[-1]:
                simplex[-1] = x_c
                fvals[-1] = f_c
            else:
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    simplex[i] = np.clip(simplex[i], [b[0] for b in bounds], [b[1] for b in bounds])
                    fvals[i] = f(simplex[i], bounds, start_time)
    
    best_idx = np.argmin(fvals)
    return simplex[best_idx], fvals[best_idx]

def _differential_evolution(f, bounds, pop_size=60, max_iter=80, CR=0.8, F=0.8, time_limit=40.0, start_time=None):
    """
    Differential Evolution global optimizer.
    """
    if start_time is None:
        start_time = time.time()
    
    n = len(bounds)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    
    # Initialize population
    pop = np.random.uniform(lo, hi, size=(pop_size, n))
    # Ensure one individual is a good heuristic start (quarter-wave stack)
    pop[0] = np.array([0.25 * 600.0 / ((hi[i]+lo[i])/2) for i in range(n)]) # Rough heuristic, will be clipped/evaluated
    
    fvals = np.array([f(pop[i], bounds, start_time) for i in range(pop_size)])
    best_idx = np.argmin(fvals)
    best_x = pop[best_idx].copy()
    best_f = fvals[best_idx]
    
    for gen in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        new_pop = pop.copy()
        new_fvals = fvals.copy()
        
        for i in range(pop_size):
            if i == best_idx:
                continue
            
            # Select three distinct random individuals
            r1, r2, r3 = np.random.choice(pop_size, 3, replace=False)
            while r1 == i or r2 == i or r3 == i:
                r1, r2, r3 = np.random.choice(pop_size, 3, replace=False)
            
            # Mutation
            v = pop[r1] + F * (pop[r2] - pop[r3])
            v = np.clip(v, lo, hi)
            
            # Crossover
            trial = pop[i].copy()
            j_rand = np.random.randint(n)
            for j in range(n):
                if np.random.rand() < CR or j == j_rand:
                    trial[j] = v[j]
            
            trial = np.clip(trial, lo, hi)
            
            # Selection
            f_trial = f(trial, bounds, start_time)
            if f_trial < fvals[i]:
                new_pop[i] = trial
                new_fvals[i] = f_trial
                
                if f_trial < best_f:
                    best_f = f_trial
                    best_x = trial.copy()
                    best_idx = i
        
        pop = new_pop
        fvals = new_fvals
        
        # Convergence check
        if np.std(fvals) < 1e-8:
            break
    
    return best_x, best_f

def design(problem):
    """
    Design a broadband antireflection coating.
    Returns: (labels, thicknesses)
    """
    start_time = time.time()
    time_limit = 140.0  # Safety margin before 150s kill
    
    # Problem parameters
    ambient_idx = problem["ambient_index"]
    substrate_idx = problem["substrate_index"]
    materials = problem["materials"]
    band_nm = problem["band_nm"]
    max_layers = problem.get("max_layers", 20)
    max_total_nm = problem.get("max_total_nm", 1000)
    max_layer_nm = problem.get("max_layer_nm", 400)
    
    # Wavelengths for evaluation
    wavelengths_nm = np.linspace(band_nm[0], band_nm[1], 401)
    
    # We will optimize a fixed number of layers, say 12, and then prune.
    # This allows the optimizer to decide the effective number of layers.
    n_opt_layers = 12
    
    # Create alternating labels for the optimization space
    labels = []
    for i in range(n_opt_layers):
        if i % 2 == 0:
            labels.append('L')
        else:
            labels.append('H')
    
    problem_local = dict(problem)
    problem_local["labels"] = labels
    
    # Bounds for physical thicknesses
    bounds = [(0, max_layer_nm)] * n_opt_layers
    
    # Objective function wrapper
    def obj_func(x, bounds_local, start_time_local):
        return _objective(x, problem_local, wavelengths_nm, max_layers, max_total_nm, max_layer_nm)
    
    best_labels = None
    best_thicknesses = None
    best_score = np.inf
    
    # 1. Global Search with Differential Evolution
    try:
        x_de, f_de = _differential_evolution(obj_func, bounds, 
                                             pop_size=50, max_iter=60, 
                                             time_limit=60.0, start_time=start_time)
    except Exception:
        x_de = np.zeros(n_opt_layers)
        f_de = np.inf
    
    # 2. Polish the best candidate from DE
    # Remove zero-thickness layers for polishing to reduce dimensionality
    active_mask = x_de > 1.0
    if np.sum(active_mask) > 0:
        x_active = x_de[active_mask]
        labels_active = [labels[i] for i in range(n_opt_layers) if active_mask[i]]
        bounds_active = [bounds[i] for i in range(n_opt_layers) if active_mask[i]]
        
        problem_active = dict(problem_local)
        problem_active["labels"] = labels_active
        
        def obj_func_active(x, bounds_local, start_time_local):
            return _objective(x, problem_active, wavelengths_nm, max_layers, max_total_nm, max_layer_nm)
        
        try:
            x_polished, f_polished = _nelder_mead(obj_func_active, x_active, bounds_active,
                                                   max_iter=300, time_limit=30.0, start_time=start_time)
            
            if f_polished < best_score:
                best_score = f_polished
                best_labels = labels_active
                best_thicknesses = x_polished
        except Exception:
            pass
    else:
        # Fallback if DE failed completely
        best_score = f_de
        best_labels = labels
        best_thicknesses = x_de

    # 3. Try a few heuristic starts with Nelder-Mead if time permits
    # Heuristic: Quarter-wave stack at center wavelength
    center_wl = 0.5 * (band_nm[0] + band_nm[1])
    
    for n_heur in [3, 5, 7]:
        if time.time() - start_time > time_limit - 10:
            break
            
        heur_labels = []
        heur_thick = []
        for i in range(n_heur):
            if i % 2 == 0:
                lab = 'L'
                n_idx = materials['L']
            else:
                lab = 'H'
                n_idx = materials['H']
            heur_labels.append(lab)
            heur_thick.append(center_wl / (4.0 * n_idx))
        
        heur_labels = heur_labels[:n_heur]
        heur_thick = np.array(heur_thick[:n_heur])
        
        problem_heur = dict(problem_local)
        problem_heur["labels"] = heur_labels
        bounds_heur = [(0, max_layer_nm)] * n_heur
        
        def obj_func_heur(x, bounds_local, start_time_local):
            return _objective(x, problem_heur, wavelengths_nm, max_layers, max_total_nm, max_layer_nm)
        
        try:
            x_h, f_h = _nelder_mead(obj_func_heur, heur_thick, bounds_heur,
                                     max_iter=200, time_limit=10.0, start_time=start_time)
            if f_h < best_score:
                best_score = f_h
                best_labels = heur_labels
                best_thicknesses = x_h
        except Exception:
            pass
    
    # Finalize result
    if best_labels is None:
        # Absolute fallback: single layer
        return ["L"], [center_wl / (4.0 * materials["L"])]
    
    # Prune any remaining negligible layers
    final_labels = []
    final_thick = []
    for i in range(len(best_labels)):
        if best_thicknesses[i] > 0.5:
            final_labels.append(best_labels[i])
            final_thick.append(best_thicknesses[i])
    
    # Ensure constraints
    final_thick = np.clip(final_thick, 0, max_layer_nm)
    if np.sum(final_thick) > max_total_nm:
        final_thick = final_thick * (max_total_nm / np.sum(final_thick))
    
    # If pruning resulted in 0 layers, fallback
    if len(final_labels) == 0:
        return ["L"], [center_wl / (4.0 * materials["L"])]
        
    return final_labels, final_thick.tolist()
