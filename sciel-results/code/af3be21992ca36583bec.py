"""
Broadband antireflection coating design via differential evolution + Nelder-Mead polish.

Strategy:
- Use differential evolution for global exploration over the space of thicknesses
- Then use Nelder-Mead simplex for local refinement
- Try multiple layer configurations (different numbers and patterns)
- Select the best result within the time budget
"""
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

def _nelder_mead(f, x0, bounds, max_iter=2000, tol=1e-10, time_deadline=None):
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

def _differential_evolution(f, bounds, pop_size=20, max_iter=100, CR=0.8, F=0.8, time_deadline=None):
    """
    Differential Evolution optimizer.
    """
    n = len(bounds)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    
    # Initialize population
    pop = np.random.uniform(lo, hi, size=(pop_size, n))
    fvals = np.array([f(pop[i]) for i in range(pop_size)])
    
    best_idx = np.argmin(fvals)
    best_x = pop[best_idx].copy()
    best_f = fvals[best_idx]
    
    for it in range(max_iter):
        if time_deadline is not None and time.time() > time_deadline:
            break
        
        for i in range(pop_size):
            if time_deadline is not None and time.time() > time_deadline:
                break
            
            # Select three distinct individuals
            mask = np.ones(pop_size, dtype=bool)
            mask[i] = False
            candidates = np.where(mask)[0]
            a, b, c = np.random.choice(candidates, size=3, replace=False)
            
            # Create trial vector
            v = pop[a] + F * (pop[b] - pop[c])
            v = np.clip(v, lo, hi)
            
            # Crossover
            trial = pop[i].copy()
            j_rand = np.random.randint(0, n)
            for j in range(n):
                if np.random.rand() < CR or j == j_rand:
                    trial[j] = v[j]
            
            # Selection
            f_trial = f(trial)
            if f_trial < fvals[i]:
                pop[i] = trial
                fvals[i] = f_trial
                if f_trial < best_f:
                    best_x = trial.copy()
                    best_f = f_trial
    
    return best_x, best_f

def _design_direct(problem):
    """
    Direct physical construction of broadband AR coating.
    """
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
    
    for n_layers in [7, 9, 11, 13]:
        if n_layers > max_layers:
            continue
        
        labels_1 = ['L' if i % 2 == 0 else 'H' for i in range(n_layers)]
        labels_2 = ['H' if i % 2 == 0 else 'L' for i in range(n_layers)]
        
        for labels in [labels_1, labels_2]:
            indices = [materials[label] for label in labels]
            
            x0 = np.array([center_wl / (4.0 * n_i) for n_i in indices])
            
            if np.sum(x0) > max_total_nm:
                x0 = x0 * (max_total_nm / np.sum(x0))
            x0 = np.clip(x0, 0, max_layer_nm)
            
            f0 = _objective(x0, indices, wavelengths_nm, ambient_idx, substrate_idx,
                           max_layer_nm, max_total_nm)
            
            if f0 < best_score:
                best_score = f0
                best_labels = labels.copy()
                best_thicknesses = x0.copy()
    
    if best_labels is None:
        return ["L"], [center_wl / (4.0 * materials["L"])]
    
    return best_labels, best_thicknesses

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
    
    # Get initial design
    init_labels, init_thicknesses = _design_direct(problem)
    indices = [materials[label] for label in init_labels]
    n_layers = len(indices)
    bounds = [(0.0, float(max_layer_nm))] * n_layers
    
    def make_f(indices):
        def f(x):
            return _objective(x, indices, wavelengths_nm, ambient_idx, substrate_idx,
                              max_layer_nm, max_total_nm)
        return f
    
    f = make_f(indices)
    
    # Run differential evolution for global exploration
    de_deadline = min(start_time + 60, deadline - 80)
    try:
        x_de, f_de = _differential_evolution(f, bounds, pop_size=15, max_iter=80, 
                                              CR=0.8, F=0.7, time_deadline=de_deadline)
    except Exception:
        x_de = np.array(init_thicknesses)
        f_de = _objective(x_de, indices, wavelengths_nm, ambient_idx, substrate_idx,
                         max_layer_nm, max_total_nm)
    
    # Run Nelder-Mead for local refinement
    nm_deadline = min(time.time() + 40, deadline - 10)
    try:
        x_nm, f_nm = _nelder_mead(f, x_de, bounds, max_iter=3000, tol=1e-10, 
                                   time_deadline=nm_deadline)
        if f_nm < f_de:
            x_de = x_nm
            f_de = f_nm
    except Exception:
        pass
    
    # Try a few more layer configurations if time permits
    if time.time() < deadline - 30:
        best_overall_x = x_de.copy()
        best_overall_f = f_de
        best_overall_labels = init_labels.copy()
        
        for n_try in [9, 11]:
            if time.time() > deadline - 25:
                break
            if n_try > max_layers:
                continue
            
            for start_char in ['L', 'H']:
                if time.time() > deadline - 20:
                    break
                
                labels_try = [start_char if i % 2 == 0 else ('H' if start_char == 'L' else 'L') 
                              for i in range(n_try)]
                indices_try = [materials[label] for label in labels_try]
                n_try_layers = len(indices_try)
                bounds_try = [(0.0, float(max_layer_nm))] * n_try_layers
                
                f_try = make_f(indices_try)
                
                x0_try = np.array([center_wl / (4.0 * n_i) for n_i in indices_try])
                if np.sum(x0_try) > max_total_nm:
                    x0_try = x0_try * (max_total_nm / np.sum(x0_try))
                x0_try = np.clip(x0_try, 0, max_layer_nm)
                
                # Quick DE
                de_deadline_try = min(time.time() + 20, deadline - 15)
                try:
                    x_de_try, f_de_try = _differential_evolution(f_try, bounds_try, pop_size=10, 
                                                                  max_iter=40, CR=0.8, F=0.7, 
                                                                  time_deadline=de_deadline_try)
                    
                    # Quick NM
                    nm_deadline_try = min(time.time() + 10, deadline - 5)
                    x_nm_try, f_nm_try = _nelder_mead(f_try, x_de_try, bounds_try, max_iter=1000, 
                                                       tol=1e-10, time_deadline=nm_deadline_try)
                    if f_nm_try < f_de_try:
                        x_de_try = x_nm_try
                        f_de_try = f_nm_try
                    
                    if f_de_try < best_overall_f:
                        best_overall_x = x_de_try.copy()
                        best_overall_f = f_de_try
                        best_overall_labels = labels_try.copy()
                except Exception:
                    pass
        
        if best_overall_f < f_de:
            x_de = best_overall_x
            f_de = best_overall_f
            init_labels = best_overall_labels
    
    # Final constraint check
    thicknesses = np.array(x_de)
    thicknesses = np.clip(thicknesses, 0, max_layer_nm)
    if np.sum(thicknesses) > max_total_nm:
        thicknesses = thicknesses * (max_total_nm / np.sum(thicknesses))
    
    return init_labels, thicknesses.tolist()
