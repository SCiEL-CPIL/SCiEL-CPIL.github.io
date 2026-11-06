"""
Broadband antireflection coating design via differential evolution + Nelder-Mead polish.

Strategy:
- Encode the full design (layer types + thicknesses) into a single vector
- Use differential evolution for global exploration over the combined space
- Polish the best result with Nelder-Mead
- Multiple layer counts are tried, and the best overall is returned
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

def _objective_from_vector(x, n_layers, materials, wavelengths_nm, ambient_idx, substrate_idx, 
                           max_layer_nm, max_total_nm):
    """
    Decode the vector x into layer labels and thicknesses, then compute mean reflectance.
    x is of length 2*n_layers: first n_layers are continuous values in [0,1] for layer type,
    last n_layers are thicknesses in [0, max_layer_nm].
    """
    # Decode layer types: values in [0, 0.5) -> L, [0.5, 1) -> H
    type_vals = x[:n_layers]
    thicknesses = x[n_layers:]
    
    # Ensure thicknesses are valid
    thicknesses = np.clip(thicknesses, 0, max_layer_nm)
    
    if np.sum(thicknesses) > max_total_nm:
        # Scale down proportionally
        thicknesses = thicknesses * (max_total_nm / np.sum(thicknesses))
    
    # Determine layer types
    labels = ['L' if v < 0.5 else 'H' for v in type_vals]
    indices = [materials[label] for label in labels]
    
    # Remove zero-thickness layers (they don't affect reflectance)
    valid_mask = thicknesses > 1e-6
    if not np.any(valid_mask):
        r0 = (ambient_idx - substrate_idx) / (ambient_idx + substrate_idx)
        return r0 * r0 * 100.0
    
    labels_valid = [labels[i] for i in range(n_layers) if valid_mask[i]]
    thicknesses_valid = thicknesses[valid_mask]
    indices_valid = [materials[label] for label in labels_valid]
    
    R = _compute_reflectance(wavelengths_nm, indices_valid, thicknesses_valid, 
                             ambient_idx, substrate_idx)
    return np.mean(R) * 100.0

def _differential_evolution(objective, bounds, pop_size=50, max_iter=200, 
                            F=0.8, CR=0.9, time_deadline=None, verbose=False):
    """
    Differential evolution with bounds.
    """
    n = len(bounds)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    span = hi - lo
    
    # Initialize population
    pop = np.random.uniform(lo, hi, size=(pop_size, n))
    fvals = np.array([objective(ind) for ind in pop])
    
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
            candidates = [j for j in range(pop_size) if j != i]
            a, b, c = np.random.choice(candidates, size=3, replace=False)
            
            # Create trial vector
            v = pop[a] + F * (pop[b] - pop[c])
            v = np.clip(v, lo, hi)
            
            # Crossover
            r = np.random.random(n)
            jrand = np.random.randint(n)
            trial = np.where(r < CR, v, pop[i])
            trial[jrand] = v[jrand]
            trial = np.clip(trial, lo, hi)
            
            # Selection
            f_trial = objective(trial)
            if f_trial <= fvals[i]:
                pop[i] = trial
                fvals[i] = f_trial
                if f_trial < best_f:
                    best_f = f_trial
                    best_x = trial.copy()
        
        if verbose and it % 20 == 0:
            print(f"DE iter {it}: best_f = {best_f:.6f}")
    
    return best_x, best_f

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
    
    # Try different layer counts
    for n_layers in [8, 10, 12, 14, 16]:
        if n_layers > max_layers:
            continue
        
        if time.time() > deadline - 30:
            break
        
        # Define bounds: first n_layers are type values in [0,1], 
        # last n_layers are thicknesses in [0, max_layer_nm]
        bounds = [(0.0, 1.0)] * n_layers + [(0.0, float(max_layer_nm))] * n_layers
        
        def make_objective(n_l):
            def obj(x):
                return _objective_from_vector(x, n_l, materials, wavelengths_nm, 
                                              ambient_idx, substrate_idx,
                                              max_layer_nm, max_total_nm)
            return obj
        
        obj = make_objective(n_layers)
        
        # Run differential evolution
        de_deadline = min(time.time() + 40, deadline - 20)
        try:
            x_de, f_de = _differential_evolution(obj, bounds, pop_size=40, max_iter=150,
                                                 F=0.8, CR=0.9, time_deadline=de_deadline)
            
            if f_de < best_score:
                best_score = f_de
                # Decode the result
                type_vals = x_de[:n_layers]
                thicknesses_raw = x_de[n_layers:]
                thicknesses_raw = np.clip(thicknesses_raw, 0, max_layer_nm)
                if np.sum(thicknesses_raw) > max_total_nm:
                    thicknesses_raw = thicknesses_raw * (max_total_nm / np.sum(thicknesses_raw))
                
                labels = ['L' if v < 0.5 else 'H' for v in type_vals]
                
                # Remove zero-thickness layers
                valid_mask = thicknesses_raw > 1e-6
                labels = [labels[i] for i in range(n_layers) if valid_mask[i]]
                thicknesses_raw = thicknesses_raw[valid_mask]
                
                best_labels = labels
                best_thicknesses = thicknesses_raw.tolist()
                
        except Exception:
            pass
        
        # Polish with Nelder-Mead if we have time
        if time.time() < deadline - 10 and best_labels is not None:
            n_polish = len(best_labels)
            if n_polish > 0:
                indices = [materials[label] for label in best_labels]
                bounds_thick = [(0.0, float(max_layer_nm))] * n_polish
                
                def make_f(indices):
                    def f(x):
                        x = np.clip(x, 0, max_layer_nm)
                        if np.sum(x) > max_total_nm:
                            x = x * (max_total_nm / np.sum(x))
                        R = _compute_reflectance(wavelengths_nm, indices, x, 
                                                 ambient_idx, substrate_idx)
                        return np.mean(R) * 100.0
                    return f
                
                f = make_f(indices)
                x0 = np.array(best_thicknesses)
                
                nm_deadline = min(time.time() + 15, deadline - 5)
                try:
                    x_opt, f_opt = _nelder_mead(f, x0, bounds_thick, max_iter=1000, 
                                                 tol=1e-10, time_deadline=nm_deadline)
                    if f_opt < best_score:
                        best_score = f_opt
                        best_thicknesses = x_opt.tolist()
                except Exception:
                    pass
    
    # Fallback: single quarter-wave layer
    if best_labels is None:
        return ["L"], [center_wl / (4.0 * materials["L"])]
    
    # Final constraint check
    thicknesses = np.array(best_thicknesses)
    thicknesses = np.clip(thicknesses, 0, max_layer_nm)
    if np.sum(thicknesses) > max_total_nm:
        thicknesses = thicknesses * (max_total_nm / np.sum(thicknesses))
    
    return best_labels, thicknesses.tolist()
