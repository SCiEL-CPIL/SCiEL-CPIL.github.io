"""
Broadband antireflection coating design via differential evolution + Nelder-Mead polish.

Strategy:
- Use differential evolution for robust global search over layer thicknesses
- Polish the best result with Nelder-Mead for local convergence
- Try multiple layer counts and alternating patterns
- Enforce constraints: thickness in [0, max_layer_nm], sum <= max_total_nm
"""
import numpy as np
import time
import math

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
    if np.any(thicknesses < -1e-9) or np.any(thicknesses > max_layer_nm + 1e-9):
        return 100.0
    if np.sum(thicknesses) > max_total_nm + 1e-9:
        return 100.0
    
    thicknesses_clipped = np.clip(thicknesses, 0.0, max_layer_nm)
    if np.sum(thicknesses_clipped) > max_total_nm:
        return 100.0
    
    R = _compute_reflectance(wavelengths_nm, indices, thicknesses_clipped, 
                             ambient_idx, substrate_idx)
    return np.mean(R) * 100.0

def _nelder_mead(f, x0, bounds, max_iter=3000, tol=1e-12, time_deadline=None):
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

def _differential_evolution(f, bounds, pop_size=50, max_gen=200, F=0.8, CR=0.9,
                            time_deadline=None, seed=42):
    """
    Differential evolution for global optimization.
    """
    n = len(bounds)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    
    rng = np.random.RandomState(seed)
    
    # Initialize population
    pop = rng.uniform(lo, hi, size=(pop_size, n))
    fvals = np.array([f(pop[i]) for i in range(pop_size)])
    
    best_idx = np.argmin(fvals)
    best_x = pop[best_idx].copy()
    best_f = fvals[best_idx]
    
    for gen in range(max_gen):
        if time_deadline is not None and time.time() > time_deadline:
            break
        
        # Update best
        current_best_idx = np.argmin(fvals)
        if fvals[current_best_idx] < best_f:
            best_f = fvals[current_best_idx]
            best_x = pop[current_best_idx].copy()
        
        # Generate offspring
        new_pop = pop.copy()
        new_fvals = fvals.copy()
        
        for i in range(pop_size):
            if time_deadline is not None and time.time() > time_deadline:
                break
            
            # Select three distinct random individuals
            while True:
                r1, r2, r3 = rng.choice(pop_size, size=3, replace=False)
                if r1 != r2 and r2 != r3 and r1 != r3:
                    break
            
            # Mutation
            v = pop[r1] + F * (pop[r2] - pop[r3])
            v = np.clip(v, lo, hi)
            
            # Crossover
            j_rand = rng.randint(n)
            for j in range(n):
                if rng.random() < CR or j == j_rand:
                    v[j] = v[j]
                else:
                    v[j] = pop[i][j]
            
            # Selection
            f_v = f(v)
            if f_v < fvals[i]:
                new_pop[i] = v
                new_fvals[i] = f_v
        
        pop = new_pop
        fvals = new_fvals
    
    return best_x, best_f

def _polish(f, x0, bounds, time_deadline=None):
    """
    Polish with Nelder-Mead.
    """
    return _nelder_mead(f, x0, bounds, max_iter=2000, tol=1e-12, time_deadline=time_deadline)

def _design_for_layer_count(n_layers, materials, ambient_idx, substrate_idx, 
                            wavelengths_nm, center_wl, max_total_nm, max_layer_nm,
                            start_char, time_deadline):
    """
    Design for a specific layer count and starting material.
    """
    labels = [start_char if i % 2 == 0 else ('H' if start_char == 'L' else 'L') 
              for i in range(n_layers)]
    indices = [materials[label] for label in labels]
    
    bounds = [(0.0, float(max_layer_nm))] * n_layers
    
    def make_f(indices):
        def f(x):
            return _objective(x, indices, wavelengths_nm, ambient_idx, substrate_idx,
                              max_layer_nm, max_total_nm)
        return f
    
    f = make_f(indices)
    
    # Initial guess: quarter-wave stack
    x0 = np.array([center_wl / (4.0 * n_i) for n_i in indices])
    if np.sum(x0) > max_total_nm:
        x0 = x0 * (max_total_nm / np.sum(x0))
    x0 = np.clip(x0, 0, max_layer_nm)
    
    # Differential evolution
    de_deadline = time.time() + 8.0
    try:
        x_de, f_de = _differential_evolution(f, bounds, pop_size=40, max_gen=150, 
                                              F=0.7, CR=0.85, time_deadline=de_deadline)
    except Exception:
        x_de, f_de = x0, f(x0)
    
    # Polish with Nelder-Mead
    nm_deadline = min(time.time() + 5.0, time_deadline)
    try:
        x_nm, f_nm = _polish(f, x_de, bounds, time_deadline=nm_deadline)
        if f_nm < f_de:
            x_de, f_de = x_nm, f_nm
    except Exception:
        pass
    
    return labels, x_de, f_de

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
    
    # Try different layer counts and starting materials
    layer_configs = []
    for n_layers in [7, 9, 11, 13, 15]:
        if n_layers > max_layers:
            continue
        for start_char in ['L', 'H']:
            layer_configs.append((n_layers, start_char))
    
    for n_layers, start_char in layer_configs:
        if time.time() > deadline - 20:
            break
        
        try:
            labels, thicknesses, score = _design_for_layer_count(
                n_layers, materials, ambient_idx, substrate_idx,
                wavelengths_nm, center_wl, max_total_nm, max_layer_nm,
                start_char, deadline - 10
            )
            
            if score < best_score:
                best_score = score
                best_labels = labels
                best_thicknesses = thicknesses
        except Exception:
            continue
    
    # Fallback if nothing worked
    if best_labels is None:
        best_labels = ['L']
        best_thicknesses = np.array([center_wl / (4.0 * materials['L'])])
    
    # Final constraint check
    thicknesses = np.array(best_thicknesses)
    thicknesses = np.clip(thicknesses, 0, max_layer_nm)
    if np.sum(thicknesses) > max_total_nm:
        thicknesses = thicknesses * (max_total_nm / np.sum(thicknesses))
    
    return best_labels, thicknesses.tolist()
