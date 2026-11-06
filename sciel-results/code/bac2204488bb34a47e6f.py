"""Broadband antireflection coating design via discrete genetic algorithm + continuous Nelder-Mead polish."""
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

def _objective(params, problem, wavelengths_nm):
    """
    Objective function: mean reflectance in percent.
    params: flat array of thicknesses
    """
    max_layer_nm = problem.get("max_layer_nm", 400)
    max_total_nm = problem.get("max_total_nm", 1000)
    
    thicknesses = np.clip(params, 0, max_layer_nm)
    
    if np.sum(thicknesses) > max_total_nm:
        return 100.0 + (np.sum(thicknesses) - max_total_nm)
    
    indices = [problem["materials"][label] for label in problem["labels"]]
    
    R = _compute_reflectance(wavelengths_nm, indices, thicknesses, 
                             problem["ambient_index"], problem["substrate_index"])
    
    return np.mean(R) * 100.0

def _nelder_mead(f, x0, bounds, max_iter=3000, tol=1e-12, time_deadline=None):
    """
    Nelder-Mead simplex with bounds and adaptive step size.
    """
    if time_deadline is None:
        time_deadline = time.time() + 100.0
    
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
        if time.time() > time_deadline:
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

def _ga_discrete_search(problem, wavelengths_nm, time_budget=60.0):
    """
    Genetic algorithm over discrete quarter-wave thickness indices and layer patterns.
    Returns best (labels, thicknesses) found.
    """
    start_time = time.time()
    deadline = start_time + time_budget
    
    materials = problem["materials"]
    max_layers = problem.get("max_layers", 20)
    max_total_nm = problem.get("max_total_nm", 1000)
    max_layer_nm = problem.get("max_layer_nm", 400)
    ambient_idx = problem["ambient_index"]
    substrate_idx = problem["substrate_index"]
    
    # Define discrete thickness candidates (quarter-wave at various wavelengths)
    # Optical thickness = n * d, so d = (lambda/4) / n
    # We'll use a set of candidate optical thicknesses and map to physical thicknesses per layer type
    # Simpler: just use a set of physical thickness candidates
    # Based on quarter-wave at 400-800nm for n=1.38 and n=2.35
    # For n=1.38: d = lambda/(4*1.38) -> for lambda=400-800: d = 72.5 - 145 nm
    # For n=2.35: d = lambda/(4*2.35) -> for lambda=400-800: d = 42.6 - 85.1 nm
    
    # Let's use a grid of thickness values
    # We'll represent each layer as (label, thickness_index)
    # thickness_index maps to a specific thickness value
    
    # Create a set of candidate thicknesses
    # Use quarter-wave at multiple wavelengths
    candidate_thicknesses = []
    for wl in [400, 450, 500, 550, 600, 650, 700, 750, 800]:
        d_L = wl / (4.0 * materials['L'])
        d_H = wl / (4.0 * materials['H'])
        if 0 <= d_L <= max_layer_nm:
            candidate_thicknesses.append(d_L)
        if 0 <= d_H <= max_layer_nm:
            candidate_thicknesses.append(d_H)
    
    # Also add some intermediate values
    for wl in [425, 475, 525, 575, 625, 675, 725, 775]:
        d_L = wl / (4.0 * materials['L'])
        d_H = wl / (4.0 * materials['H'])
        if 0 <= d_L <= max_layer_nm:
            candidate_thicknesses.append(d_L)
        if 0 <= d_H <= max_layer_nm:
            candidate_thicknesses.append(d_H)
    
    # Remove duplicates and sort
    candidate_thicknesses = sorted(set(candidate_thicknesses))
    n_cand = len(candidate_thicknesses)
    
    # GA parameters
    pop_size = 40
    max_gen = 50
    cr = 0.8  # crossover rate
    mutation_rate = 0.1
    
    # Chromosome: (n_layers, [label_1, ..., label_n], [thick_idx_1, ..., thick_idx_n])
    # We'll encode as: first integer is n_layers (3 to max_layers), then n_layers pairs of (label_idx, thick_idx)
    # label_idx: 0 for L, 1 for H
    # thick_idx: 0 to n_cand-1
    
    def decode_chrom(chrom):
        n_layers = int(chrom[0])
        n_layers = min(n_layers, max_layers)
        n_layers = max(n_layers, 1)
        
        labels = []
        thicknesses = []
        for i in range(n_layers):
            label_idx = int(chrom[1 + 2*i])
            thick_idx = int(chrom[2 + 2*i])
            
            label = 'L' if label_idx == 0 else 'H'
            thickness = candidate_thicknesses[min(thick_idx, n_cand-1)]
            
            labels.append(label)
            thicknesses.append(thickness)
        
        return labels, thicknesses
    
    def evaluate_chrom(chrom):
        labels, thicknesses = decode_chrom(chrom)
        
        # Check total thickness constraint
        if np.sum(thicknesses) > max_total_nm:
            return 100.0 + (np.sum(thicknesses) - max_total_nm)
        
        indices = [materials[label] for label in labels]
        R = _compute_reflectance(wavelengths_nm, indices, thicknesses, 
                                  ambient_idx, substrate_idx)
        return np.mean(R) * 100.0
    
    # Initialize population
    population = []
    for i in range(pop_size):
        if i == 0:
            # Start with a good initial guess: alternating L/H with quarter-wave at 600nm
            n_layers = 9
            chrom = [n_layers]
            for j in range(n_layers):
                label_idx = 0 if j % 2 == 0 else 1
                # Find closest quarter-wave at 600nm
                target_d = 600.0 / (4.0 * materials['L' if label_idx == 0 else 'H'])
                thick_idx = np.argmin([abs(t - target_d) for t in candidate_thicknesses])
                chrom.append(label_idx)
                chrom.append(thick_idx)
        else:
            # Random initialization
            n_layers = np.random.randint(3, max_layers + 1)
            chrom = [n_layers]
            for j in range(n_layers):
                label_idx = np.random.randint(0, 2)
                thick_idx = np.random.randint(0, n_cand)
                chrom.append(label_idx)
                chrom.append(thick_idx)
        
        population.append(chrom)
    
    # Evaluate initial population
    fvals = np.array([evaluate_chrom(population[i]) for i in range(pop_size)])
    best_idx = np.argmin(fvals)
    best_chrom = population[best_idx].copy()
    best_score = fvals[best_idx]
    
    # GA loop
    for gen in range(max_gen):
        if time.time() > deadline:
            break
        
        # Sort population by fitness
        idx = np.argsort(fvals)
        population = [population[i] for i in idx]
        fvals = fvals[idx]
        
        # Elitism: keep top 10%
        elite_size = max(1, pop_size // 10)
        new_population = [population[i].copy() for i in range(elite_size)]
        
        # Generate rest of population
        while len(new_population) < pop_size:
            if time.time() > deadline:
                break
            
            # Tournament selection
            def tournament_select():
                contenders = np.random.choice(pop_size, size=3, replace=False)
                best_contender = min(contenders, key=lambda i: fvals[i])
                return population[best_contender]
            
            parent1 = tournament_select()
            parent2 = tournament_select()
            
            # Crossover: uniform crossover
            child = parent1.copy()
            for j in range(1, len(parent1)):
                if np.random.random() < cr:
                    child[j] = parent2[j]
                else:
                    child[j] = parent1[j]
            
            # Mutation
            if np.random.random() < mutation_rate:
                # Mutate n_layers
                if np.random.random() < 0.3:
                    child[0] = np.random.randint(3, max_layers + 1)
                    # Regenerate the rest
                    n_layers = int(child[0])
                    n_layers = min(n_layers, max_layers)
                    n_layers = max(n_layers, 1)
                    child = [n_layers]
                    for j in range(n_layers):
                        label_idx = np.random.randint(0, 2)
                        thick_idx = np.random.randint(0, n_cand)
                        child.append(label_idx)
                        child.append(thick_idx)
                else:
                    # Mutate a random gene
                    if len(child) > 1:
                        gene_pos = np.random.randint(1, len(child))
                        if gene_pos % 2 == 1:  # label
                            child[gene_pos] = np.random.randint(0, 2)
                        else:  # thickness
                            child[gene_pos] = np.random.randint(0, n_cand)
            
            new_population.append(child)
        
        population = new_population
        fvals = np.array([evaluate_chrom(population[i]) for i in range(pop_size)])
        
        # Update best
        current_best_idx = np.argmin(fvals)
        if fvals[current_best_idx] < best_score:
            best_score = fvals[current_best_idx]
            best_chrom = population[current_best_idx].copy()
    
    # Decode best chromosome
    labels, thicknesses = decode_chrom(best_chrom)
    return labels, thicknesses, best_score

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
    
    # Phase 1: Discrete GA search to find good layer patterns and approximate thicknesses
    ga_time_budget = min(50.0, (deadline - start_time) * 0.5)
    ga_labels, ga_thicknesses, ga_score = _ga_discrete_search(problem, wavelengths_nm, ga_time_budget)
    
    if ga_score < best_score:
        best_score = ga_score
        best_labels = ga_labels
        best_thicknesses = ga_thicknesses
    
    # Phase 2: Continuous Nelder-Mead polish starting from GA result and other good initial guesses
    elapsed = time.time() - start_time
    remaining = deadline - elapsed - 5.0
    
    if remaining > 10.0 and best_labels is not None:
        # Polish the GA result
        problem_local = dict(problem)
        problem_local["labels"] = best_labels
        bounds = [(0, max_layer_nm)] * len(best_labels)
        
        def obj_func(x):
            return _objective(x, problem_local, wavelengths_nm)
        
        x0 = np.array(best_thicknesses)
        x0 = np.clip(x0, 0, max_layer_nm)
        
        x_opt, f_opt = _nelder_mead(obj_func, x0, bounds, max_iter=3000, tol=1e-12, time_deadline=deadline - 5.0)
        
        x_opt = np.clip(x_opt, 0, max_layer_nm)
        if np.sum(x_opt) > max_total_nm:
            x_opt = x_opt * (max_total_nm / np.sum(x_opt))
        
        f_opt = obj_func(x_opt)
        
        if f_opt < best_score:
            best_score = f_opt
            best_labels = best_labels.copy()
            best_thicknesses = x_opt.tolist()
    
    # Also try a few other initial structures with NM polish
    if time.time() < deadline - 20.0:
        # Try alternating L/H patterns with quarter-wave at center wavelength
        for n_layers in [5, 7, 9, 11]:
            if n_layers > max_layers:
                continue
            
            labels = ['L' if i % 2 == 0 else 'H' for i in range(n_layers)]
            indices = [materials[label] for label in labels]
            init = np.array([center_wl / (4.0 * n) for n in indices])
            
            problem_local = dict(problem)
            problem_local["labels"] = labels
            bounds = [(0, max_layer_nm)] * n_layers
            
            def obj_func(x):
                return _objective(x, problem_local, wavelengths_nm)
            
            x_opt, f_opt = _nelder_mead(obj_func, init, bounds, max_iter=2000, tol=1e-12, time_deadline=deadline - 10.0)
            
            x_opt = np.clip(x_opt, 0, max_layer_nm)
            if np.sum(x_opt) > max_total_nm:
                x_opt = x_opt * (max_total_nm / np.sum(x_opt))
            
            f_opt = obj_func(x_opt)
            
            if f_opt < best_score:
                best_score = f_opt
                best_labels = labels.copy()
                best_thicknesses = x_opt.tolist()
            
            if time.time() > deadline - 15.0:
                break
    
    # Fallback
    if best_labels is None:
        return ["L"], [center_wl / (4.0 * materials["L"])]
    
    # Ensure constraints
    thicknesses = np.clip(best_thicknesses, 0, max_layer_nm)
    if np.sum(thicknesses) > max_total_nm:
        thicknesses = thicknesses * (max_total_nm / np.sum(thicknesses))
    
    return best_labels, thicknesses.tolist()
