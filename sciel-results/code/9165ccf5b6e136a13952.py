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

def _fitness(materials_int, thicknesses, wavelengths_nm, ambient_idx, substrate_idx, 
             max_layer_nm, max_total_nm, materials_map):
    """
    Calculate fitness (mean reflectance in percent).
    materials_int: array of 0/1 (0=L, 1=H)
    thicknesses: array of floats
    """
    # Filter out zero thickness layers to speed up calculation slightly and handle "empty" slots
    active_mask = thicknesses > 1e-6
    if not np.any(active_mask):
        # Bare substrate
        r0 = (ambient_idx - substrate_idx) / (ambient_idx + substrate_idx)
        return r0 * r0 * 100.0
        
    active_materials = [materials_map[materials_int[i]] for i in range(len(materials_int)) if active_mask[i]]
    active_thicknesses = thicknesses[active_mask]
    
    # Check constraints
    if len(active_materials) == 0:
         r0 = (ambient_idx - substrate_idx) / (ambient_idx + substrate_idx)
         return r0 * r0 * 100.0

    if np.sum(active_thicknesses) > max_total_nm + 1e-6:
        # Scale down to meet total thickness constraint
        active_thicknesses = active_thicknesses * (max_total_nm / np.sum(active_thicknesses))
        
    if np.any(active_thicknesses > max_layer_nm + 1e-6):
        # Clip individual layers
        active_thicknesses = np.clip(active_thicknesses, 0, max_layer_nm)
        
    R = _compute_reflectance(wavelengths_nm, active_materials, active_thicknesses, 
                             ambient_idx, substrate_idx)
    return np.mean(R) * 100.0

def _nelder_mead(f, x0, bounds, max_iter=500, tol=1e-10, time_deadline=None):
    """Nelder-Mead simplex with bounds."""
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
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    ambient_idx = problem["ambient_index"]
    substrate_idx = problem["substrate_index"]
    materials_dict = problem["materials"]
    band_nm = problem["band_nm"]
    max_layers = problem.get("max_layers", 20)
    max_total_nm = problem.get("max_total_nm", 1000)
    max_layer_nm = problem.get("max_layer_nm", 400)
    
    wavelengths_nm = np.linspace(band_nm[0], band_nm[1], 401)
    center_wl = 0.5 * (band_nm[0] + band_nm[1])
    
    # Map 0->L, 1->H
    mat_L = materials_dict['L']
    mat_H = materials_dict['H']
    materials_map = [mat_L, mat_H]
    
    N = max_layers # Fixed max layers for GA representation
    pop_size = 60
    max_gen = 200 # Will be limited by time
    
    # Initialize population
    # Mix of random and physics-based (quarter-wave)
    population = []
    for i in range(pop_size):
        if i < 10:
            # Physics based: Quarter wave stack
            n_layers_actual = 5 # Start with 5 layers
            mats = [0] * n_layers_actual + [0] * (N - n_layers_actual)
            # Alternate L/H
            for j in range(n_layers_actual):
                mats[j] = j % 2
            thicks = np.zeros(N)
            for j in range(n_layers_actual):
                thicks[j] = center_wl / (4.0 * materials_map[mats[j]])
            # Add some noise
            thicks[:n_layers_actual] *= np.random.uniform(0.8, 1.2, n_layers_actual)
        else:
            # Random
            mats = np.random.randint(0, 2, N)
            thicks = np.random.uniform(0, max_layer_nm * 0.5, N)
            # Ensure total thickness constraint roughly
            if np.sum(thicks) > max_total_nm:
                thicks = thicks * (max_total_nm / np.sum(thicks))
        
        population.append((mats, thicks))
    
    best_fitness = np.inf
    best_individual = None
    
    # Pre-compute fitness for initial population
    fitnesses = np.array([_fitness(mats, thicks, wavelengths_nm, ambient_idx, substrate_idx, 
                                   max_layer_nm, max_total_nm, materials_map) 
                          for mats, thicks in population])
    
    best_idx = np.argmin(fitnesses)
    best_fitness = fitnesses[best_idx]
    best_individual = population[best_idx]
    
    # GA Loop
    for gen in range(max_gen):
        if time.time() > deadline - 30:
            break
        
        # Local Search on top individuals
        # Every 5 generations, polish top 5
        if gen % 5 == 0 and gen > 0:
            top_indices = np.argsort(fitnesses)[:5]
            for idx in top_indices:
                mats, thicks = population[idx]
                # Only polish if there are active layers
                active_mask = thicks > 1e-6
                if not np.any(active_mask):
                    continue
                    
                # Define objective for NM
                def nm_obj(x, mats=mats, active_mask=active_mask):
                    # x is the full thickness vector, but we only care about active ones?
                    # NM works on the full vector. We will just pass the full vector.
                    return _fitness(mats, x, wavelengths_nm, ambient_idx, substrate_idx, 
                                    max_layer_nm, max_total_nm, materials_map)
                
                bounds = [(0, max_layer_nm)] * N
                # NM is expensive, limit iterations
                try:
                    x_opt, f_opt = _nelder_mead(nm_obj, thicks, bounds, max_iter=300, time_deadline=time.time() + 2.0)
                    if f_opt < fitnesses[idx]:
                        population[idx] = (mats, x_opt)
                        fitnesses[idx] = f_opt
                        if f_opt < best_fitness:
                            best_fitness = f_opt
                            best_individual = population[idx]
                except Exception:
                    pass
        
        # Selection: Tournament
        new_population = []
        
        # Elitism: Keep best individual
        best_idx = np.argmin(fitnesses)
        new_population.append(population[best_idx])
        
        while len(new_population) < pop_size:
            # Tournament of 3
            idx1 = np.random.randint(0, pop_size)
            idx2 = np.random.randint(0, pop_size)
            idx3 = np.random.randint(0, pop_size)
            
            best_in_tournament = idx1
            if fitnesses[idx2] < fitnesses[best_in_tournament]:
                best_in_tournament = idx2
            if fitnesses[idx3] < fitnesses[best_in_tournament]:
                best_in_tournament = idx3
                
            parent1 = population[best_in_tournament]
            
            # Second parent
            idx4 = np.random.randint(0, pop_size)
            idx5 = np.random.randint(0, pop_size)
            idx6 = np.random.randint(0, pop_size)
            
            best_in_tournament2 = idx4
            if fitnesses[idx5] < fitnesses[best_in_tournament2]:
                best_in_tournament2 = idx5
            if fitnesses[idx6] < fitnesses[best_in_tournament2]:
                best_in_tournament2 = idx6
                
            parent2 = population[best_in_tournament2]
            
            # Crossover: Uniform
            cross_mask = np.random.rand(N) < 0.5
            child_mats = np.where(cross_mask, parent1[0], parent2[0])
            child_thicks = np.where(cross_mask, parent1[1], parent2[1])
            
            # Mutation
            # Material mutation
            mat_mut_mask = np.random.rand(N) < 0.1
            child_mats[mat_mut_mask] = 1 - child_mats[mat_mut_mask]
            
            # Thickness mutation
            th_mut_mask = np.random.rand(N) < 0.5
            noise = np.random.normal(0, 20, N) # 20nm noise
            child_thicks[th_mut_mask] += noise[th_mut_mask]
            child_thicks = np.clip(child_thicks, 0, max_layer_nm)
            
            # "Zero-out" mutation to allow layer removal
            zero_mask = np.random.rand(N) < 0.05
            child_thicks[zero_mask] = 0
            
            # "Add" mutation to allow layer addition (if currently 0)
            add_mask = (child_thicks == 0) & (np.random.rand(N) < 0.05)
            child_thicks[add_mask] = np.random.uniform(50, 200, np.sum(add_mask))
            
            # Check total thickness constraint
            total_th = np.sum(child_thicks)
            if total_th > max_total_nm:
                child_thicks = child_thicks * (max_total_nm / total_th)
            
            new_population.append((child_mats, child_thicks))
        
        population = new_population
        
        # Evaluate new population
        fitnesses = np.array([_fitness(mats, thicks, wavelengths_nm, ambient_idx, substrate_idx, 
                                   max_layer_nm, max_total_nm, materials_map) 
                          for mats, thicks in population])
        
        # Update best
        gen_best_idx = np.argmin(fitnesses)
        if fitnesses[gen_best_idx] < best_fitness:
            best_fitness = fitnesses[gen_best_idx]
            best_individual = population[gen_best_idx]
    
    # Final Polish
    if time.time() < deadline - 5:
        mats, thicks = best_individual
        active_mask = thicks > 1e-6
        if np.any(active_mask):
            def final_obj(x):
                return _fitness(mats, x, wavelengths_nm, ambient_idx, substrate_idx, 
                                max_layer_nm, max_total_nm, materials_map)
            
            bounds = [(0, max_layer_nm)] * N
            try:
                x_opt, f_opt = _nelder_mead(final_obj, thicks, bounds, max_iter=1000, time_deadline=deadline - 2.0)
                if f_opt < best_fitness:
                    best_fitness = f_opt
                    best_individual = (mats, x_opt)
            except Exception:
                pass
    
    # Extract final result
    mats, thicks = best_individual
    active_mask = thicks > 1e-6
    final_mats = [materials_dict['L'] if mats[i] == 0 else materials_dict['H'] for i in range(N) if active_mask[i]]
    final_thicks = thicks[active_mask]
    
    # Ensure constraints
    if len(final_mats) == 0:
        return ["L"], [center_wl / (4.0 * mat_L)]
        
    # Re-check total thickness
    if np.sum(final_thicks) > max_total_nm:
        final_thicks = final_thicks * (max_total_nm / np.sum(final_thicks))
        
    # Clip
    final_thicks = np.clip(final_thicks, 0, max_layer_nm)
    
    return final_mats, final_thicks.tolist()
