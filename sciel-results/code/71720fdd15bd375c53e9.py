import numpy as np
import time
import random

# EVOLVE-BLOCK-START

def _compute_reflectance(labels, thicknesses, materials, ambient_index, substrate_index, band_nm):
    """Compute mean reflectance in percent over the band using characteristic matrix method."""
    if len(labels) == 0:
        r0 = (ambient_index - substrate_index) / (ambient_index + substrate_index)
        return r0 * r0 * 100.0
    
    n_L = materials['L']
    n_H = materials['H']
    wavelengths = np.linspace(band_nm[0], band_nm[1], 401)
    n_layers = len(labels)
    
    # Build index array
    n_indices = np.empty(n_layers + 2, dtype=float)
    n_indices[0] = ambient_index
    n_indices[-1] = substrate_index
    for i, lab in enumerate(labels):
        n_indices[i + 1] = n_L if lab == 'L' else n_H
    
    d = np.array(thicknesses, dtype=float)
    
    # Vectorized computation over all wavelengths
    delta = 2.0 * np.pi * np.outer(n_indices[1:-1] * d, 1.0 / wavelengths)
    cos_d = np.cos(delta)
    sin_d = np.sin(delta)
    
    m11 = np.ones(401, dtype=complex)
    m12 = np.zeros(401, dtype=complex)
    m21 = np.zeros(401, dtype=complex)
    m22 = np.ones(401, dtype=complex)
    
    for j in range(n_layers):
        n_j = n_indices[j + 1]
        lm11 = cos_d[j]
        lm12 = 1j * sin_d[j] / n_j
        lm21 = 1j * n_j * sin_d[j]
        lm22 = cos_d[j]
        
        new_m11 = m11 * lm11 + m12 * lm21
        new_m12 = m11 * lm12 + m12 * lm22
        new_m21 = m21 * lm11 + m22 * lm21
        new_m22 = m21 * lm12 + m22 * lm22
        
        m11 = new_m11
        m12 = new_m12
        m21 = new_m21
        m22 = new_m22
    
    Y = (m21 + m22 * substrate_index) / (m11 + m12 * substrate_index)
    r = (ambient_index - Y) / (ambient_index + Y)
    reflectances = np.abs(r) ** 2
    
    return np.mean(reflectances) * 100.0

def _nelder_mead(objective, x0, step=10.0, max_iter=200, tol=1e-6, bounds=None):
    """
    Nelder-Mead optimizer with bounds.
    """
    n = len(x0)
    
    # Initialize simplex
    simplex = [np.array(x0, dtype=float)]
    for i in range(n):
        xi = np.array(x0, dtype=float)
        xi[i] += step
        if bounds:
            if xi[i] > bounds[i][1]:
                xi[i] = bounds[i][1]
            elif xi[i] < bounds[i][0]:
                xi[i] = bounds[i][0]
        simplex.append(xi)
    
    # Evaluate
    fvals = [objective(x) for x in simplex]
    
    for _ in range(max_iter):
        # Sort by function value
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]
        
        if abs(fvals[-1] - fvals[0]) < tol:
            break
        
        # Centroid of all but worst
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        alpha = 1.0
        xr = centroid + alpha * (centroid - simplex[-1])
        if bounds:
            for i in range(n):
                xr[i] = np.clip(xr[i], bounds[i][0], bounds[i][1])
        fr = objective(xr)
        
        if fr < fvals[0]:
            # Expansion
            gamma = 2.0
            xe = centroid + gamma * (centroid - simplex[-1])
            if bounds:
                for i in range(n):
                    xe[i] = np.clip(xe[i], bounds[i][0], bounds[i][1])
            fe = objective(xe)
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
            rho = 0.5
            xc = centroid + rho * (simplex[-1] - centroid)
            if bounds:
                for i in range(n):
                    xc[i] = np.clip(xc[i], bounds[i][0], bounds[i][1])
            fc = objective(xc)
            if fc < fvals[-1]:
                simplex[-1] = xc
                fvals[-1] = fc
            else:
                # Shrink
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    if bounds:
                        for j in range(n):
                            simplex[i][j] = np.clip(simplex[i][j], bounds[j][0], bounds[j][1])
                    fvals[i] = objective(simplex[i])
    
    # Return best
    order = np.argsort(fvals)
    return simplex[order[0]], fvals[order[0]]

def _perturb_and_polish(labels, thicknesses, materials, ambient_index, substrate_index, band_nm, 
                         max_layer_nm, max_total_nm, max_layers, time_budget):
    """
    Structural local search: perturb thicknesses, swap materials, add/remove layers.
    """
    start_time = time.time()
    best_labels = list(labels)
    best_thicknesses = list(thicknesses)
    best_score = _compute_reflectance(best_labels, best_thicknesses, materials, ambient_index, substrate_index, band_nm)
    
    def is_valid(labels, thicknesses):
        if len(labels) > max_layers:
            return False
        if len(labels) != len(thicknesses):
            return False
        if any(t < 0 or t > max_layer_nm for t in thicknesses):
            return False
        if sum(thicknesses) > max_total_nm:
            return False
        return True
    
    def try_improve(labels, thicknesses, temperature=1.0):
        nonlocal best_labels, best_thicknesses, best_score
        if not is_valid(labels, thicknesses):
            return
        score = _compute_reflectance(labels, thicknesses, materials, ambient_index, substrate_index, band_nm)
        if score < best_score:
            best_labels = list(labels)
            best_thicknesses = list(thicknesses)
            best_score = score
    
    # Phase 1: Perturb thicknesses with local search
    for _ in range(50):
        if time.time() - start_time > time_budget:
            break
        
        # Random perturb
        new_thicknesses = list(best_thicknesses)
        idx = random.randint(0, len(new_thicknesses) - 1)
        delta = random.uniform(-50, 50) * temperature
        new_thicknesses[idx] = max(0.1, min(max_layer_nm, new_thicknesses[idx] + delta))
        
        try_improve(best_labels, new_thicknesses, temperature)
        temperature *= 0.95
    
    # Phase 2: Material swaps
    for _ in range(20):
        if time.time() - start_time > time_budget:
            break
        
        new_labels = list(best_labels)
        idx = random.randint(0, len(new_labels) - 1)
        new_labels[idx] = 'L' if new_labels[idx] == 'H' else 'H'
        
        try_improve(new_labels, best_thicknesses)
    
    # Phase 3: Add/remove layers
    for _ in range(10):
        if time.time() - start_time > time_budget:
            break
        
        action = random.choice(['add', 'remove'])
        
        if action == 'add' and len(best_labels) < max_layers:
            # Add layer at random position
            pos = random.randint(0, len(best_labels))
            new_labels = list(best_labels)
            new_thicknesses = list(best_thicknesses)
            
            new_labels.insert(pos, random.choice(['L', 'H']))
            new_thicknesses.insert(pos, random.uniform(20, 100))
            
            try_improve(new_labels, new_thicknesses)
        elif action == 'remove' and len(best_labels) > 1:
            # Remove random layer
            pos = random.randint(0, len(best_labels) - 1)
            new_labels = list(best_labels)
            new_thicknesses = list(best_thicknesses)
            
            new_labels.pop(pos)
            new_thicknesses.pop(pos)
            
            try_improve(new_labels, new_thicknesses)
    
    # Final polish with Nelder-Mead on best found
    if time.time() - start_time < time_budget * 0.3:
        bounds = [(0.1, max_layer_nm) for _ in range(len(best_labels))]
        
        def objective(thicknesses):
            if not is_valid(best_labels, thicknesses):
                return 1000.0
            return _compute_reflectance(best_labels, thicknesses, materials, ambient_index, substrate_index, band_nm)
        
        x_nm, f_nm = _nelder_mead(objective, np.array(best_thicknesses), step=15.0, max_iter=100, tol=1e-6, bounds=bounds)
        
        if f_nm < best_score and is_valid(best_labels, x_nm.tolist()):
            best_labels = list(best_labels)
            best_thicknesses = x_nm.tolist()
            best_score = f_nm
    
    return best_labels, best_thicknesses, best_score

def _differential_evolution(objective, bounds, pop_size=30, max_iter=40, F=0.8, CR=0.9, time_limit=None, start_time=None):
    """
    Differential Evolution optimizer.
    """
    n = len(bounds)
    lb = np.array([b[0] for b in bounds])
    ub = np.array([b[1] for b in bounds])
    
    # Initialize population
    pop = np.random.uniform(lb, ub, size=(pop_size, n))
    fvals = np.array([objective(p) for p in pop])
    
    best_idx = np.argmin(fvals)
    best_x = pop[best_idx].copy()
    best_f = fvals[best_idx]
    
    for i in range(max_iter):
        if time_limit is not None and time.time() - start_time > time_limit:
            break
            
        for j in range(pop_size):
            # Select three distinct random individuals
            a, b, c = random.sample(range(pop_size), 3)
            
            # Mutation
            v = pop[a] + F * (pop[b] - pop[c])
            v = np.clip(v, lb, ub)
            
            # Crossover
            r = random.random()
            jrand = random.randint(0, n - 1)
            trial = np.copy(pop[j])
            for k in range(n):
                if random.random() < CR or k == jrand:
                    trial[k] = v[k]
            
            # Selection
            ft = objective(trial)
            if ft <= fvals[j]:
                pop[j] = trial
                fvals[j] = ft
                if ft < best_f:
                    best_f = ft
                    best_x = trial.copy()
            
    return best_x, best_f

def design(problem):
    """Design an antireflection coating stack using structural local search + DE + Nelder-Mead."""
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    materials = problem["materials"]
    ambient_index = problem["ambient_index"]
    substrate_index = problem["substrate_index"]
    band_nm = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    n_L = materials['L']
    n_H = materials['H']
    
    best_score = float('inf')
    best_labels = ['L']
    best_thicknesses = [600.0 / (4.0 * n_L)]
    
    # Fallback
    fallback_score = _compute_reflectance(best_labels, best_thicknesses, materials, ambient_index, substrate_index, band_nm)
    if fallback_score < best_score:
        best_score = fallback_score
    
    # Strategy: Try different layer counts with alternating patterns,
    # then use structural local search to explore the landscape
    
    layer_counts = [2, 4, 6, 8, 10, 12, 14]
    layer_counts = [lc for lc in layer_counts if lc <= max_layers]
    
    # For each layer count, try both LH and HL patterns
    for n_layers in layer_counts:
        if time.time() > deadline:
            break
        
        labels_LH = ['L' if i % 2 == 0 else 'H' for i in range(n_layers)]
        labels_HL = ['H' if i % 2 == 0 else 'L' for i in range(n_layers)]
        
        for labels in [labels_LH, labels_HL]:
            if time.time() > deadline:
                break
            
            bounds = [(0.1, max_layer_nm) for _ in range(n_layers)]
            
            def objective(thicknesses):
                if any(t < 0.1 or t > max_layer_nm for t in thicknesses):
                    return 1000.0
                if sum(thicknesses) > max_total_nm:
                    return 1000.0
                return _compute_reflectance(labels, thicknesses, materials, ambient_index, substrate_index, band_nm)
            
            # 1. Differential Evolution with larger population
            if n_layers <= 4:
                pop_size = 40
                max_iter_de = 50
            elif n_layers <= 8:
                pop_size = 30
                max_iter_de = 40
            else:
                pop_size = 25
                max_iter_de = 30
                
            try:
                x_de, f_de = _differential_evolution(
                    objective, 
                    bounds, 
                    pop_size=pop_size, 
                    max_iter=max_iter_de,
                    time_limit=deadline - time.time(),
                    start_time=start_time
                )
                
                # 2. Structural local search
                time_remaining = deadline - time.time()
                if time_remaining > 5.0:
                    labels_ls, thicknesses_ls, score_ls = _perturb_and_polish(
                        labels, x_de.tolist(), materials, ambient_index, substrate_index, band_nm,
                        max_layer_nm, max_total_nm, max_layers, time_remaining * 0.4
                    )
                    
                    if score_ls < best_score:
                        best_score = score_ls
                        best_labels = labels_ls
                        best_thicknesses = thicknesses_ls
                
                # 3. Final Nelder-Mead polish
                if f_de < best_score:
                    x_nm, f_nm = _nelder_mead(
                        objective, 
                        x_de, 
                        step=15.0, 
                        max_iter=150, 
                        tol=1e-6, 
                        bounds=bounds
                    )
                    
                    if f_nm < best_score:
                        if sum(x_nm) <= max_total_nm + 1e-6 and all(0.1 <= t <= max_layer_nm for t in x_nm):
                            best_score = f_nm
                            best_labels = labels[:]
                            best_thicknesses = x_nm.tolist()
            except Exception:
                pass
    
    # Final validation
    if sum(best_thicknesses) > max_total_nm:
        best_labels = ['L']
        best_thicknesses = [600.0 / (4.0 * n_L)]
        best_score = _compute_reflectance(best_labels, best_thicknesses, materials, ambient_index, substrate_index, band_nm)
    
    if any(t < 0 or t > max_layer_nm for t in best_thicknesses):
        best_labels = ['L']
        best_thicknesses = [600.0 / (4.0 * n_L)]
        best_score = _compute_reflectance(best_labels, best_thicknesses, materials, ambient_index, substrate_index, band_nm)
        
    return best_labels, best_thicknesses

# EVOLVE-BLOCK-END
