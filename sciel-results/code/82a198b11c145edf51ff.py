"""Broadband antireflection coating design via joint discrete-continuous DE."""
import time
import numpy as np

def _compute_mean_R(labels, thicknesses, n_amb, n_sub, n_L, n_H, band_nm):
    """Compute mean reflectance in percent over 401 equally spaced wavelengths."""
    lam_min, lam_max = band_nm
    wavelengths = np.linspace(lam_min, lam_max, 401)
    n_wl = len(wavelengths)
    
    if len(labels) == 0:
        r = (n_amb - n_sub) / (n_amb + n_sub)
        return np.abs(r) ** 2 * 100.0
    
    n_mat = np.array([n_L if l == 'L' else n_H for l in labels], dtype=float)
    d = np.array(thicknesses, dtype=float)
    beta = 2.0 * np.pi * n_mat * d  # (n_layers,)
    
    total_r = 0.0
    chunk = 401  # process all at once for small stacks
    for start in range(0, n_wl, chunk):
        end = min(start + chunk, n_wl)
        lams = wavelengths[start:end]
        nl = len(lams)
        
        M00 = np.ones(nl, dtype=complex)
        M01 = np.zeros(nl, dtype=complex)
        M10 = np.zeros(nl, dtype=complex)
        M11 = np.ones(nl, dtype=complex)
        
        for i in range(len(labels)):
            delta = beta[i] / lams
            c = np.cos(delta)
            s = np.sin(delta)
            
            a = c
            b = 1j * s / n_mat[i]
            cc = 1j * n_mat[i] * s
            dd = c
            
            new_M00 = M00 * a + M01 * cc
            new_M01 = M00 * b + M01 * dd
            new_M10 = M10 * a + M11 * cc
            new_M11 = M10 * b + M11 * dd
            M00, M01, M10, M11 = new_M00, new_M01, new_M10, new_M11
        
        Y = M10 / M00
        r = (n_amb - Y) / (n_amb + Y)
        R = np.abs(r) ** 2 * 100.0
        total_r += np.sum(R)
    
    return total_r / n_wl


def design(problem):
    """Design a broadband antireflection coating stack using DE with joint discrete/continuous search."""
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin before 150s kill
    
    n_amb = problem["ambient_index"]
    n_sub = problem["substrate_index"]
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    band_nm = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    centre = 0.5 * (band_nm[0] + band_nm[1])
    
    def _obj(labels, x):
        """Objective with hard penalty for constraint violations."""
        x = np.clip(x, 0.0, max_layer_nm)
        total = np.sum(x)
        penalty = 0.0
        if total > max_total_nm:
            penalty = 50.0 * (total - max_total_nm) / max_total_nm
        val = _compute_mean_R(labels, list(x), n_amb, n_sub, n_L, n_H, band_nm)
        return val + penalty
    
    def _is_feasible(x):
        return all(0.0 <= t <= max_layer_nm for t in x) and sum(x) <= max_total_nm + 1e-9
    
    # --- Evaluate initial structured configurations ---
    best_labels = None
    best_thicknesses = None
    best_score = float('inf')
    
    initial_configs = [
        (["L"], [centre / (4.0 * n_L)]),
        (["L", "H"], [centre / (4.0 * n_L), centre / (4.0 * n_H)]),
        (["H", "L"], [centre / (4.0 * n_H), centre / (4.0 * n_L)]),
        (["L", "H", "L"], [centre / (4.0 * n_L), centre / (4.0 * n_H), centre / (4.0 * n_L)]),
        (["H", "L", "H"], [centre / (4.0 * n_H), centre / (4.0 * n_L), centre / (4.0 * n_H)]),
        (["L", "H", "L", "H"], [centre / (4.0 * n_L), centre / (4.0 * n_H), centre / (4.0 * n_L), centre / (4.0 * n_H)]),
        (["H", "L", "H", "L"], [centre / (4.0 * n_H), centre / (4.0 * n_L), centre / (4.0 * n_H), centre / (4.0 * n_L)]),
        (["L", "H", "L", "H", "L"], [centre / (4.0 * n_L), centre / (4.0 * n_H), centre / (4.0 * n_L), centre / (4.0 * n_H), centre / (4.0 * n_L)]),
        (["H", "L", "H", "L", "H"], [centre / (4.0 * n_H), centre / (4.0 * n_L), centre / (4.0 * n_H), centre / (4.0 * n_L), centre / (4.0 * n_H)]),
        (["L", "H", "L", "H", "L", "H"], [centre / (4.0 * n_L), centre / (4.0 * n_H), centre / (4.0 * n_L), centre / (4.0 * n_H), centre / (4.0 * n_L), centre / (4.0 * n_H)]),
        (["H", "L", "H", "L", "H", "L"], [centre / (4.0 * n_H), centre / (4.0 * n_L), centre / (4.0 * n_H), centre / (4.0 * n_L), centre / (4.0 * n_H), centre / (4.0 * n_L)]),
    ]
    
    for labels, thicknesses in initial_configs:
        if len(labels) > max_layers:
            continue
        if sum(thicknesses) > max_total_nm:
            continue
        if any(t < 0 or t > max_layer_nm for t in thicknesses):
            continue
        score = _compute_mean_R(labels, thicknesses, n_amb, n_sub, n_L, n_H, band_nm)
        if score < best_score:
            best_score = score
            best_labels = labels[:]
            best_thicknesses = thicknesses[:]
    
    # --- Differential Evolution with joint discrete/continuous encoding ---
    # Encoding: each individual is (n_layers, labels[0..n_layers-1], thicknesses[0..n_layers-1])
    # We encode labels as 0/1 integers and thicknesses as continuous in [0, max_layer_nm]
    
    import random
    random.seed(42)
    np.random.seed(42)
    
    def _decode(ind):
        """Decode individual to (labels, thicknesses)."""
        n_layers = int(round(ind[0]))
        n_layers = max(1, min(n_layers, max_layers))
        labels = ['L' if int(round(ind[1 + i])) == 0 else 'H' for i in range(n_layers)]
        thicknesses = [float(ind[2 + i]) for i in range(n_layers)]
        thicknesses = [max(0.0, min(t, max_layer_nm)) for t in thicknesses]
        return labels, thicknesses
    
    def _encode(labels, thicknesses):
        """Encode (labels, thicknesses) to individual vector."""
        n_layers = len(labels)
        ind = [float(n_layers)]
        for l in labels:
            ind.append(0.0 if l == 'L' else 1.0)
        for t in thicknesses:
            ind.append(float(t))
        return ind
    
    def _evaluate_individual(ind):
        """Evaluate an individual, returning (score, labels, thicknesses, feasible)."""
        labels, thicknesses = _decode(ind)
        if len(labels) == 0:
            return float('inf'), labels, thicknesses, False
        x = np.array(thicknesses)
        if not _is_feasible(x):
            # Try to fix
            x = np.clip(x, 0.0, max_layer_nm)
            total = np.sum(x)
            if total > max_total_nm:
                x = x * (max_total_nm / total)
                x = np.clip(x, 0.0, max_layer_nm)
            thicknesses = list(x)
        score = _compute_mean_R(labels, thicknesses, n_amb, n_sub, n_L, n_H, band_nm)
        feasible = _is_feasible(np.array(thicknesses))
        return score, labels, thicknesses, feasible
    
    # DE parameters
    n_max_layers = min(max_layers, 12)  # limit search space
    # Individual size: 1 (n_layers) + n_max_layers (labels) + n_max_layers (thicknesses)
    ind_size = 1 + 2 * n_max_layers
    
    # Initial population
    pop_size = 40
    population = []
    
    # Add the best initial config
    if best_labels is not None:
        population.append(_encode(best_labels, best_thicknesses))
    
    # Add alternating patterns
    for n_l in range(1, min(n_max_layers + 1, 8)):
        seq_L = ["L" if i % 2 == 0 else "H" for i in range(n_l)]
        thick_L = [centre / (4.0 * (n_L if l == 'L' else n_H)) for l in seq_L]
        if sum(thick_L) <= max_total_nm:
            population.append(_encode(seq_L, thick_L))
        seq_H = ["H" if i % 2 == 0 else "L" for i in range(n_l)]
        thick_H = [centre / (4.0 * (n_L if l == 'L' else n_H)) for l in seq_H]
        if sum(thick_H) <= max_total_nm:
            population.append(_encode(seq_H, thick_H))
    
    # Add random individuals
    while len(population) < pop_size:
        n_l = random.randint(2, n_max_layers)
        labels = [random.choice(['L', 'H']) for _ in range(n_l)]
        # Random thicknesses around quarter-wave
        thicknesses = []
        for l in labels:
            n_mat = n_L if l == 'L' else n_H
            t = centre / (4.0 * n_mat)
            t *= random.uniform(0.5, 1.5)
            t = max(0.0, min(t, max_layer_nm))
            thicknesses.append(t)
        # Scale to fit total
        total = sum(thicknesses)
        if total > max_total_nm:
            scale = max_total_nm / total
            thicknesses = [t * scale for t in thicknesses]
        population.append(_encode(labels, thicknesses))
    
    # Evaluate initial population
    pop_scores = [float('inf')] * len(population)
    pop_labels = [None] * len(population)
    pop_thick = [None] * len(population)
    
    for i, ind in enumerate(population):
        if time.time() > deadline - 10.0:
            break
        score, labels, thicknesses, feasible = _evaluate_individual(ind)
        pop_scores[i] = score
        pop_labels[i] = labels
        pop_thick[i] = thicknesses
        if feasible and score < best_score:
            best_score = score
            best_labels = labels[:]
            best_thicknesses = thicknesses[:]
    
    # DE iteration
    F = 0.8  # mutation factor
    CR = 0.9  # crossover rate
    max_gen = 300
    
    for gen in range(max_gen):
        if time.time() > deadline - 15.0:
            break
        
        # Sort population by score
        order = np.argsort(pop_scores)
        population = [population[i] for i in order]
        pop_scores = [pop_scores[i] for i in order]
        pop_labels = [pop_labels[i] for i in order]
        pop_thick = [pop_thick[i] for i in order]
        
        # Check convergence
        if pop_scores[-1] - pop_scores[0] < 1e-6:
            break
        
        new_population = population[:]
        new_scores = pop_scores[:]
        new_labels = pop_labels[:]
        new_thick = pop_thick[:]
        
        for i in range(len(population)):
            if time.time() > deadline - 10.0:
                break
            
            # Select three distinct random individuals
            a, b, c = random.sample(range(len(population)), 3)
            
            # Mutation: v = x_a + F * (x_b - x_c)
            v = [0.0] * ind_size
            for j in range(ind_size):
                v[j] = population[a][j] + F * (population[b][j] - population[c][j])
            
            # Crossover: binomial crossover with CR
            trial = [0.0] * ind_size
            for j in range(ind_size):
                if random.random() < CR or j == 0:  # always crossover n_layers dimension
                    trial[j] = v[j]
                else:
                    trial[j] = population[i][j]
            
            # Evaluate trial
            score, labels, thicknesses, feasible = _evaluate_individual(trial)
            
            if score <= new_scores[i]:
                new_population[i] = trial
                new_scores[i] = score
                new_labels[i] = labels
                new_thick[i] = thicknesses
                if feasible and score < best_score:
                    best_score = score
                    best_labels = labels[:]
                    best_thicknesses = thicknesses[:]
        
        population = new_population
        pop_scores = new_scores
        pop_labels = new_labels
        pop_thick = new_thick
        
        # Elitism: ensure best is kept
        if best_labels is not None:
            best_ind = _encode(best_labels, best_thicknesses)
            if pop_scores[0] > best_score:
                population[0] = best_ind
                pop_scores[0] = best_score
                pop_labels[0] = best_labels[:]
                pop_thick[0] = best_thicknesses[:]
    
    # --- Local refinement on the best solution using Nelder-Mead ---
    if time.time() < deadline - 12.0 and best_labels is not None:
        def _nm_optimize(labels, x0, max_iter=150):
            n = len(labels)
            if n == 0:
                return None, float('inf')
            
            x0 = np.array(x0, dtype=float)
            x0 = np.clip(x0, 0.0, max_layer_nm)
            if np.sum(x0) > max_total_nm:
                x0 = x0 * (max_total_nm / np.sum(x0))
                x0 = np.clip(x0, 0.0, max_layer_nm)
            
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5
            
            simplex = [x0.copy()]
            for i in range(n):
                perturbed = x0.copy()
                step = 0.1 * max_layer_nm
                perturbed[i] = np.clip(perturbed[i] + step, 0.0, max_layer_nm)
                simplex.append(perturbed)
            
            fvals = [_obj(labels, x) for x in simplex]
            
            for it in range(max_iter):
                if time.time() > deadline - 8.0:
                    break
                
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]
                
                if abs(fvals[-1] - fvals[0]) < 1e-8:
                    break
                
                centroid = np.mean(simplex[:-1], axis=0)
                
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, 0.0, max_layer_nm)
                fr = _obj(labels, xr)
                
                if fr < fvals[0]:
                    xe = centroid + gamma * (xr - centroid)
                    xe = np.clip(xe, 0.0, max_layer_nm)
                    fe = _obj(labels, xe)
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
                    if fr < fvals[-1]:
                        xc = centroid + rho * (xr - centroid)
                    else:
                        xc = centroid - rho * (simplex[-1] - centroid)
                    xc = np.clip(xc, 0.0, max_layer_nm)
                    fc = _obj(labels, xc)
                    
                    if fc < fvals[-1]:
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], 0.0, max_layer_nm)
                            fvals[i] = _obj(labels, simplex[i])
            
            best_idx = np.argmin(fvals)
            best_x = simplex[best_idx]
            if _is_feasible(best_x):
                actual_score = _compute_mean_R(labels, list(best_x), n_amb, n_sub, n_L, n_H, band_nm)
                return list(best_x), actual_score
            return None, float('inf')
        
        refined, refined_score = _nm_optimize(best_labels, best_thicknesses, max_iter=200)
        if refined is not None and refined_score < best_score:
            if _is_feasible(np.array(refined)):
                best_score = refined_score
                best_thicknesses = refined[:]
    
    # --- Final constraint enforcement ---
    if best_labels is None:
        centre_val = 0.5 * (band_nm[0] + band_nm[1])
        best_labels = ["L"]
        best_thicknesses = [centre_val / (4.0 * n_L)]
    
    best_thicknesses = [max(0.0, min(t, max_layer_nm)) for t in best_thicknesses]
    if sum(best_thicknesses) > max_total_nm:
        scale = max_total_nm / sum(best_thicknesses)
        best_thicknesses = [t * scale for t in best_thicknesses]
    
    return best_labels, best_thicknesses
