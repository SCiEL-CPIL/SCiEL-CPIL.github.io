"""Broadband antireflection coating design via Genetic Algorithm + Nelder-Mead polish."""
import time
import numpy as np
import random

# EVOLVE-BLOCK-START

def _mean_R(labels, thicknesses, n_amb, n_sub, n_L, n_H, band_nm, n_wl=401):
    """Mean reflectance (%) over n_wl equally spaced wavelengths in band_nm."""
    lam_min, lam_max = band_nm
    lams = np.linspace(lam_min, lam_max, n_wl)
    n_mat = np.array([n_L if l == 'L' else n_H for l in labels], dtype=float)
    d = np.array(thicknesses, dtype=float)
    if len(labels) == 0:
        r = (n_amb - n_sub) / (n_amb + n_sub)
        return float(np.abs(r) ** 2 * 100.0)
    beta = 2.0 * np.pi * n_mat * d  # beta_i = 2*pi*n_i*d_i ; delta_i = beta_i/lambda
    total = 0.0
    chunk = 64
    for s in range(0, n_wl, chunk):
        e = min(s + chunk, n_wl)
        L = lams[s:e]
        m = L.shape[0]
        M00 = np.ones(m, dtype=complex); M01 = np.zeros(m, dtype=complex)
        M10 = np.zeros(m, dtype=complex); M11 = np.ones(m, dtype=complex)
        for i in range(len(labels)):
            delta = beta[i] / L
            c = np.cos(delta); sn = np.sin(delta)
            b = 1j * sn / n_mat[i]
            cc = 1j * n_mat[i] * sn
            n00 = M00 * c + M01 * cc
            n01 = M00 * b + M01 * c
            n10 = M10 * c + M11 * cc
            n11 = M10 * b + M11 * c
            M00, M01, M10, M11 = n00, n01, n10, n11
        Y = M10 / M00
        r = (n_amb - Y) / (n_amb + Y)
        total += np.sum(np.abs(r) ** 2 * 100.0)
    return total / n_wl

def _nm_optimize(labels, x0, n_amb, n_sub, n_L, n_H, band_nm, max_layer_nm,
                 max_total_nm, deadline, iters=100, n_wl=401):
    """Constrained Nelder-Mead optimization."""
    n = len(labels)
    if n == 0:
        return None, float('inf')

    def proj(x):
        x = np.clip(x, 0.0, max_layer_nm)
        tot = x.sum()
        if tot > max_total_nm:
            x = x * (max_total_nm / tot)
        return x

    def obj(x):
        return _mean_R(labels, x, n_amb, n_sub, n_L, n_H, band_nm, n_wl)

    x0 = proj(x0)
    simplex = [x0.copy()]
    for i in range(n):
        p = x0.copy()
        p[i] = np.clip(p[i] + 0.1 * max_layer_nm, 0.0, max_layer_nm)
        simplex.append(proj(p))
    fvals = [obj(v) for v in simplex]

    alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
    for it in range(iters):
        if time.time() > deadline - 2.0:
            break
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]
        if abs(fvals[-1] - fvals[0]) < 1e-8:
            break
        centroid = np.mean(simplex[:-1], axis=0)
        xr = proj(centroid + alpha * (centroid - simplex[-1]))
        fr = obj(xr)
        if fr < fvals[0]:
            xe = proj(centroid + gamma * (xr - centroid))
            fe = obj(xe)
            simplex[-1], fvals[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < fvals[-2]:
            simplex[-1], fvals[-1] = xr, fr
        else:
            xc = proj(centroid + rho * (xr - centroid)) if fr < fvals[-1] \
                else proj(centroid - rho * (simplex[-1] - centroid))
            fc = obj(xc)
            if fc < fvals[-1]:
                simplex[-1], fvals[-1] = xc, fc
            else:
                for i in range(1, len(simplex)):
                    simplex[i] = proj(simplex[0] + sigma * (simplex[i] - simplex[0]))
                    fvals[i] = obj(simplex[i])
    bi = int(np.argmin(fvals))
    return simplex[bi].tolist(), fvals[bi]

def design(problem):
    start = time.time()
    deadline = start + 140.0  # 10s safety margin

    n_amb = problem["ambient_index"]
    n_sub = problem["substrate_index"]
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    band_nm = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]

    centre = 0.5 * (band_nm[0] + band_nm[1])
    
    # Discrete thickness grid for GA
    # We use a grid of 20 points from 0 to max_layer_nm
    # This allows GA to explore the space without continuous gradients
    n_grid = 20
    grid = np.linspace(0, max_layer_nm, n_grid)
    
    # GA Parameters
    pop_size = 60
    n_gen = 40
    crossover_rate = 0.8
    mutation_rate = 0.15
    elite_size = 5
    
    # Initialize Population
    # Each individual is a tuple (labels, thickness_indices)
    # labels: list of 'L'/'H'
    # thickness_indices: list of indices into 'grid'
    
    def decode(ind):
        labels, t_inds = ind
        # Filter out zero-thickness layers to keep the stack valid and simple
        # If a layer has 0 thickness, it effectively doesn't exist
        valid_labels = []
        valid_thick = []
        for l, ti in zip(labels, t_inds):
            t = grid[ti]
            if t > 0.0:
                valid_labels.append(l)
                valid_thick.append(t)
        
        # Enforce max layers constraint by truncating if necessary
        # (GA should mostly respect this, but safety check)
        if len(valid_labels) > max_layers:
            valid_labels = valid_labels[:max_layers]
            valid_thick = valid_thick[:max_layers]
            
        # Enforce total thickness constraint
        total_t = sum(valid_thick)
        if total_t > max_total_nm:
            scale = max_total_nm / total_t
            valid_thick = [t * scale for t in valid_thick]
        
        return valid_labels, valid_thick

    def evaluate(ind):
        labels, thick = decode(ind)
        if not labels:
            return 100.0 # High penalty for empty stack
        return _mean_R(labels, thick, n_amb, n_sub, n_L, n_H, band_nm, n_wl=101)

    rng = random.Random(42)
    
    # Generate initial population
    population = []
    for _ in range(pop_size):
        n_layers = rng.randint(1, min(10, max_layers))
        labels = [rng.choice(['L', 'H']) for _ in range(n_layers)]
        t_inds = [rng.randint(0, n_grid - 1) for _ in range(n_layers)]
        population.append((labels, t_inds))
    
    # Ensure some structured starts (alternating)
    for n in [3, 5, 7]:
        if n <= max_layers:
            labels = ['L' if i % 2 == 0 else 'H' for i in range(n)]
            # Quarter wave approx at center
            qwl = centre / (4.0 * n_L)
            qwh = centre / (4.0 * n_H)
            t_inds = []
            for l in labels:
                target = qwl if l == 'L' else qwh
                # Find closest grid index
                idx = np.argmin(np.abs(grid - target))
                t_inds.append(idx)
            population.append((labels, t_inds))
            
    # Truncate population if we added too many
    population = population[:pop_size]
    
    # Evaluate initial population
    scores = [evaluate(ind) for ind in population]
    
    best_score = min(scores)
    best_ind = population[np.argmin(scores)]
    
    for gen in range(n_gen):
        if time.time() > deadline - 10.0:
            break
            
        # Sort population by score
        sorted_inds = [ind for _, ind in sorted(zip(scores, population))]
        sorted_scores = [s for s, _ in sorted(zip(scores, population))]
        
        # Update best
        if sorted_scores[0] < best_score:
            best_score = sorted_scores[0]
            best_ind = sorted_inds[0]
            
        # Elitism
        new_population = [sorted_inds[i] for i in range(min(elite_size, pop_size))]
        
        # Selection and Crossover
        while len(new_population) < pop_size:
            if time.time() > deadline - 10.0:
                break
                
            # Tournament Selection
            def select():
                i1 = rng.randint(0, len(sorted_inds) - 1)
                i2 = rng.randint(0, len(sorted_inds) - 1)
                if sorted_scores[i1] < sorted_scores[i2]:
                    return sorted_inds[i1]
                else:
                    return sorted_inds[i2]
            
            parent1 = select()
            parent2 = select()
            
            if rng.random() < crossover_rate:
                # Crossover: Mix labels and thicknesses
                # Find min length for crossover
                len1 = len(parent1[0])
                len2 = len(parent2[0])
                min_len = min(len1, len2)
                
                if min_len == 0:
                    child = parent1
                else:
                    point = rng.randint(1, min_len)
                    
                    child_labels = parent1[0][:point] + parent2[0][point:]
                    child_t_inds = parent1[1][:point] + parent2[1][point:]
                    
                    # Mutation
                    for i in range(len(child_labels)):
                        if rng.random() < mutation_rate:
                            # Mutate label
                            if rng.random() < 0.5:
                                child_labels[i] = 'L' if child_labels[i] == 'H' else 'H'
                        if rng.random() < mutation_rate:
                            # Mutate thickness index
                            child_t_inds[i] = rng.randint(0, n_grid - 1)
                    
                    # Occasionally add/remove a layer
                    if rng.random() < 0.1 and len(child_labels) < max_layers:
                        pos = rng.randint(0, len(child_labels))
                        child_labels.insert(pos, rng.choice(['L', 'H']))
                        child_t_inds.insert(pos, rng.randint(0, n_grid - 1))
                    elif rng.random() < 0.1 and len(child_labels) > 1:
                        pos = rng.randint(0, len(child_labels) - 1)
                        child_labels.pop(pos)
                        child_t_inds.pop(pos)
                        
                    child = (child_labels, child_t_inds)
            else:
                child = parent1
                
            new_population.append(child)
            
        # Evaluate new population
        population = new_population
        scores = [evaluate(ind) for ind in population]
        
    # --- Phase 2: Polish the best discrete candidate with Nelder-Mead ---
    if best_ind is not None and time.time() < deadline - 5.0:
        best_labels, best_thick = decode(best_ind)
        if best_labels:
            # Run NM optimization starting from the GA solution
            thick, score = _nm_optimize(best_labels, best_thick, n_amb, n_sub, n_L, n_H, 
                                        band_nm, max_layer_nm, max_total_nm, deadline, 
                                        iters=200, n_wl=401)
            if thick is not None:
                best_labels, best_thick = best_labels, thick
                best_score = score
            else:
                # If NM failed, use the discrete one
                best_score = _mean_R(best_labels, best_thick, n_amb, n_sub, n_L, n_H, band_nm, n_wl=401)
        else:
            # Fallback if GA produced empty stack (shouldn't happen with penalty)
            best_labels = ["L"]
            best_thick = [centre / (4.0 * n_L)]
            best_score = _mean_R(best_labels, best_thick, n_amb, n_sub, n_L, n_H, band_nm, n_wl=401)
    else:
        # Fallback
        best_labels = ["L"]
        best_thick = [centre / (4.0 * n_L)]
        best_score = _mean_R(best_labels, best_thick, n_amb, n_sub, n_L, n_H, band_nm, n_wl=401)

    # --- Final Constraint Enforcement ---
    best_thick = [max(0.0, min(t, max_layer_nm)) for t in best_thick]
    tot = sum(best_thick)
    if tot > max_total_nm:
        sc = max_total_nm / tot
        best_thick = [t * sc for t in best_thick]
        
    return best_labels, best_thick

# EVOLVE-BLOCK-END
