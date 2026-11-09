"""Broadband antireflection coating design: Genetic Algorithm with Local Polish."""
import time
import random
import numpy as np

def _mean_reflectance(materials, thicknesses, problem):
    """Compute mean reflectance (percent) over the band using characteristic matrices."""
    if not materials:
        return 100.0 * ((problem["substrate_index"] - problem["ambient_index"]) /
                        (problem["substrate_index"] + problem["ambient_index"])) ** 2
    
    band = problem["band_nm"]
    wavs = np.linspace(band[0], band[1], 401)
    n_mat = {k: v for k, v in problem["materials"].items()}
    
    ns = np.array([n_mat[m] for m in materials], dtype=float)
    ds = np.asarray(thicknesses, dtype=float)
    
    # delta_j = 2*pi*n_j*d_j/lambda  -> (N, K)
    delta = 2.0 * np.pi * (ns[:, None] * ds[:, None]) / wavs[None, :]
    c = np.cos(delta)
    s = np.sin(delta)
    
    # Initialize characteristic matrix product as identity for all wavelengths
    A = np.ones_like(wavs)
    B = np.zeros_like(wavs)
    C = np.zeros_like(wavs)
    D = np.ones_like(wavs)
    
    i = 1.0j
    for j in range(len(materials)):
        n_j = ns[j]
        cj = c[j]
        sj = s[j]
        A_new = A * cj + 1j * B * n_j * sj
        B_new = 1j * A * sj / n_j + B * cj
        C_new = C * cj + 1j * D * n_j * sj
        D_new = 1j * C * sj / n_j + D * cj
        A, B, C, D = A_new, B_new, C_new, D_new
    
    n_s = problem["substrate_index"]
    n_0 = problem["ambient_index"]
    num = n_0 * (A + B * n_s) - (C + D * n_s)
    den = n_0 * (A + B * n_s) + (C + D * n_s)
    r = num / den
    R = np.abs(r) ** 2
    return 100.0 * np.mean(R)

def _local_polish(materials, thicknesses, problem, deadline, max_iters=50):
    """Quick Nelder-Mead polish on thicknesses for a fixed pattern."""
    if not materials:
        return materials, thicknesses, _mean_reflectance(materials, thicknesses, problem)
    
    max_layer = problem["max_layer_nm"]
    max_total = problem["max_total_nm"]
    
    def constrain(ths):
        ths = np.array(ths, dtype=float)
        ths = np.clip(ths, 0.0, max_layer)
        total = np.sum(ths)
        if total > max_total:
            ths = ths * (max_total / total)
        return ths

    ths = np.array(thicknesses, dtype=float)
    ths = constrain(ths)
    current_score = _mean_reflectance(materials, ths, problem)
    
    n = len(ths)
    if n == 0:
        return materials, ths.tolist(), current_score
        
    # Initialize simplex
    simplex = [ths.copy()]
    step = 5.0
    for i in range(n):
        pt = ths.copy()
        pt[i] = min(max_layer, pt[i] + step)
        simplex.append(constrain(pt))
        
    scores = [_mean_reflectance(materials, s, problem) for s in simplex]
    
    alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
    
    for _ in range(max_iters):
        if time.time() > deadline:
            break
            
        order = np.argsort(scores)
        simplex = [simplex[i] for i in order]
        scores = [scores[i] for i in order]
        
        if scores[0] < current_score:
            current_score = scores[0]
            
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        xr = constrain(centroid + alpha * (centroid - simplex[-1]))
        fr = _mean_reflectance(materials, xr, problem)
        
        if fr < scores[0]:
            # Expansion
            xe = constrain(centroid + gamma * (centroid - simplex[-1]))
            fe = _mean_reflectance(materials, xe, problem)
            if fe < fr:
                simplex[-1] = xe
                scores[-1] = fe
            else:
                simplex[-1] = xr
                scores[-1] = fr
        elif fr < scores[-2]:
            simplex[-1] = xr
            scores[-1] = fr
        else:
            # Contraction
            xc = constrain(centroid + rho * (simplex[-1] - centroid))
            fc = _mean_reflectance(materials, xc, problem)
            if fc <= fr:
                simplex[-1] = xc
                scores[-1] = fc
            else:
                # Shrink
                for i in range(1, n + 1):
                    simplex[i] = constrain(simplex[0] + sigma * (simplex[i] - simplex[0]))
                    scores[i] = _mean_reflectance(materials, simplex[i], problem)
                    
                if max(scores) - min(scores) < 1e-8:
                    break

    best_idx = np.argmin(scores)
    return materials, simplex[best_idx].tolist(), scores[best_idx]

def design(problem):
    start_time = time.time()
    # 150s limit, keep 15s buffer
    deadline = start_time + 135 
    
    band = problem["band_nm"]
    wl_center = 0.5 * (band[0] + band[1])
    max_layers = problem["max_layers"]
    max_total = problem["max_total_nm"]
    max_layer = problem["max_layer_nm"]
    materials_dict = problem["materials"]
    
    random.seed(42)
    np.random.seed(42)
    
    # Precompute quarter-wave thicknesses
    qw = {m: wl_center / (4.0 * n) for m, n in materials_dict.items()}
    
    def random_individual():
        # Random number of layers between 4 and max_layers
        n_layers = random.randint(4, max_layers)
        # Random pattern
        mats = [random.choice(['L', 'H']) for _ in range(n_layers)]
        # Random thicknesses around quarter-wave, constrained
        ths = []
        for m in mats:
            t = qw[m] * random.uniform(0.5, 1.5)
            ths.append(t)
        
        # Constrain total thickness
        total = sum(ths)
        if total > max_total:
            scale = max_total / total
            ths = [t * scale for t in ths]
        
        # Constrain individual layers
        ths = [max(0.0, min(max_layer, t)) for t in ths]
        
        # Re-constrain total just in case
        total2 = sum(ths)
        if total2 > max_total:
            scale = max_total / total2
            ths = [t * scale for t in ths]
            
        return mats, ths

    def evaluate(ind):
        mats, ths = ind
        return _mean_reflectance(mats, ths, problem)

    def crossover(parent1, parent2):
        mats1, ths1 = parent1
        mats2, ths2 = parent2
        
        # Handle different lengths by padding shorter with random layers
        # Or just swap segments. Let's do segment swap.
        n1 = len(mats1)
        n2 = len(mats2)
        
        if n1 == 0 or n2 == 0:
            return parent1, parent2
            
        # Pick a crossover point for each
        p1 = random.randint(1, max(1, n1 - 1))
        p2 = random.randint(1, max(1, n2 - 1))
        
        child1_mats = mats1[:p1] + mats2[p2:]
        child1_ths = ths1[:p1] + ths2[p2:]
        
        child2_mats = mats2[:p2] + mats1[p1:]
        child2_ths = ths2[:p2] + ths1[p1:]
        
        # Truncate or pad to max_layers? 
        # Better to enforce max_layers constraint
        if len(child1_mats) > max_layers:
             child1_mats = child1_mats[:max_layers]
             child1_ths = child1_ths[:max_layers]
        if len(child2_mats) > max_layers:
             child2_mats = child2_mats[:max_layers]
             child2_ths = child2_ths[:max_layers]
             
        # Ensure min layers (e.g., 1)
        while len(child1_mats) < 1:
             child1_mats.append(random.choice(['L', 'H']))
             child1_ths.append(qw[child1_mats[-1]])
        while len(child2_mats) < 1:
             child2_mats.append(random.choice(['L', 'H']))
             child2_ths.append(qw[child2_mats[-1]])
             
        return (child1_mats, child1_ths), (child2_mats, child2_ths)

    def mutate(ind):
        mats, ths = ind
        mats = list(mats)
        ths = list(ths)
        
        # Mutate pattern
        if random.random() < 0.1:
            idx = random.randint(0, len(mats) - 1)
            mats[idx] = 'H' if mats[idx] == 'L' else 'L'
            
        # Mutate thickness
        if random.random() < 0.3:
            idx = random.randint(0, len(ths) - 1)
            # Gaussian perturbation
            ths[idx] += random.gauss(0, 10)
            
        # Add or remove layer
        if random.random() < 0.05 and len(mats) < max_layers:
            idx = random.randint(0, len(mats))
            new_m = random.choice(['L', 'H'])
            mats.insert(idx, new_m)
            ths.insert(idx, qw[new_m])
        elif random.random() < 0.05 and len(mats) > 1:
            idx = random.randint(0, len(mats) - 1)
            mats.pop(idx)
            ths.pop(idx)
            
        # Constrain thicknesses
        total = sum(ths)
        if total > max_total:
            scale = max_total / total
            ths = [t * scale for t in ths]
        ths = [max(0.0, min(max_layer, t)) for t in ths]
        total2 = sum(ths)
        if total2 > max_total:
            scale = max_total / total2
            ths = [t * scale for t in ths]
            
        return mats, ths

    # Initialize population
    pop_size = 40
    population = []
    
    # Seed with some structured individuals
    # 1. Alternating L H L H...
    alt1 = (['L' if i%2==0 else 'H' for i in range(max_layers)], [qw['L' if i%2==0 else 'H'] for i in range(max_layers)])
    alt1_ths = [t * (max_total / sum(t)) for t in alt1[1]]
    population.append((alt1[0], alt1_ths))
    
    # 2. Alternating H L H L...
    alt2 = (['H' if i%2==0 else 'L' for i in range(max_layers)], [qw['H' if i%2==0 else 'L'] for i in range(max_layers)])
    alt2_ths = [t * (max_total / sum(t)) for t in alt2[1]]
    population.append((alt2[0], alt2_ths))
    
    # 3. Graded L...H
    grad = (['L']*10 + ['H']*10, [qw['L']]*10 + [qw['H']]*10)
    grad_ths = [t * (max_total / sum(t)) for t in grad[1]]
    population.append((grad[0], grad_ths))
    
    # Rest random
    while len(population) < pop_size:
        population.append(random_individual())
        
    scores = [evaluate(ind) for ind in population]
    
    best_ind = population[0]
    best_score = scores[0]
    
    for i in range(pop_size):
        if scores[i] < best_score:
            best_score = scores[i]
            best_ind = population[i]
            
    # GA Loop
    gen = 0
    while time.time() < deadline - 20: # Leave 20s for polish
        gen += 1
        new_population = []
        
        # Elitism: keep top 2
        order = np.argsort(scores)
        for i in range(2):
            new_population.append(population[order[i]])
            
        # Fill rest with children
        while len(new_population) < pop_size:
            # Tournament selection
            def select():
                idxs = random.sample(range(pop_size), 3)
                best_idx = min(idxs, key=lambda x: scores[x])
                return population[best_idx]
                
            p1 = select()
            p2 = select()
            
            c1, c2 = crossover(p1, p2)
            c1 = mutate(c1)
            c2 = mutate(c2)
            
            new_population.append(c1)
            if len(new_population) < pop_size:
                new_population.append(c2)
                
        population = new_population
        scores = [evaluate(ind) for ind in population]
        
        # Update best
        min_idx = np.argmin(scores)
        if scores[min_idx] < best_score:
            best_score = scores[min_idx]
            best_ind = population[min_idx]

    # Final Polish
    if best_ind:
        mats, ths = best_ind
        final_mats, final_ths, final_score = _local_polish(mats, ths, problem, deadline, max_iters=100)
        if final_score < best_score:
            best_score = final_score
            best_ind = (final_mats, final_ths)
            
    # Fallback
    if not best_ind:
        return ["L"], [qw["L"]]
        
    mats, ths = best_ind
    return list(mats), list(ths)
