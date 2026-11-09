"""
Antireflection coating design: broadband (400-800 nm) low reflectance on glass.

Strategy:
  Evolutionary Search over the full design space (material sequence + thicknesses).
  
  Unlike the previous approach which fixed a set of candidate material sequences and 
  optimized thicknesses locally, this program uses a Genetic Algorithm (GA) where 
  both the material sequence and the layer thicknesses are part of the genome.
  
  1. Genome: A list of (material, thickness) tuples.
  2. Fitness: Negative mean reflectance (to be maximized).
  3. Operators:
     - Crossover: Single-point crossover on the sequence of layers.
     - Mutation: 
       a) Material swap (change L to H or vice versa).
       b) Thickness perturbation (Gaussian noise).
       c) Layer insertion/deletion.
  4. Selection: Tournament selection.
  5. Elitism: Carry over top performers.
  
  This allows the algorithm to explore structural changes (e.g., discovering that 
  a graded index profile is better than a symmetric one) simultaneously with 
  thickness optimization.
"""

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
    ns = [n_mat[m] for m in materials]
    ds = np.asarray(thicknesses, dtype=float)
    
    if len(ns) == 0:
        return 100.0 * ((problem["substrate_index"] - problem["ambient_index"]) /
                        (problem["substrate_index"] + problem["ambient_index"])) ** 2

    # Build per-layer optical path lengths
    # delta_j = 2*pi*n_j*d_j/lambda
    n_ds = np.array([n * d for n, d in zip(ns, ds)])  # (N,)
    delta = 2.0 * np.pi * n_ds[:, None] / wavs[None, :]  # (N, K)
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
        # Update A,B,C,D
        A_new = A * cj + 1j * B * n_j * sj
        B_new = 1j * A * sj / n_j + B * cj
        C_new = C * cj + 1j * D * n_j * sj
        D_new = 1j * C * sj / n_j + D * cj
        A, B, C, D = A_new, B_new, C_new, D_new
        
    n_s = problem["substrate_index"]
    n_0 = problem["ambient_index"]
    # Reflectance: r = (n_0 * (A + B * n_s) - (C + D * n_s)) / (n_0 * (A + B * n_s) + (C + D * n_s))
    num = n_0 * (A + B * n_s) - (C + D * n_s)
    den = n_0 * (A + B * n_s) + (C + D * n_s)
    
    # Handle division by zero if any
    den_safe = np.where(np.abs(den) < 1e-12, 1e-12, den)
    r = num / den_safe
    R = np.abs(r) ** 2
    return 100.0 * np.mean(R)

def design(problem):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    band = problem["band_nm"]
    wl_center = 0.5 * (band[0] + band[1])
    max_layers = problem["max_layers"]
    max_total = problem["max_total_nm"]
    max_layer = problem["max_layer_nm"]
    materials_dict = problem["materials"]
    materials_list = list(materials_dict.keys())
    
    # Precompute quarter-wave thicknesses at center wavelength for initialization
    qw = {}
    for m, n in materials_dict.items():
        qw[m] = wl_center / (4.0 * n)
        
    def create_individual():
        """Create a random valid individual."""
        n_layers = random.randint(1, min(max_layers, 10)) # Start with reasonable size
        ind = []
        total_th = 0
        for _ in range(n_layers):
            m = random.choice(materials_list)
            # Thickness between 0.2x and 1.5x quarter wave, but respecting max_layer
            base_th = qw[m]
            th = random.uniform(0.2 * base_th, 1.5 * base_th)
            th = max(0.0, min(max_layer, th))
            
            # Check total thickness constraint
            if total_th + th > max_total:
                th = max(0.0, max_total - total_th)
                if th < 1.0: # Too small, skip or break
                    continue
            total_th += th
            ind.append((m, th))
        return ind

    def evaluate(ind):
        if not ind:
            return 100.0
        mats = [x[0] for x in ind]
        ths = [x[1] for x in ind]
        # Check constraints
        if len(mats) > max_layers:
            return 100.0
        if sum(ths) > max_total + 1e-6:
            return 100.0
        for t in ths:
            if t < -1e-6 or t > max_layer + 1e-6:
                return 100.0
        # Clamp small negatives
        ths = [max(0.0, t) for t in ths]
        return _mean_reflectance(mats, ths, problem)

    def mutate(ind):
        """Apply mutations to an individual."""
        if not ind:
            return ind
        
        new_ind = [x for x in ind]
        
        # 1. Material Mutation
        if random.random() < 0.3:
            idx = random.randint(0, len(new_ind) - 1)
            m, t = new_ind[idx]
            other_m = [x for x in materials_list if x != m][0]
            new_ind[idx] = (other_m, t)
            
        # 2. Thickness Mutation
        if random.random() < 0.7:
            idx = random.randint(0, len(new_ind) - 1)
            m, t = new_ind[idx]
            # Gaussian perturbation
            perturbation = random.gauss(0, qw[m] * 0.2)
            new_t = t + perturbation
            new_t = max(0.0, min(max_layer, new_t))
            new_ind[idx] = (m, new_t)
            
        # 3. Insertion
        if random.random() < 0.1 and len(new_ind) < max_layers:
            m = random.choice(materials_list)
            t = qw[m] * random.uniform(0.5, 1.0)
            t = max(0.0, min(max_layer, t))
            # Check if we can add it
            current_sum = sum(x[1] for x in new_ind)
            if current_sum + t <= max_total:
                pos = random.randint(0, len(new_ind))
                new_ind.insert(pos, (m, t))
                
        # 4. Deletion
        if random.random() < 0.1 and len(new_ind) > 1:
            idx = random.randint(0, len(new_ind) - 1)
            new_ind.pop(idx)
            
        # 5. Swap
        if random.random() < 0.1 and len(new_ind) > 1:
            i, j = random.sample(range(len(new_ind)), 2)
            new_ind[i], new_ind[j] = new_ind[j], new_ind[i]
            
        return new_ind

    def crossover(ind1, ind2):
        """Single point crossover."""
        if len(ind1) == 0 or len(ind2) == 0:
            return ind1, ind2
            
        # Pick crossover point based on the shorter one to ensure valid children
        min_len = min(len(ind1), len(ind2))
        if min_len == 0:
            return ind1, ind2
            
        point = random.randint(1, min_len)
        
        child1 = ind1[:point] + ind2[point:]
        child2 = ind2[:point] + ind1[point:]
        
        # Enforce constraints on children
        # 1. Max layers
        if len(child1) > max_layers:
            child1 = child1[:max_layers]
        if len(child2) > max_layers:
            child2 = child2[:max_layers]
            
        # 2. Max total thickness
        # We need to scale down if it exceeds max_total
        def enforce_total(child):
            total = sum(t for _, t in child)
            if total > max_total:
                scale = max_total / total
                child = [(m, t * scale) for m, t in child]
            # Clamp individual layers
            child = [(m, max(0.0, min(max_layer, t))) for m, t in child]
            # Re-check total after clamping (clamping might reduce sum, but if we scaled, it should be <= max_total)
            # However, if we just clamped without scaling, sum might still be > max_total? 
            # No, scaling ensures sum <= max_total. Clamping individual layers might make sum < max_total.
            # If we didn't scale (total <= max_total), clamping might make sum < max_total.
            # So sum is always <= max_total.
            return child
            
        child1 = enforce_total(child1)
        child2 = enforce_total(child2)
        
        return child1, child2

    # Initialize Population
    pop_size = 60
    population = []
    for _ in range(pop_size):
        ind = create_individual()
        population.append(ind)
        
    # Evaluate initial population
    scores = [evaluate(ind) for ind in population]
    
    # Elitism
    elite_size = 5
    best_score = min(scores)
    best_ind = population[scores.index(best_score)]
    
    generations = 0
    max_generations = 10000 # Will be limited by time
    
    while time.time() < deadline - 10:
        generations += 1
        
        # Sort by score (lower is better)
        sorted_inds = sorted(zip(scores, population), key=lambda x: x[0])
        scores = [s for s, _ in sorted_inds]
        population = [ind for _, ind in sorted_inds]
        
        # Update best
        if scores[0] < best_score:
            best_score = scores[0]
            best_ind = population[0]
            
        # Create new population
        new_population = []
        
        # Elitism
        for i in range(elite_size):
            new_population.append(population[i])
            
        # Fill rest with offspring
        while len(new_population) < pop_size:
            if time.time() > deadline - 10:
                break
                
            # Tournament Selection
            k = 3
            participants = random.sample(range(len(population)), k)
            winner_idx = min(participants, key=lambda i: scores[i])
            parent1 = population[winner_idx]
            
            # Second parent: random or tournament
            if random.random() < 0.5:
                participants2 = random.sample(range(len(population)), k)
                winner_idx2 = min(participants2, key=lambda i: scores[i])
                parent2 = population[winner_idx2]
            else:
                parent2 = random.choice(population)
                
            # Crossover
            child1, child2 = crossover(parent1, parent2)
            
            # Mutation
            child1 = mutate(child1)
            child2 = mutate(child2)
            
            new_population.append(child1)
            if len(new_population) < pop_size:
                new_population.append(child2)
                
        population = new_population
        scores = [evaluate(ind) for ind in population]
        
        # Check for convergence or improvement stagnation? 
        # For now, just run until time.

    # Final Polish: Nelder-Mead on the best individual found
    if best_ind:
        best_mats = [m for m, _ in best_ind]
        best_ths = [t for _, t in best_ind]
        
        # Simple local polish using Nelder-Mead on thicknesses only, keeping sequence fixed
        # This helps refine the continuous part
        if time.time() < deadline - 2:
            def objective(th_list):
                ths_clamped = [max(0.0, min(max_layer, t)) for t in th_list]
                total = sum(ths_clamped)
                if total > max_total:
                    scale = max_total / total
                    ths_clamped = [t * scale for t in ths_clamped]
                return _mean_reflectance(best_mats, ths_clamped, problem)
            
            # Manual Nelder-Mead for small N
            n = len(best_ths)
            if n > 0 and n < 15: # Nelder-Mead is slow for high N
                # Initial simplex
                simplex = [list(best_ths)]
                for i in range(n):
                    pt = list(best_ths)
                    pt[i] += 10.0
                    pt[i] = max(0.0, min(max_layer, pt[i]))
                    simplex.append(pt)
                
                scores_nm = [objective(pt) for pt in simplex]
                
                alpha = 1.0
                gamma = 2.0
                rho = 0.5
                sigma = 0.5
                max_iters = 100
                
                for _ in range(max_iters):
                    if time.time() > deadline - 2:
                        break
                    order = np.argsort(scores_nm)
                    simplex = [simplex[i] for i in order]
                    scores_nm = [scores_nm[i] for i in order]
                    
                    centroid = np.mean(simplex[:-1], axis=0)
                    xr = centroid + alpha * (centroid - simplex[-1])
                    xr = [max(0.0, min(max_layer, t)) for t in xr]
                    fr = objective(xr)
                    
                    if fr < scores_nm[0]:
                        xe = centroid + gamma * (centroid - simplex[-1])
                        xe = [max(0.0, min(max_layer, t)) for t in xe]
                        fe = objective(xe)
                        if fe < fr:
                            simplex[-1] = xe
                            scores_nm[-1] = fe
                        else:
                            simplex[-1] = xr
                            scores_nm[-1] = fr
                    elif fr < scores_nm[-2]:
                        simplex[-1] = xr
                        scores_nm[-1] = fr
                    else:
                        xc = centroid + rho * (simplex[-1] - centroid)
                        xc = [max(0.0, min(max_layer, t)) for t in xc]
                        fc = objective(xc)
                        if fc <= fr:
                            simplex[-1] = xc
                            scores_nm[-1] = fc
                        else:
                            # Shrink
                            for i in range(1, n + 1):
                                simplex[i] = [simplex[0][j] + sigma * (simplex[i][j] - simplex[0][j]) for j in range(n)]
                                simplex[i] = [max(0.0, min(max_layer, t)) for t in simplex[i]]
                                scores_nm[i] = objective(simplex[i])
                            if np.max(scores_nm) - np.min(scores_nm) < 1e-6:
                                break
                
                # Extract best from NM
                best_nm_score = min(scores_nm)
                best_nm_ths = simplex[scores_nm.index(best_nm_score)]
                if best_nm_score < best_score:
                    best_score = best_nm_score
                    best_ind = list(zip(best_mats, best_nm_ths))

    # Return best result
    if not best_ind:
        # Fallback
        return ["L"], [qw["L"]]
        
    mats = [m for m, _ in best_ind]
    ths = [t for _, t in best_ind]
    
    # Final constraint check
    if sum(ths) > max_total:
        scale = max_total / sum(ths)
        ths = [t * scale for t in ths]
    ths = [max(0.0, min(max_layer, t)) for t in ths]
    
    return mats, ths
