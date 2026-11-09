"""
Broadband antireflection coating design: Joint Pattern and Thickness Optimization via Differential Evolution.
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
    
    # Handle empty or invalid inputs gracefully
    if len(materials) == 0 or len(thicknesses) == 0:
         return 100.0 * ((problem["substrate_index"] - problem["ambient_index"]) /
                        (problem["substrate_index"] + problem["ambient_index"])) ** 2

    ns = np.array([n_mat[m] for m in materials], dtype=float)
    ds = np.asarray(thicknesses, dtype=float)

    # delta_j = 2*pi*n_j*d_j/lambda -> (N, K)
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

def design(problem):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
    band = problem["band_nm"]
    wl_center = 0.5 * (band[0] + band[1])
    max_layers = problem["max_layers"]
    max_total = problem["max_total_nm"]
    max_layer = problem["max_layer_nm"]
    materials_dict = problem["materials"]
    
    n_L = materials_dict['L']
    n_H = materials_dict['H']
    
    # DE Parameters
    pop_size = 40
    F = 0.8  # Differential weight
    CR = 0.9  # Crossover probability
    
    # Encoding:
    # Vector length = max_layers (20)
    # First max_layers elements: 0.0 (L) or 1.0 (H) -> Material
    # Next max_layers elements: Thickness in nm
    # We will evolve a fixed-length vector of size 2 * max_layers.
    # Layers with thickness < epsilon or material undefined? 
    # Better: Always use all max_layers layers to simplify DE. 
    # If a layer is "empty", its thickness is 0.
    # However, the problem allows up to 20 layers. Using exactly 20 with some zero thickness is equivalent.
    # But characteristic matrix with d=0 is identity, so it's safe.
    
    # Initialize Population
    population = []
    for i in range(pop_size):
        # Random materials
        mats = np.random.choice([0.0, 1.0], size=max_layers)
        # Random thicknesses
        # Initialize with some reasonable structure to help convergence? 
        # Or just random uniform?
        # Let's use quarter-wave randomization for better start
        ths = np.zeros(max_layers)
        for j in range(max_layers):
            if mats[j] == 0.0:
                ths[j] = random.uniform(10, wl_center / n_L)
            else:
                ths[j] = random.uniform(10, wl_center / n_H)
        
        # Ensure constraints
        total = sum(ths)
        if total > max_total:
            ths *= (max_total / total)
        
        # Combine into single vector
        vec = np.concatenate([mats, ths])
        population.append(vec)
        
    # Evaluate initial population
    def evaluate(vec):
        mats = vec[:max_layers]
        ths = vec[max_layers:]
        
        # Decode materials
        mat_labels = []
        for m in mats:
            if m > 0.5:
                mat_labels.append('H')
            else:
                mat_labels.append('L')
                
        # Clip thicknesses
        ths = np.clip(ths, 0.0, max_layer)
        
        # Check total thickness constraint
        if np.sum(ths) > max_total:
             # Penalize heavily
             return 1000.0 + (np.sum(ths) - max_total)
             
        # Compute reflectance
        # If all thicknesses are 0, reflectance is bare glass
        if np.all(ths == 0):
             return 100.0 * ((problem["substrate_index"] - problem["ambient_index"]) /
                            (problem["substrate_index"] + problem["ambient_index"])) ** 2
        
        return _mean_reflectance(mat_labels, ths, problem)
        
    scores = np.array([evaluate(vec) for vec in population])
    best_idx = np.argmin(scores)
    best_vec = population[best_idx].copy()
    best_score = scores[best_idx]
    
    # DE Loop
    # Estimate time per eval
    # 401 points, 20 layers. ~0.5ms?
    # 140s / 0.0005s = 280,000 evals.
    # Pop size 40. 7000 generations.
    
    gen = 0
    while time.time() < deadline:
        gen += 1
        # Create new population
        new_population = [best_vec.copy()] # Elitism: keep best
        new_scores = [best_score]
        
        for i in range(1, pop_size):
            # Select 3 distinct random individuals (excluding i if possible, but standard DE just picks random)
            a_idx, b_idx, c_idx = np.random.choice(pop_size, 3, replace=False)
            # Ensure they are distinct from i? Not strictly required but helps.
            while a_idx == b_idx or b_idx == c_idx or a_idx == c_idx:
                 a_idx, b_idx, c_idx = np.random.choice(pop_size, 3, replace=False)
            
            a = population[a_idx]
            b = population[b_idx]
            c = population[c_idx]
            
            # Mutation: v = a + F * (b - c)
            v = a + F * (b - c)
            
            # Crossover
            trial = np.zeros_like(v)
            j_rand = np.random.randint(0, len(v))
            for j in range(len(v)):
                if random.random() < CR or j == j_rand:
                    trial[j] = v[j]
                else:
                    trial[j] = population[i][j]
            
            # Binary repair for material part (first half)
            # DE produces continuous values. We need to binarize for materials.
            # Threshold at 0.5
            trial_mats = np.where(trial[:max_layers] > 0.5, 1.0, 0.0)
            trial[:max_layers] = trial_mats
            
            # Clip thicknesses
            trial_ths = np.clip(trial[max_layers:], 0.0, max_layer)
            trial[max_layers:] = trial_ths
            
            # Evaluate
            score = evaluate(trial)
            
            # Selection
            if score <= scores[i]:
                new_population.append(trial)
                new_scores.append(score)
            else:
                new_population.append(population[i])
                new_scores.append(scores[i])
                
        population = new_population
        scores = np.array(new_scores)
        
        # Update best
        current_best_idx = np.argmin(scores)
        if scores[current_best_idx] < best_score:
            best_score = scores[current_best_idx]
            best_vec = population[current_best_idx].copy()
            
        # Periodic restart if stuck?
        # If no improvement for 100 generations, randomize 10% of population
        if gen % 100 == 0:
             # Check stagnation roughly
             if np.min(scores) > best_score * 1.01: # 1% worse than best
                 for _ in range(pop_size // 5):
                     idx = random.randint(0, pop_size-1)
                     population[idx] = np.random.uniform(0, 1, size=max_layers) # Random mats
                     population[idx][max_layers:] = np.random.uniform(0, max_layer, size=max_layers) # Random ths
                     scores[idx] = evaluate(population[idx])
                     if scores[idx] < best_score:
                         best_score = scores[idx]
                         best_vec = population[idx].copy()
        
        # Check time for polish
        if time.time() > deadline - 20:
            break
            
    # Final Polish with Nelder-Mead on the best vector found
    # We need to handle the mixed discrete/continuous nature.
    # NM works best on continuous. 
    # Strategy: Fix the material sequence from best_vec, and optimize thicknesses with NM.
    # Then, try flipping a few materials and re-polish?
    # Given time constraints, just polish thicknesses for the best material sequence.
    
    best_mats = best_vec[:max_layers]
    best_ths = best_vec[max_layers:]
    
    # Decode best mats
    best_mat_labels = []
    for m in best_mats:
        if m > 0.5:
            best_mat_labels.append('H')
        else:
            best_mat_labels.append('L')
            
    # Filter out zero-thickness layers for cleaner NM? 
    # NM handles 20 variables fine.
    
    def nelder_mead_polish(mat_labels, init_ths, deadline):
        ths = list(init_ths)
        # Ensure constraints
        total = sum(ths)
        if total > max_total:
            scale = max_total / total
            ths = [t * scale for t in ths]
        for i in range(len(ths)):
            ths[i] = max(0.0, min(max_layer, ths[i]))
            
        n = len(ths)
        # Simplex
        simplex = [ths.copy()]
        step = 5.0
        for i in range(n):
            pt = ths.copy()
            pt[i] = max(0.0, min(max_layer, pt[i] + step))
            simplex.append(pt)
            
        scores = [_mean_reflectance(mat_labels, simplex[i], problem) for i in range(n + 1)]
        
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        
        it = 0
        max_iters = 200
        while it < max_iters and time.time() < deadline:
            it += 1
            order = np.argsort(scores)
            simplex = [simplex[i] for i in order]
            scores = [scores[i] for i in order]
            
            centroid = np.mean(simplex[:-1], axis=0)
            
            xr = centroid + alpha * (centroid - simplex[-1])
            # Constrain xr
            xr_total = sum(xr)
            if xr_total > max_total:
                scale = max_total / xr_total
                xr = [x * scale for x in xr]
            xr = [max(0.0, min(max_layer, x)) for x in xr]
            
            fr = _mean_reflectance(mat_labels, xr, problem)
            
            if fr < scores[0]:
                xe = centroid + gamma * (centroid - simplex[-1])
                xe_total = sum(xe)
                if xe_total > max_total:
                    scale = max_total / xe_total
                    xe = [x * scale for x in xe]
                xe = [max(0.0, min(max_layer, x)) for x in xe]
                fe = _mean_reflectance(mat_labels, xe, problem)
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
                xc = centroid + rho * (simplex[-1] - centroid)
                xc_total = sum(xc)
                if xc_total > max_total:
                    scale = max_total / xc_total
                    xc = [x * scale for x in xc]
                xc = [max(0.0, min(max_layer, x)) for x in xc]
                fc = _mean_reflectance(mat_labels, xc, problem)
                if fc <= fr:
                    simplex[-1] = xc
                    scores[-1] = fc
                else:
                    for i in range(1, n + 1):
                        simplex[i] = [simplex[0][j] + sigma * (simplex[i][j] - simplex[0][j]) for j in range(n)]
                        # Constrain
                        s_total = sum(simplex[i])
                        if s_total > max_total:
                            scale = max_total / s_total
                            simplex[i] = [x * scale for x in simplex[i]]
                        simplex[i] = [max(0.0, min(max_layer, x)) for x in simplex[i]]
                        scores[i] = _mean_reflectance(mat_labels, simplex[i], problem)
                    if np.max(scores) - np.min(scores) < 1e-7:
                        break
                        
        return simplex[0], scores[0]
        
    # Run Polish
    if time.time() < deadline - 5:
        polished_ths, polished_score = nelder_mead_polish(best_mat_labels, best_ths, deadline - 5)
        if polished_score < best_score:
            best_ths = polished_ths
            best_score = polished_score
            best_vec[max_layers:] = best_ths
            
    # Final Result
    # Extract non-zero thickness layers? 
    # The problem asks for a list of labels and thicknesses.
    # We can include zero-thickness layers or filter them.
    # Filtering is cleaner and reduces "layers" count if any are 0.
    # But keeping them is valid if thickness is 0.
    # Let's filter out layers with thickness < 1e-6 to be clean.
    
    final_mats = []
    final_ths = []
    for i in range(max_layers):
        if best_ths[i] > 1e-6:
            if best_mats[i] > 0.5:
                final_mats.append('H')
            else:
                final_mats.append('L')
            final_ths.append(best_ths[i])
            
    # If we filtered all, return empty?
    # If empty, reflectance is bare glass.
    # Check if we should add a layer if all are 0?
    # DE should find non-zero.
    
    # Final validation check
    if not final_mats:
         return [], []
         
    # Ensure sum constraint one last time
    total = sum(final_ths)
    if total > max_total:
         scale = max_total / total
         final_ths = [t * scale for t in final_ths]
         
    # Ensure layer constraint
    for i in range(len(final_ths)):
         final_ths[i] = max(0.0, min(max_layer, final_ths[i]))
         
    return final_mats, final_ths
