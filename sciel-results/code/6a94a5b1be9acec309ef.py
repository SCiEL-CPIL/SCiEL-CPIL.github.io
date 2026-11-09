"""
Broadband antireflection coating design: Genetic Algorithm co-optimizing sequence and thicknesses.
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

def _constrain_thicknesses(ths, max_total, max_layer):
    """Constrain thicknesses to be within bounds and total sum."""
    ths = np.asarray(ths, dtype=float)
    ths = np.clip(ths, 0.0, max_layer)
    total = np.sum(ths)
    if total > max_total:
        scale = max_total / total
        ths = ths * scale
        ths = np.clip(ths, 0.0, max_layer)
    return ths

def design(problem):
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit
    
    band = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total = problem["max_total_nm"]
    max_layer = problem["max_layer_nm"]
    materials_dict = problem["materials"]
    
    n_L = materials_dict['L']
    n_H = materials_dict['H']
    
    def evaluate_genome(materials, thicknesses):
        ths = _constrain_thicknesses(thicknesses, max_total, max_layer)
        score = _mean_reflectance(materials, ths, problem)
        return score, ths

    # Initialize population
    pop_size = 50
    population = []
    scores = []
    
    # Elite initial guess: alternating L/H with quarter-wave thicknesses at center wavelength
    wl_center = 0.5 * (band[0] + band[1])
    init_mats = []
    init_ths = []
    for i in range(max_layers):
        if i % 2 == 0:
            init_mats.append('L')
            init_ths.append(wl_center / (4.0 * n_L))
        else:
            init_mats.append('H')
            init_ths.append(wl_center / (4.0 * n_H))
            
    population.append((init_mats, init_ths))
    score, ths = evaluate_genome(init_mats, init_ths)
    population[-1] = (init_mats, ths)
    scores.append(score)
    
    # Random initialization for rest
    for _ in range(pop_size - 1):
        mats = ['L' if random.random() < 0.5 else 'H' for _ in range(max_layers)]
        ths = [random.uniform(0, max_layer) for _ in range(max_layers)]
        s, c_ths = evaluate_genome(mats, ths)
        population.append((mats, c_ths))
        scores.append(s)
    
    best_score = min(scores)
    best_idx = scores.index(best_score)
    best_mats, best_ths = population[best_idx]
    
    # GA Parameters
    tournament_size = 5
    mutation_rate = 0.1
    crossover_rate = 0.8
    elite_count = 5
    
    while time.time() < deadline - 5:
        # Selection and Crossover
        new_population = []
        
        # Keep elites
        sorted_indices = np.argsort(scores)
        for i in range(elite_count):
            new_population.append(population[sorted_indices[i]])
            
        while len(new_population) < pop_size:
            # Tournament selection
            participants = random.sample(range(pop_size), tournament_size)
            winner1_idx = min(participants, key=lambda i: scores[i])
            winner2_idx = min(participants, key=lambda i: scores[i])
            
            parent1 = population[winner1_idx]
            parent2 = population[winner2_idx]
            
            if random.random() < crossover_rate:
                # Uniform crossover for materials
                child_mats = [p1 if random.random() < 0.5 else p2 for p1, p2 in zip(parent1[0], parent2[0])]
                
                # Blend crossover for thicknesses
                alpha = random.random()
                child_ths = [alpha * t1 + (1 - alpha) * t2 for t1, t2 in zip(parent1[1], parent2[1])]
            else:
                child_mats = parent1[0][:]
                child_ths = parent1[1][:]
                
            # Mutation
            if random.random() < mutation_rate:
                # Mutate material
                idx = random.randint(0, max_layers - 1)
                if child_mats[idx] == 'L':
                    child_mats[idx] = 'H'
                else:
                    child_mats[idx] = 'L'
                    
            # Mutate thicknesses
            for i in range(max_layers):
                if random.random() < mutation_rate:
                    child_ths[i] += random.gauss(0, 20.0)
                    child_ths[i] = max(0.0, min(max_layer, child_ths[i]))
            
            # Constrain and evaluate
            score, c_ths = evaluate_genome(child_mats, child_ths)
            new_population.append((child_mats, c_ths))
            scores.append(score) # Temporarily store, will be cleared
            
            if score < best_score:
                best_score = score
                best_mats = child_mats
                best_ths = c_ths
                
        # Replace population
        population = new_population
        # Re-evaluate scores for the new population to keep them in sync
        scores = []
        for mats, ths in population:
            s, _ = evaluate_genome(mats, ths)
            scores.append(s)
            
    return list(best_mats), list(best_ths)
