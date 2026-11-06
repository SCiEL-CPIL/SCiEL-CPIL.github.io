"""Genetic Algorithm approach for broadband antireflection coating design."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import random

def _mean_reflectance_vec(materials, thicknesses, problem):
    """
    Compute mean reflectance (percent) over the band using vectorized characteristic matrix method.
    materials: list of 'L'/'H'
    thicknesses: list of floats
    """
    n_amb = problem["ambient_index"]
    n_sub = problem["substrate_index"]
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    band = problem["band_nm"]
    wls = np.linspace(band[0], band[1], 401)
    
    n_mat = np.array([n_L if m == 'L' else n_H for m in materials])
    d = np.array(thicknesses, dtype=float)
    
    # Precompute phases for all wavelengths and layers
    # shape: (n_layers, n_wls)
    phases = 2.0 * np.pi * n_mat[:, None] * d[:, None] / wls[None, :]
    
    cos_p = np.cos(phases)
    sin_p = np.sin(phases)
    
    # Initialize characteristic matrix product for each wavelength
    # M starts as Identity for each wavelength
    # We iterate through layers and update M for all wavelengths in parallel
    # M is shape (n_wls, 2, 2)
    M = np.zeros((len(wls), 2, 2), dtype=complex)
    M[:, 0, 0] = 1.0
    M[:, 1, 1] = 1.0
    
    for i in range(len(materials)):
        n_i = n_mat[i]
        # B matrix for each wavelength
        # B = [[cos, i*sin/n], [i*n*sin, cos]]
        B = np.zeros((len(wls), 2, 2), dtype=complex)
        B[:, 0, 0] = cos_p[i]
        B[:, 0, 1] = 1j * sin_p[i] / n_i
        B[:, 1, 0] = 1j * n_i * sin_p[i]
        B[:, 1, 1] = cos_p[i]
        
        # M = M @ B
        # (n_wls, 2, 2) @ (n_wls, 2, 2)
        M = np.einsum('wij,wjk->wik', M, B)
        
    # Calculate Reflectance
    Y = M[:, 0, 1] / M[:, 1, 1]
    r = (n_amb - Y) / (n_amb + Y)
    R = np.abs(r) ** 2
    return np.mean(R) * 100.0

def _is_valid(materials, thicknesses, problem):
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    max_layers = problem["max_layers"]
    
    if len(materials) != len(thicknesses):
        return False
    if len(materials) > max_layers:
        return False
    if any(t < 0 or t > max_layer_nm for t in thicknesses):
        return False
    if sum(thicknesses) > max_total_nm:
        return False
    return True

def design(problem):
    t0 = time.time()
    deadline = t0 + 140.0  # Safety margin
    
    n_amb = problem["ambient_index"]
    n_sub = problem["substrate_index"]
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    centre = 0.5 * (problem["band_nm"][0] + problem["band_nm"][1])
    
    # Fallback: single quarter-wave layer
    best_mats = ["L"]
    best_thick = [centre / (4.0 * n_L)]
    best_score = _mean_reflectance_vec(best_mats, best_thick, problem)
    
    # Initial population for GA
    pop_size = 100
    n_layers_max = max_layers
    
    # Helper to generate random valid individual
    def random_individual():
        # Random number of layers between 1 and max_layers
        n_layers = random.randint(1, n_layers_max)
        mats = [random.choice(['L', 'H']) for _ in range(n_layers)]
        
        # Random thicknesses, ensuring sum constraint is likely met
        # We can generate random weights and scale them
        weights = np.random.rand(n_layers)
        total_w = np.sum(weights)
        if total_w == 0:
            weights = np.ones(n_layers)
            total_w = n_layers
        
        # Max total thickness we can use is max_total_nm
        # But individual layers must be <= max_layer_nm
        # Let's just generate random thicknesses in [0, max_layer_nm] and check sum
        # If sum is too high, scale down.
        
        thicks = np.random.rand(n_layers) * max_layer_nm
        
        # Ensure sum constraint
        if np.sum(thicks) > max_total_nm:
            thicks = thicks * (max_total_nm / np.sum(thicks))
        
        return mats, thicks.tolist()
    
    # Initialize population
    population = []
    scores = []
    for _ in range(pop_size):
        mats, thicks = random_individual()
        if _is_valid(mats, thicks, problem):
            score = _mean_reflectance_vec(mats, thicks, problem)
            population.append((mats, thicks, score))
            scores.append(score)
    
    if not population:
        return best_mats, best_thick
    
    # Update best
    min_idx = np.argmin(scores)
    if scores[min_idx] < best_score:
        best_score = scores[min_idx]
        best_mats = population[min_idx][0]
        best_thick = population[min_idx][1]
    
    # GA Loop
    mutation_rate = 0.2
    crossover_rate = 0.7
    elite_size = 10
    max_generations = 500
    
    gen = 0
    while time.time() < deadline - 5.0 and gen < max_generations:
        gen += 1
        
        # Sort population by score
        population.sort(key=lambda x: x[2])
        
        # Elitism
        new_population = population[:elite_size]
        
        # Fill rest with offspring
        while len(new_population) < pop_size:
            # Selection (Tournament)
            def select():
                candidates = random.sample(population, 3)
                return min(candidates, key=lambda x: x[2])
            
            parent1 = select()
            parent2 = select()
            
            if random.random() < crossover_rate:
                # Crossover
                # Materials
                len1 = len(parent1[0])
                len2 = len(parent2[0])
                if len1 == len2:
                    # Uniform crossover
                    child_mats = []
                    child_thicks = []
                    for i in range(len1):
                        if random.random() < 0.5:
                            child_mats.append(parent1[0][i])
                            child_thicks.append(parent1[1][i])
                        else:
                            child_mats.append(parent2[0][i])
                            child_thicks.append(parent2[1][i])
                else:
                    # Different lengths: take shorter or random
                    if random.random() < 0.5:
                        child_mats = parent1[0][:]
                        child_thicks = parent1[1][:]
                    else:
                        child_mats = parent2[0][:]
                        child_thicks = parent2[1][:]
            else:
                # No crossover
                if random.random() < 0.5:
                    child_mats = parent1[0][:]
                    child_thicks = parent1[1][:]
                else:
                    child_mats = parent2[0][:]
                    child_thicks = parent2[1][:]
            
            # Mutation
            # 1. Change material
            if random.random() < mutation_rate:
                idx = random.randint(0, len(child_mats) - 1)
                child_mats[idx] = 'H' if child_mats[idx] == 'L' else 'L'
            
            # 2. Change thickness
            if random.random() < mutation_rate:
                idx = random.randint(0, len(child_thicks) - 1)
                # Gaussian mutation
                delta = random.gauss(0, 20)
                new_t = child_thicks[idx] + delta
                if new_t < 0: new_t = 0
                if new_t > max_layer_nm: new_t = max_layer_nm
                child_thicks[idx] = new_t
            
            # 3. Add layer
            if random.random() < 0.1 * mutation_rate and len(child_mats) < max_layers:
                pos = random.randint(0, len(child_mats))
                child_mats.insert(pos, random.choice(['L', 'H']))
                child_thicks.insert(pos, random.uniform(10, 100))
            
            # 4. Remove layer
            if random.random() < 0.1 * mutation_rate and len(child_mats) > 1:
                pos = random.randint(0, len(child_mats) - 1)
                child_mats.pop(pos)
                child_thicks.pop(pos)
            
            # 5. Change number of layers (reset)
            if random.random() < 0.05:
                child_mats, child_thicks = random_individual()
            
            # Validate
            if _is_valid(child_mats, child_thicks, problem):
                score = _mean_reflectance_vec(child_mats, child_thicks, problem)
                new_population.append((child_mats, child_thicks, score))
            else:
                # If invalid, just add parent to maintain population size
                new_population.append(parent1)
        
        population = new_population
        
        # Update best
        current_best = min(population, key=lambda x: x[2])
        if current_best[2] < best_score:
            best_score = current_best[2]
            best_mats = current_best[0]
            best_thick = current_best[1]
        
        # Check time
        if time.time() > deadline - 10:
            break
    
    # Local refinement of best solution using Nelder-Mead on thicknesses
    if time.time() < deadline - 2 and len(best_mats) > 0:
        try:
            from scipy.optimize import minimize
            
            def obj(x):
                x = np.clip(x, 0, max_layer_nm)
                if np.sum(x) > max_total_nm:
                    return 100.0
                return _mean_reflectance_vec(best_mats, list(x), problem)
            
            x0 = np.array(best_thick, dtype=float)
            res = minimize(obj, x0, method='Nelder-Mead', options={'maxiter': 200, 'xatol': 1e-3, 'fatol': 1e-6})
            
            if res.fun < best_score:
                x_opt = np.clip(res.x, 0, max_layer_nm)
                if np.sum(x_opt) <= max_total_nm:
                    best_score = float(res.fun)
                    best_thick = list(x_opt)
        except Exception:
            pass
            
    return best_mats, best_thick
# EVOLVE-BLOCK-END
