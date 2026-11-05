"""Broadband antireflection coating design via Genetic Algorithm with vectorized evaluation."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def _vectorized_mean_reflectance_pct(thicknesses, materials, n_ambient, n_substrate, n_L, n_H, wls):
    """
    Compute mean reflectance in percent for a single design across all wavelengths.
    Vectorized over wavelengths for speed.
    
    thicknesses: numpy array of physical thicknesses in nm
    materials: numpy array of 'L' or 'H' strings (or indices 0/1)
    wls: numpy array of wavelengths in nm
    """
    n_layers = len(thicknesses)
    if n_layers == 0:
        r = (n_ambient - n_substrate) / (n_ambient + n_substrate)
        return (r * r) * 100.0
    
    # Initialize M for all wavelengths
    M11 = np.ones_like(wls, dtype=complex)
    M12 = np.zeros_like(wls, dtype=complex)
    M21 = np.zeros_like(wls, dtype=complex)
    M22 = np.ones_like(wls, dtype=complex)
    
    for i in range(n_layers):
        if materials[i] == 'H':
            n = n_H
        else:
            n = n_L
            
        d = thicknesses[i]
        beta = 2.0 * np.pi * n * d / wls
        cos_b = np.cos(beta)
        sin_b = np.sin(beta)
        
        # Layer matrix components
        L11 = cos_b
        L12 = -1j * sin_b / n
        L21 = 1j * n * sin_b
        L22 = cos_b
        
        # M = M @ L
        new_M11 = M11 * L11 + M12 * L21
        new_M12 = M11 * L12 + M12 * L22
        new_M21 = M21 * L11 + M22 * L21
        new_M22 = M21 * L12 + M22 * L22
        M11, M12, M21, M22 = new_M11, new_M12, new_M21, new_M22
    
    # Admittance Y
    Y = (M11 * n_substrate + M12) / (M21 * n_substrate + M22)
    r = (n_ambient - Y) / (n_ambient + Y)
    R = (r * r.conjugate()).real
    return np.mean(R) * 100.0

def design(problem):
    """Design a broadband antireflection coating using a Genetic Algorithm."""
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    n_ambient = problem["ambient_index"]
    n_substrate = problem["substrate_index"]
    wl_min, wl_max = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # 401 equally spaced wavelengths
    wls = np.linspace(wl_min, wl_max, 401)
    
    # Pre-compute some constants
    qw_L = 0.5 * (wl_min + wl_max) / (4.0 * n_L)
    qw_H = 0.5 * (wl_min + wl_max) / (4.0 * n_H)
    
    def is_valid(materials, thicknesses):
        if len(materials) > max_layers:
            return False
        if any(t < 0 or t > max_layer_nm for t in thicknesses):
            return False
        if sum(thicknesses) > max_total_nm:
            return False
        return True

    def evaluate(materials, thicknesses):
        if not is_valid(materials, thicknesses):
            return float('inf')
        return _vectorized_mean_reflectance_pct(thicknesses, materials, n_ambient, n_substrate, n_L, n_H, wls)
    
    # GA Parameters
    pop_size = 60
    max_generations = 50
    
    # Initialize population
    # Mix of random designs and some structured initializations (quarter-wave stacks)
    population = []
    
    # Add some good initial guesses
    # 1. Single L layer
    population.append((["L"], [qw_L]))
    # 2. Single H layer
    population.append((["H"], [qw_H]))
    # 3. H-L-H
    population.append((["H", "L", "H"], [qw_H, qw_L, qw_H]))
    # 4. L-H-L
    population.append((["L", "H", "L"], [qw_L, qw_H, qw_L]))
    
    # Fill rest with random designs
    while len(population) < pop_size:
        n_layers = np.random.randint(1, max_layers + 1)
        mats = []
        ths = []
        for _ in range(n_layers):
            mat = 'H' if np.random.random() < 0.5 else 'L'
            mats.append(mat)
            # Random thickness, biased towards quarter-wave range
            base = qw_H if mat == 'H' else qw_L
            th = base * (0.5 + np.random.random())
            th = min(th, max_layer_nm)
            ths.append(th)
        population.append((mats, ths))
    
    best_materials = None
    best_thicknesses = None
    best_score = float('inf')
    
    # Evaluate initial population
    for mats, ths in population:
        score = evaluate(mats, ths)
        if score < best_score:
            best_score = score
            best_materials = mats[:]
            best_thicknesses = ths[:]
    
    # GA Loop
    for gen in range(max_generations):
        if time.time() > deadline:
            break
            
        # Evaluate population
        scores = []
        for mats, ths in population:
            scores.append(evaluate(mats, ths))
            
        # Update global best
        min_idx = np.argmin(scores)
        if scores[min_idx] < best_score:
            best_score = scores[min_idx]
            best_materials = population[min_idx][0][:]
            best_thicknesses = population[min_idx][1][:]
            
        # Selection: Tournament selection
        def tournament_select(pop, scores, k=3):
            indices = np.random.randint(0, len(pop), size=k)
            best_idx = indices[np.argmin([scores[i] for i in indices])]
            return pop[best_idx]
        
        new_population = []
        
        # Elitism: Keep top 2
        sorted_indices = np.argsort(scores)
        for i in range(2):
            new_population.append((population[sorted_indices[i]][0][:], population[sorted_indices[i]][1][:]))
        
        # Generate rest
        while len(new_population) < pop_size:
            if time.time() > deadline:
                break
                
            # Crossover
            parent1 = tournament_select(population, scores)
            parent2 = tournament_select(population, scores)
            
            child_mats = []
            child_ths = []
            
            # Determine child length: random mix or one of parents
            # We need to handle different lengths. Simple approach:
            # Take a random prefix length from one parent, or interpolate.
            # Let's try: random length between min(len1, len2) and max(len1, len2)
            len1 = len(parent1[0])
            len2 = len(parent2[0])
            min_len = min(len1, len2)
            max_len = max(len1, len2)
            
            if np.random.random() < 0.1:
                # Random length mutation
                target_len = np.random.randint(1, max_layers + 1)
            else:
                target_len = np.random.randint(min_len, max_len + 1)
                
            for i in range(target_len):
                # Choose material and thickness from parents if available, else random
                if i < len1 and np.random.random() < 0.5:
                    mat = parent1[0][i]
                    th = parent1[1][i]
                elif i < len2:
                    mat = parent2[0][i]
                    th = parent2[1][i]
                else:
                    mat = 'H' if np.random.random() < 0.5 else 'L'
                    th = (qw_H if mat == 'H' else qw_L) * (0.5 + np.random.random())
                    
                # Mutation
                if np.random.random() < 0.1: # Material mutation
                    mat = 'L' if mat == 'H' else 'H'
                    
                if np.random.random() < 0.3: # Thickness mutation
                    # Gaussian noise
                    th += np.random.normal(0, 50)
                    th = max(0, min(th, max_layer_nm))
                    
                child_mats.append(mat)
                child_ths.append(th)
            
            new_population.append((child_mats, child_ths))
        
        population = new_population
    
    # Final check and return
    if best_materials is None:
        return ["L"], [qw_L]
        
    # Ensure final validity
    if not is_valid(best_materials, best_thicknesses):
        # Fallback
        return ["L"], [qw_L]
        
    return best_materials, best_thicknesses
# EVOLVE-BLOCK-END
