import time
import math
import numpy as np

def _reflectance_array(materials, thicknesses, n_ambient, n_substrate, wls, n_mat):
    """
    Compute reflectance array for given materials and thicknesses using
    the thin-film characteristic-matrix method.
    Returns array of reflectance values (0 to 1).
    """
    if len(materials) == 0:
        r = (n_ambient - n_substrate) / (n_ambient + n_substrate)
        return np.full(len(wls), r * r)

    n_layers = len(materials)
    n_wls = len(wls)
    
    # Precompute beta = 2*pi*n*d/wl for all layers and wavelengths
    # beta shape: (n_layers, n_wls)
    beta = np.zeros((n_layers, n_wls))
    for i, m in enumerate(materials):
        n = n_mat[m]
        d = thicknesses[i]
        if d != 0:
            beta[i, :] = 2.0 * math.pi * n * d / wls

    cosb = np.cos(beta)
    sinb = np.sin(beta)

    # Build the total characteristic matrix
    # M = M_1 * M_2 * ... * M_N
    # Each M_i = [[cos(beta_i), i*sin(beta_i)/n_i], [i*n_i*sin(beta_i), cos(beta_i)]]
    
    M11 = np.ones(n_wls, dtype=complex)
    M12 = np.zeros(n_wls, dtype=complex)
    M21 = np.zeros(n_wls, dtype=complex)
    M22 = np.ones(n_wls, dtype=complex)

    for i, m in enumerate(materials):
        n = n_mat[m]
        c = cosb[i]
        s = sinb[i]
        
        # Matrix elements for layer i
        # A = cos(beta), B = i*sin(beta)/n, C = i*n*sin(beta), D = cos(beta)
        A = c
        B = 1j * s / n
        C = 1j * n * s
        D = c

        # Matrix multiplication: M_new = M_old * M_i
        # M_old is [[M11, M12], [M21, M22]]
        
        nM11 = M11 * A + M12 * C
        nM12 = M11 * B + M12 * D
        nM21 = M21 * A + M22 * C
        nM22 = M21 * B + M22 * D
        
        M11, M12, M21, M22 = nM11, nM12, nM21, nM22

    # Admittance Y
    # Y = (M21 + n_substrate * M22) / (M11 + n_substrate * M12)
    Y = (M21 + n_substrate * M22) / (M11 + n_substrate * M12)
    
    # Reflectance r
    # r = (n_ambient - Y) / (n_ambient + Y)
    r = (n_ambient - Y) / (n_ambient + Y)
    
    # Reflectance R = |r|^2
    R = np.abs(r) ** 2
    return R

def _mean_reflectance_percent(materials, thicknesses, problem, wls, n_mat):
    """
    Compute mean reflectance in percent.
    Returns penalty if constraints are violated.
    """
    if len(materials) != len(thicknesses):
        return 1e6
    if len(materials) > problem["max_layers"]:
        return 1e6
        
    max_layer = problem["max_layer_nm"]
    max_total = problem["max_total_nm"]
    
    # Check individual thickness bounds
    for t in thicknesses:
        if t < -1e-9 or t > max_layer + 1e-9:
            return 1e6
            
    total_th = sum(thicknesses)
    if total_th > max_total + 1e-6:
        # Heavy penalty for total thickness violation
        return 1e6 + (total_th - max_total) * 100.0

    if len(materials) == 0:
        r = (problem["ambient_index"] - problem["substrate_index"]) / (
            problem["ambient_index"] + problem["substrate_index"])
        return r * r * 100.0

    R = _reflectance_array(materials, thicknesses, problem["ambient_index"],
                           problem["substrate_index"], wls, n_mat)
    return float(np.mean(R)) * 100.0

def _genetic_algorithm(materials, problem, wls, n_mat, deadline, pop_size=50, max_gens=100):
    """
    Genetic algorithm to optimize layer thicknesses.
    """
    n = len(materials)
    max_layer = problem["max_layer_nm"]
    max_total = problem["max_total_nm"]
    
    # Initial population
    np.random.seed(42)
    population = []
    
    # Add some structured initial guesses
    lambda_gm = math.sqrt(problem["band_nm"][0] * problem["band_nm"][1])
    lambda_am = (problem["band_nm"][0] + problem["band_nm"][1]) / 2.0
    
    # Quarter-wave at geometric mean
    x_gm = [lambda_gm / (4.0 * n_mat[m]) for m in materials]
    total_gm = sum(x_gm)
    if total_gm > max_total:
        x_gm = [d * max_total / total_gm for d in x_gm]
    population.append(np.array(x_gm))
    
    # Quarter-wave at arithmetic mean
    x_am = [lambda_am / (4.0 * n_mat[m]) for m in materials]
    total_am = sum(x_am)
    if total_am > max_total:
        x_am = [d * max_total / total_am for d in x_am]
    population.append(np.array(x_am))
    
    # Random individuals
    for _ in range(pop_size - 2):
        x = np.random.uniform(0, max_layer, size=n)
        total = np.sum(x)
        if total > max_total:
            x = x * max_total / total
        population.append(x)
    
    def evaluate(x):
        x_clipped = np.clip(x, 0, max_layer)
        total = np.sum(x_clipped)
        if total > max_total:
            return 1e6 + (total - max_total) * 10.0
        return _mean_reflectance_percent(materials, list(x_clipped), problem, wls, n_mat)
    
    fitness = [evaluate(ind) for ind in population]
    best_idx = np.argmin(fitness)
    best_x = population[best_idx].copy()
    best_f = fitness[best_idx]
    
    for gen in range(max_gens):
        if time.time() > deadline - 15.0:
            break
        
        # Sort by fitness
        order = np.argsort(fitness)
        population = [population[i] for i in order]
        fitness = [fitness[i] for i in order]
        
        best_x = population[0].copy()
        best_f = fitness[0]
        
        # Elitism: keep top 2
        new_population = [population[0].copy(), population[1].copy()]
        new_fitness = [fitness[0], fitness[1]]
        
        # Generate offspring
        while len(new_population) < pop_size:
            if time.time() > deadline - 15.0:
                break
            
            # Tournament selection
            def select():
                idx = np.random.randint(0, pop_size, size=3)
                best = min(idx, key=lambda i: fitness[i])
                return population[best].copy()
            
            parent1 = select()
            parent2 = select()
            
            # Crossover: blend crossover
            alpha = np.random.uniform(0.2, 0.8)
            child = alpha * parent1 + (1 - alpha) * parent2
            
            # Mutation: Gaussian perturbation
            mutation_rate = 0.3
            for i in range(n):
                if np.random.random() < mutation_rate:
                    sigma = max_layer * 0.1
                    child[i] += np.random.normal(0, sigma)
            
            # Constraint handling: clip and scale
            child = np.clip(child, 0, max_layer)
            total = np.sum(child)
            if total > max_total:
                child = child * max_total / total
            
            f_child = evaluate(child)
            new_population.append(child)
            new_fitness.append(f_child)
        
        population = new_population
        fitness = new_fitness
        
        # Check convergence
        if fitness[0] - fitness[-1] < 1e-8:
            break
    
    return np.clip(best_x, 0, max_layer), best_f

def _nelder_mead_polish(materials, x0, problem, wls, n_mat, deadline, max_iter=200):
    """
    Nelder-Mead optimization for thicknesses of a fixed material sequence.
    """
    n = len(x0)
    lo = np.zeros(n)
    hi = np.full(n, problem["max_layer_nm"])
    max_total = problem["max_total_nm"]
    
    # Ensure initial guess is valid
    x = np.array(x0, dtype=float)
    x = np.clip(x, lo, hi)
    total = np.sum(x)
    if total > max_total:
        x = x * (max_total / total)
        x = np.clip(x, lo, hi)
        
    def objective(x):
        # Check total thickness constraint strictly during optimization
        if np.sum(x) > max_total:
            # Penalize proportionally to help the optimizer back off
            return 1e6 + (np.sum(x) - max_total) * 10.0
        return _mean_reflectance_percent(materials, list(x), problem, wls, n_mat)

    # Initialize simplex
    simplex = [x.copy()]
    for i in range(n):
        xi = x.copy()
        # Step size: 10% of range or 50nm, whichever is smaller
        step = min(50.0, 0.1 * (hi[i] - lo[i]))
        if step < 1.0: step = 1.0
        xi[i] += step
        simplex.append(np.clip(xi, lo, hi))
        
    f_vals = [objective(x) for x in simplex]
    
    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    sigma = 0.5
    
    for iter_count in range(max_iter):
        if time.time() > deadline:
            break
            
        # Sort simplex
        order = np.argsort(f_vals)
        simplex = [simplex[i] for i in order]
        f_vals = [f_vals[i] for i in order]
        
        # Check convergence
        spread = abs(f_vals[0] - f_vals[-1])
        if spread < 1e-6:
            break
        
        # Centroid of best n points
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        x_r = np.clip(centroid + alpha * (centroid - simplex[-1]), lo, hi)
        f_r = objective(x_r)
        
        if f_vals[0] <= f_r < f_vals[-2]:
            simplex[-1] = x_r
            f_vals[-1] = f_r
            continue
            
        # Expansion
        if f_r < f_vals[0]:
            x_e = np.clip(centroid + gamma * (x_r - centroid), lo, hi)
            f_e = objective(x_e)
            if f_e < f_r:
                simplex[-1] = x_e
                f_vals[-1] = f_e
            else:
                simplex[-1] = x_r
                f_vals[-1] = f_r
            continue
            
        # Contraction
        x_c = np.clip(centroid + rho * (simplex[-1] - centroid), lo, hi)
        f_c = objective(x_c)
        if f_c < f_vals[-1]:
            simplex[-1] = x_c
            f_vals[-1] = f_c
            continue
            
        # Shrink
        for i in range(1, n + 1):
            simplex[i] = np.clip(simplex[0] + sigma * (simplex[i] - simplex[0]), lo, hi)
            f_vals[i] = objective(simplex[i])
            
    order = np.argsort(f_vals)
    best_idx = order[0]
    return np.clip(simplex[best_idx], lo, hi), f_vals[best_idx]

def design(problem):
    """
    Design a broadband antireflection coating.
    """
    start_time = time.time()
    # 150s wall-clock limit. Keep a safety margin of 10s.
    deadline = start_time + 140.0
    
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    n_ambient = problem["ambient_index"]
    n_substrate = problem["substrate_index"]
    max_layers = problem["max_layers"]
    max_total = problem["max_total_nm"]
    max_layer = problem["max_layer_nm"]
    
    band_start, band_end = problem["band_nm"]
    wls = np.linspace(band_start, band_end, 401)
    n_mat = problem["materials"]
    
    best_val = float('inf')
    best_mats = None
    best_ths = None
    
    # Generate candidate material sequences
    # Focus on alternating L-H starting with L (good for air-glass transition)
    sequences = []
    for length in [4, 6, 8, 10, 12, 14, 16, 18]:
        if length > max_layers:
            break
        # Start with L, alternate
        seq = []
        for i in range(length):
            if i % 2 == 0:
                seq.append('L')
            else:
                seq.append('H')
        sequences.append(seq)
        
        # Also try starting with H if length is even? 
        # Usually L first is better for AR, but let's include H-start for even lengths >= 4
        if length >= 4:
            seq_h = []
            for i in range(length):
                if i % 2 == 0:
                    seq_h.append('H')
                else:
                    seq_h.append('L')
            sequences.append(seq_h)

    # Use genetic algorithm for global search
    for mats in sequences:
        if time.time() > deadline - 30.0:  # Leave time for polish
            break
            
        n_layers = len(mats)
        
        # Run genetic algorithm
        x_opt, f_opt = _genetic_algorithm(mats, problem, wls, n_mat, deadline, pop_size=40, max_gens=80)
        
        # Post-process: Ensure constraints
        x_opt = np.clip(x_opt, 0.0, max_layer)
        total_opt = sum(x_opt)
        if total_opt > max_total:
            factor = max_total / total_opt
            x_opt = x_opt * factor
            x_opt = np.clip(x_opt, 0.0, max_layer)
        
        # Calculate final score
        val = _mean_reflectance_percent(mats, list(x_opt), problem, wls, n_mat)
        
        if val < best_val:
            best_val = val
            best_mats = mats
            best_ths = list(x_opt)
    
    # Final high-precision polish pass if time permits
    if best_mats is not None and time.time() < deadline - 10.0:
        n_layers = len(best_mats)
        
        # Run a final polish with more iterations and tighter tolerance
        x_polish, f_polish = _nelder_mead_polish(best_mats, best_ths, problem, wls, n_mat, deadline, max_iter=300)
        
        # Post-process
        x_polish = np.clip(x_polish, 0.0, max_layer)
        total_polish = sum(x_polish)
        if total_polish > max_total:
            factor = max_total / total_polish
            x_polish = x_polish * factor
            x_polish = np.clip(x_polish, 0.0, max_layer)
            
        val_polish = _mean_reflectance_percent(best_mats, list(x_polish), problem, wls, n_mat)
        
        if val_polish < best_val:
            best_val = val_polish
            best_ths = list(x_polish)
            
    if best_mats is None:
        # Fallback: single layer quarter-wave at center
        best_mats = ['L']
        best_ths = [600.0 / (4.0 * n_L)]
        best_val = _mean_reflectance_percent(best_mats, best_ths, problem, wls, n_mat)
        
    # Final constraint check and scaling
    best_ths = [min(max(t, 0.0), max_layer) for t in best_ths]
    if sum(best_ths) > max_total + 1e-6:
        f = max_total / sum(best_ths)
        best_ths = [t * f for t in best_ths]
        
    return list(best_mats), list(best_ths)
