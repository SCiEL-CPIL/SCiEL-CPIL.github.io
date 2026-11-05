import time
import math
import numpy as np
import random

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
    beta = np.zeros((n_layers, n_wls))
    for i, m in enumerate(materials):
        n = n_mat[m]
        d = thicknesses[i]
        if d == 0:
            continue
        beta[i, :] = 2.0 * math.pi * n * d / wls

    cosb = np.cos(beta)
    sinb = np.sin(beta)

    # Build the total characteristic matrix
    M11 = np.ones(n_wls, dtype=complex)
    M12 = np.zeros(n_wls, dtype=complex)
    M21 = np.zeros(n_wls, dtype=complex)
    M22 = np.ones(n_wls, dtype=complex)

    for i, m in enumerate(materials):
        n = n_mat[m]
        c = cosb[i]
        s = sinb[i]
        
        # Matrix elements for layer i
        A = c
        B = 1j * s / n
        C = 1j * n * s
        D = c

        # Matrix multiplication: M_new = M_old * M_i
        nM11 = M11 * A + M12 * C
        nM12 = M11 * B + M12 * D
        nM21 = T21 * A + M22 * C
        nM22 = M21 * B + M22 * D
        
        M11, M12, M21, M22 = nM11, nM12, nM21, nM22

    # Admittance Y
    Y = (M21 + n_substrate * M22) / (M11 + n_substrate * M12)
    
    # Reflectance r
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
    
    for t in thicknesses:
        if t < -1e-9 or t > max_layer + 1e-9:
            return 1e6
            
    total_th = sum(thicknesses)
    if total_th > max_total + 1e-6:
        return 1e6 + (total_th - max_total) * 10.0

    if len(materials) == 0:
        r = (problem["ambient_index"] - problem["substrate_index"]) / (
            problem["ambient_index"] + problem["substrate_index"])
        return r * r * 100.0

    R = _reflectance_array(materials, thicknesses, problem["ambient_index"],
                           problem["substrate_index"], wls, n_mat)
    return float(np.mean(R)) * 100.0

def _nelder_mead(objective, x0, bounds, deadline, max_iter=200, tol=1e-8):
    """
    Fast Nelder-Mead optimization with bounds handling.
    """
    x0 = np.array(x0, dtype=float)
    n = len(x0)
    lo, hi = bounds
    
    # Initialize simplex
    simplex = [x0.copy()]
    for i in range(n):
        xi = x0.copy()
        range_i = hi[i] - lo[i]
        step = 0.05 * range_i if range_i > 1e-6 else 0.1
        xi[i] += step
        simplex.append(np.clip(xi, lo, hi))
        
    f_vals = [objective(x) for x in simplex]
    best_f = min(f_vals)
    
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
        if abs(f_vals[0] - f_vals[-1]) < tol:
            break
            
        # Centroid of best n points
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        x_r = np.clip(centroid + alpha * (centroid - simplex[-1]), lo, hi)
        f_r = objective(x_r)
        
        if f_vals[0] <= f_r < f_vals[-2]:
            simplex[-1] = x_r
            f_vals[-1] = f_r
            if f_r < best_f:
                best_f = f_r
            continue
            
        # Expansion
        if f_r < f_vals[0]:
            x_e = np.clip(centroid + gamma * (x_r - centroid), lo, hi)
            f_e = objective(x_e)
            if f_e < f_r:
                simplex[-1] = x_e
                f_vals[-1] = f_e
                if f_e < best_f:
                    best_f = f_e
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
            if f_c < best_f:
                best_f = f_c
            continue
            
        # Shrink
        for i in range(1, n + 1):
            simplex[i] = np.clip(simplex[0] + sigma * (simplex[i] - simplex[0]), lo, hi)
            f_vals[i] = objective(simplex[i])
        if min(f_vals[1:]) < best_f:
            best_f = min(f_vals[1:])

    order = np.argsort(f_vals)
    best_idx = order[0]
    return np.clip(simplex[best_idx], lo, hi), f_vals[best_idx]

def design(problem):
    start_time = time.time()
    # Safety margin: Evaluator kills at 150s. We aim to finish by 145s.
    deadline = start_time + 145.0
    
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
    
    # Population parameters
    pop_size = 100
    elite_size = 10
    mutation_rate = 0.2
    crossover_rate = 0.8
    
    # Initialize population
    population = []
    for _ in range(pop_size):
        # Random number of layers between 4 and max_layers
        n_layers = random.randint(4, max_layers)
        # Ensure total thickness can fit
        # Average layer thickness ~ max_total / max_layers
        # Let's pick random thicknesses and scale if needed
        mats = [random.choice(['L', 'H']) for _ in range(n_layers)]
        ths = [random.uniform(20, max_layer) for _ in range(n_layers)]
        
        # Scale to fit max_total if necessary
        total = sum(ths)
        if total > max_total:
            scale = max_total / total
            ths = [t * scale for t in ths]
            
        # Clip to max_layer
        ths = [min(t, max_layer) for t in ths]
        
        population.append((mats, ths))
        
    # Evaluate initial population
    best_val = float('inf')
    best_mats = None
    best_ths = None
    
    for i, (mats, ths) in enumerate(population):
        val = _mean_reflectance_percent(mats, ths, problem, wls, n_mat)
        population[i] = (mats, ths, val)
        if val < best_val:
            best_val = val
            best_mats = mats
            best_ths = ths
            
    # GA Loop
    generation = 0
    while time.time() < deadline:
        generation += 1
        
        # Sort by fitness
        population.sort(key=lambda x: x[2])
        
        # Keep best
        if population[0][2] < best_val:
            best_val = population[0][2]
            best_mats = population[0][0]
            best_ths = population[0][1]
            
        # Elitism: Keep top elite_size
        new_population = []
        for i in range(min(elite_size, len(population))):
            new_population.append(population[i])
            
        # Fill rest of population with offspring
        while len(new_population) < pop_size:
            if time.time() > deadline:
                break
                
            # Tournament selection
            p1_idx = random.randint(0, len(population) - 1)
            p2_idx = random.randint(0, len(population) - 1)
            if population[p1_idx][2] < population[p2_idx][2]:
                p1 = population[p1_idx]
            else:
                p1 = population[p2_idx]
                
            if random.random() < crossover_rate:
                # Select second parent
                p2_idx = random.randint(0, len(population) - 1)
                if population[p2_idx][2] < population[p1_idx][2]:
                    p2 = population[p2_idx]
                else:
                    p2 = population[p1_idx]
                    
                # Crossover
                mats1, ths1, _ = p1
                mats2, ths2, _ = p2
                
                # If lengths are different, truncate/pad or just pick one
                if len(mats1) != len(mats2):
                    # Simple strategy: if lengths differ, just mutate p1
                    child_mats = mats1[:]
                    child_ths = ths1[:]
                else:
                    # Point crossover
                    cut = random.randint(1, len(mats1) - 1)
                    child_mats = mats1[:cut] + mats2[cut:]
                    child_ths = ths1[:cut] + ths2[cut:]
            else:
                child_mats = p1[0][:]
                child_ths = p1[1][:]
                
            # Mutation
            if random.random() < mutation_rate:
                # Material mutation
                if child_mats:
                    idx = random.randint(0, len(child_mats) - 1)
                    child_mats[idx] = 'L' if child_mats[idx] == 'H' else 'H'
                    
                # Thickness mutation
                for i in range(len(child_ths)):
                    if random.random() < 0.5:
                        noise = random.gauss(0, 20)
                        child_ths[i] += noise
                        child_ths[i] = max(0, min(child_ths[i], max_layer))
                        
            # Length mutation (insert/delete)
            if random.random() < 0.1:
                if len(child_mats) < max_layers:
                    # Insert
                    idx = random.randint(0, len(child_mats))
                    new_mat = random.choice(['L', 'H'])
                    # Estimate thickness based on neighbors or random
                    new_th = random.uniform(20, 100)
                    child_mats.insert(idx, new_mat)
                    child_ths.insert(idx, new_th)
                elif len(child_mats) > 2:
                    # Delete
                    idx = random.randint(0, len(child_mats) - 1)
                    child_mats.pop(idx)
                    child_ths.pop(idx)
                    
            # Constraint handling: Scale thicknesses if total > max_total
            total_th = sum(child_ths)
            if total_th > max_total:
                scale = max_total / total_th
                child_ths = [t * scale for t in child_ths]
                
            # Final clip
            child_ths = [max(0, min(t, max_layer)) for t in child_ths]
            
            # Evaluate child
            val = _mean_reflectance_percent(child_mats, child_ths, problem, wls, n_mat)
            new_population.append((child_mats, child_ths, val))
            
        # Update population
        population = new_population
        
        # Local Polish: Every 10 generations, polish the best individual
        if generation % 10 == 0 and time.time() < deadline - 5.0:
            # Take the best individual and run Nelder-Mead on its thicknesses
            best_ind = min(population, key=lambda x: x[2])
            b_mats, b_ths, b_val = best_ind
            
            if len(b_mats) > 0:
                lo = np.zeros(len(b_mats))
                hi = np.full(len(b_mats), max_layer)
                
                def objective(th):
                    return _mean_reflectance_percent(b_mats, list(th), problem, wls, n_mat)
                
                # Run NM with a tighter deadline
                nm_deadline = min(deadline - 1.0, time.time() + 5.0)
                x_opt, f_opt = _nelder_mead(objective, np.array(b_ths), (lo, hi), nm_deadline, max_iter=100)
                
                # Check if improved
                if f_opt < b_val:
                    # Update population entry
                    # Find index of best_ind in population
                    for i in range(len(population)):
                        if population[i] == best_ind:
                            population[i] = (b_mats, list(x_opt), f_opt)
                            if f_opt < best_val:
                                best_val = f_opt
                                best_mats = b_mats
                                best_ths = list(x_opt)
                            break
                            
    # Final constraint check and scaling
    if best_mats is None:
        # Fallback
        best_mats = ['L']
        best_ths = [600.0 / (4.0 * n_L)]
        best_val = _mean_reflectance_percent(best_mats, best_ths, problem, wls, n_mat)
        
    best_ths = [min(max(t, 0.0), max_layer) for t in best_ths]
    if sum(best_ths) > max_total + 1e-6:
        f = max_total / sum(best_ths)
        best_ths = [t * f for t in best_ths]
        
    return list(best_mats), list(best_ths)
