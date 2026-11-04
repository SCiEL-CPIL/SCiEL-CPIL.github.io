import numpy as np
from scipy.optimize import minimize
import time
import itertools

N = 11

# Vertices of the equilateral triangle in Cartesian
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
TRI_AREA = 0.5 * np.sqrt(3) / 2  # Area of equilateral triangle with side 1

# Precompute indices for all triples
_IDX_TRIPLES = np.array([(i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)])
n_triples = len(_IDX_TRIPLES)  # 165

def _all_dets_from_bary(bary):
    """
    Compute all determinants (twice the area in barycentric coords).
    bary: (N, 2) array of [u, v]. w = 1 - u - v.
    """
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    P = np.column_stack((u, v, w))
    
    i_idx = _IDX_TRIPLES[:, 0]
    j_idx = _IDX_TRIPLES[:, 1]
    k_idx = _IDX_TRIPLES[:, 2]
    
    Pi = P[i_idx]
    Pj = P[j_idx]
    Pk = P[k_idx]
    
    # det = u_i(v_j w_k - w_j v_k) - v_i(u_j w_k - w_j u_k) + w_i(u_j v_k - v_j u_k)
    det = (Pi[:, 0] * (Pj[:, 1] * Pk[:, 2] - Pj[:, 2] * Pk[:, 1])
           - Pi[:, 1] * (Pj[:, 0] * Pk[:, 2] - Pj[:, 2] * Pk[:, 0])
           + Pi[:, 2] * (Pj[:, 0] * Pk[:, 1] - Pj[:, 1] * Pk[:, 0]))
           
    return det

def _min_area_ratio_from_bary(bary):
    """
    Returns the minimum absolute determinant (proportional to area ratio).
    """
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    
    if np.any(u < -1e-9) or np.any(v < -1e-9) or np.any(w < -1e-9):
        return -1.0
        
    dets = _all_dets_from_bary(bary)
    return np.min(np.abs(dets))

def _bary_to_cart(u, v):
    """
    Convert barycentric coordinates (u, v) to Cartesian.
    w = 1 - u - v
    Point = u*V0 + v*V1 + w*V2
    """
    w = 1.0 - u - v
    x = v * 1.0 + w * 0.5
    y = w * np.sqrt(3) / 2
    return np.column_stack((x, y))

def _project_to_bary_simplex(bary):
    """
    Project points to the valid barycentric simplex.
    """
    u = np.maximum(0, bary[:, 0])
    v = np.maximum(0, bary[:, 1])
    s = u + v
    mask = s > 1.0
    if np.any(mask):
        u[mask] /= s[mask]
        v[mask] /= s[mask]
    return np.column_stack((u, v))

def _objective_smooth(x, beta):
    """
    Objective: Minimize -softmin(areas).
    x: (2N,) array.
    """
    bary = x.reshape(N, 2)
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    
    # Penalty for invalid barycentric coords
    penalty = 0.0
    excess = np.maximum(0, u + v - 1.0)
    penalty += 100.0 * np.sum(excess**2)
    excess_u = np.maximum(0, -u)
    penalty += 100.0 * np.sum(excess_u**2)
    excess_v = np.maximum(0, -v)
    penalty += 100.0 * np.sum(excess_v**2)
    
    dets = _all_dets_from_bary(bary)
    areas = np.abs(dets)
    
    if len(areas) == 0:
        return penalty, np.zeros_like(x)
        
    # Soft min: m - (1/beta) * log(sum(exp(-beta * (areas - m))))
    m = np.min(areas)
    shifted = areas - m
    # Numerical stability
    max_exp = np.max(-beta * shifted)
    log_sum_exp = max_exp + np.log(np.sum(np.exp(-beta * shifted - max_exp)))
    
    smooth_min = m - (1.0 / beta) * log_sum_exp
    
    val = -smooth_min + penalty
    
    return val, None # None indicates numerical gradient

def _objective_exact(x):
    """
    Objective: Minimize -min_area.
    """
    bary = x.reshape(N, 2)
    u = np.clip(bary[:, 0], 0, None)
    v = np.clip(bary[:, 1], 0, None)
    s = u + v
    mask = s > 1.0
    if np.any(mask):
        u[mask] /= s[mask]
        v[mask] /= s[mask]
    
    bary_clamped = np.column_stack((u, v))
    dets = _all_dets_from_bary(bary_clamped)
    min_area = np.min(np.abs(dets))
    return -min_area

def _structured_init_fps():
    """
    Farthest Point Sampling on a grid of barycentric coordinates.
    """
    n_grid = 30
    candidates = []
    for i in range(n_grid + 1):
        for j in range(n_grid + 1 - i):
            u = i / n_grid
            v = j / n_grid
            candidates.append([u, v])
    candidates = np.array(candidates)
    
    selected = [np.array([1/3, 1/3])]
    
    for _ in range(N - 1):
        dists = []
        for c in candidates:
            min_d = np.min(np.linalg.norm(c - np.array(selected), axis=1))
            dists.append(min_d)
        dists = np.array(dists)
        max_idx = np.argmax(dists)
        selected.append(candidates[max_idx])
        
    return np.array(selected)

def _ga_fitness(individual):
    """
    Fitness function for the Genetic Algorithm.
    Returns the minimum area ratio.
    """
    bary = individual.reshape(N, 2)
    # Ensure validity
    u = np.clip(bary[:, 0], 0, 1)
    v = np.clip(bary[:, 1], 0, 1)
    s = u + v
    mask = s > 1.0
    if np.any(mask):
        u[mask] /= s[mask]
        v[mask] /= s[mask]
    
    bary_valid = np.column_stack((u, v))
    return _min_area_ratio_from_bary(bary_valid)

def _ga_evolve(population, fitnesses, n_offspring=20, crossover_rate=0.8, mutation_rate=0.1, mutation_scale=0.1):
    """
    Simple Genetic Algorithm step.
    """
    n_pop = len(population)
    # Selection: Tournament selection
    selected_indices = []
    for _ in range(n_offspring * 2):
        # Tournament of size 3
        tournament = np.random.choice(n_pop, size=3, replace=False)
        best_in_tournament = tournament[np.argmax(fitnesses[tournament])]
        selected_indices.append(best_in_tournament)
    
    offspring = []
    for i in range(0, n_offspring * 2, 2):
        parent1 = population[selected_indices[i]]
        parent2 = population[selected_indices[i+1]]
        
        if np.random.random() < crossover_rate:
            # Blend crossover
            alpha = np.random.random(N * 2)
            child1 = alpha * parent1 + (1 - alpha) * parent2
            child2 = (1 - alpha) * parent1 + alpha * parent2
        else:
            child1 = parent1.copy()
            child2 = parent2.copy()
        
        # Mutation
        if np.random.random() < mutation_rate:
            mask = np.random.random(N * 2) < mutation_rate
            child1[mask] += np.random.normal(0, mutation_scale, size=np.sum(mask))
        if np.random.random() < mutation_rate:
            mask = np.random.random(N * 2) < mutation_rate
            child2[mask] += np.random.normal(0, mutation_scale, size=np.sum(mask))
        
        # Clip to simplex bounds [0, 1] roughly, will be projected later
        child1 = np.clip(child1, 0, 1)
        child2 = np.clip(child2, 0, 1)
        
        offspring.append(child1)
        offspring.append(child2)
        
    return np.array(offspring)

def _local_optimize_slsqp(x0):
    """
    Optimize a single configuration using SLSQP.
    x0: (N, 2) barycentric coordinates.
    """
    # Flatten for SLSQP
    x_flat = x0.flatten()
    
    # Constraints: u_i >= 0, v_i >= 0, u_i + v_i <= 1
    # We can use bounds and equality/inequality constraints.
    # SLSQP handles bounds well.
    bounds = [(0, 1)] * (2 * N)
    
    # Inequality constraints: u_i + v_i <= 1 => 1 - u_i - v_i >= 0
    # scipy minimize uses g(x) >= 0 for inequality constraints
    cons = []
    for i in range(N):
        def make_constraint(i):
            def constraint(x):
                return 1.0 - x[2*i] - x[2*i+1]
            return constraint
        cons.append({'type': 'ineq', 'fun': make_constraint(i)})
        
    try:
        res = minimize(
            _objective_exact,
            x_flat,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 100, 'ftol': 1e-8}
        )
        if res.success:
            return res.x.reshape(N, 2)
    except Exception:
        pass
    
    return x0

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    def time_left():
        return deadline - time.time()
        
    best_x = None
    best_score = -float('inf')
    
    # 1. Initial Population for GA
    # Mix of FPS, Random, and Structured starts
    rng = np.random.RandomState(42)
    
    pop_size = 50
    population = []
    
    # Add FPS start
    fps_init = _structured_init_fps().flatten()
    population.append(fps_init)
    
    # Add Random starts
    for _ in range(pop_size - 1):
        pts = np.zeros((N, 2))
        for i in range(N):
            u = rng.random()
            v = rng.random()
            while u + v > 1.0:
                u = rng.random()
                v = rng.random()
            pts[i, 0] = u
            pts[i, 1] = v
        population.append(pts.flatten())
        
    population = np.array(population)
    population = _project_to_bary_simplex(population.reshape(-1, N, 2)).reshape(-1, N * 2)
    
    # Evaluate initial population
    fitnesses = np.array([_ga_fitness(ind) for ind in population])
    
    # 2. Genetic Algorithm Loop
    ga_generations = 20
    for gen in range(ga_generations):
        if time_left() < 10.0:
            break
            
        # Select best so far
        best_idx = np.argmax(fitnesses)
        if fitnesses[best_idx] > best_score:
            best_score = fitnesses[best_idx]
            best_x = population[best_idx].copy()
            
        # Evolve
        offspring = _ga_evolve(population, fitnesses, n_offspring=pop_size // 2)
        
        # Evaluate offspring
        offspring_fitnesses = np.array([_ga_fitness(ind) for ind in offspring])
        
        # Combine and select new population
        combined_pop = np.vstack((population, offspring))
        combined_fit = np.concatenate((fitnesses, offspring_fitnesses))
        
        # Keep top half
        sort_idx = np.argsort(combined_fit)[::-1]
        population = combined_pop[sort_idx[:pop_size]]
        fitnesses = combined_fit[sort_idx[:pop_size]]
        
        # Elitism: Ensure the best individual is always kept
        # (Already handled by keeping top half, but explicit check is safer)
        global_best_idx = np.argmax(combined_fit)
        if population[0] is not combined_pop[global_best_idx]: # Simple check
             population[0] = combined_pop[global_best_idx]
             fitnesses[0] = combined_fit[global_best_idx]
             
    # 3. Local Optimization of Top Candidates from GA
    # Take the top 10 candidates from the GA and refine them with SLSQP
    top_indices = np.argsort(fitnesses)[::-1][:10]
    
    for idx in top_indices:
        if time_left() < 5.0:
            break
            
        x0 = population[idx].reshape(N, 2)
        x_opt = _local_optimize_slsqp(x0)
        
        # Ensure validity
        x_opt = _project_to_bary_simplex(x_opt)
        
        score_opt = _min_area_ratio_from_bary(x_opt)
        
        if score_opt > best_score:
            best_score = score_opt
            best_x = x_opt.flatten()
            
    # 4. Final Polish with Nelder-Mead on the very best
    if best_x is not None:
        if time_left() > 2.0:
            try:
                res = minimize(
                    _objective_exact,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 50, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
                )
                x_final = res.x
                x_final = _project_to_bary_simplex(x_final.reshape(N, 2)).flatten()
                score_final = _min_area_ratio_from_bary(x_final.reshape(N, 2))
                
                if score_final > best_score:
                    best_score = score_final
                    best_x = x_final
            except Exception:
                pass

    # If we found a good solution, convert to Cartesian
    if best_x is not None:
        bary_best = best_x.reshape(N, 2)
        pts_best = _bary_to_cart(bary_best[:, 0], best_x[:, 1])
        return pts_best
    
    # Fallback: return random valid points
    pts = []
    for _ in range(N):
        u, v = np.random.random(2)
        while u + v > 1:
            u, v = np.random.random(2)
        pts.append([u + 0.5*v, np.sqrt(3)/2*v])
    return np.array(pts)

def score(points):
    """Compute the raw score: min triangle area / TRI_AREA."""
    bary = np.zeros((N, 2))
    for i in range(N):
        p = points[i]
        w = 2 * p[1] / np.sqrt(3)
        v = p[0] - w * 0.5
        u = 1 - v - w
        bary[i] = [u, v]
    
    dets = _all_dets_from_bary(bary)
    min_area = np.min(np.abs(dets))
    return min_area / (2 * TRI_AREA)

# Ensure the function name is construct as per the task
# The task says: Interface: construct() -> array of shape (n, 2)
# The current program has construct() and score(). I will keep both.
