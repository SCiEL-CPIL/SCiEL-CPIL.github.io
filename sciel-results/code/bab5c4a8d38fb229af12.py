import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 14
D = 3
DEADLINE = 140.0
_t0 = None

def _true_score(pts):
    """
    Calculate the squared ratio of max to min pairwise distance.
    """
    d = cdist(pts, pts)
    np.fill_diagonal(d, np.inf)
    mn = d.min()
    mx = d.max()
    if mn < 1e-9:
        return np.inf
    return (mx / mn) ** 2

def _smooth_ratio_obj(v, k=1.0):
    """
    Smooth approximation of the ratio objective.
    We want to minimize (max_d / min_d)^2.
    Let d2 be squared distances.
    max_d^2 = soft_max(d2, k)
    min_d^2 = soft_min(d2, k)
    ratio_sq = soft_max(d2, k) / soft_min(d2, k)
    
    To make it differentiable and avoid division by zero, we add a small epsilon.
    Also, we can minimize log(ratio) = log(soft_max) - log(soft_min).
    This is equivalent to minimizing the ratio and better conditioned.
    
    v: flattened array of N*D coordinates.
    """
    x = v.reshape(N, D)
    
    # Compute squared distances
    x_sq = np.sum(x**2, axis=1)
    d2 = x_sq[:, None] + x_sq[None, :] - 2 * (x @ x.T)
    d2 = np.maximum(d2, 0)
    np.fill_diagonal(d2, 0)
    
    iu = np.triu_indices(N, k=1)
    d2_flat = d2[iu]
    
    if len(d2_flat) == 0:
        return 0.0
    
    # Soft min and max
    # soft_min = -k * log(sum(exp(-d2/k)))
    # soft_max = k * log(sum(exp(d2/k)))
    
    # Numerical stability for soft_min
    m_min = np.min(d2_flat)
    exp_terms_min = np.exp(-(d2_flat - m_min) / k)
    sum_exp_min = np.sum(exp_terms_min)
    soft_min_val = m_min + k * np.log(sum_exp_min)
    
    # Numerical stability for soft_max
    m_max = np.max(d2_flat)
    exp_terms_max = np.exp((d2_flat - m_max) / k)
    sum_exp_max = np.sum(exp_terms_max)
    soft_max_val = m_max + k * np.log(sum_exp_max)
    
    # Add epsilon to avoid division by zero or log(0)
    eps = 1e-6
    soft_min_val = max(soft_min_val, eps)
    
    # Objective: minimize log(soft_max) - log(soft_min)
    # This is equivalent to minimizing soft_max / soft_min
    return np.log(soft_max_val) - np.log(soft_min_val)

def _local_refine(pts, max_time=5.0):
    """
    Perform local refinement using L-BFGS-B on the smooth ratio objective.
    Returns refined points and score.
    """
    t_start = time.time()
    v0 = pts.flatten()
    
    try:
        res = minimize(
            _smooth_ratio_obj,
            v0,
            args=(0.1,),  # k=0.1 for sharpness
            method='L-BFGS-B',
            options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50}
        )
        
        v_opt = res.x
        pts_opt = v_opt.reshape(N, D)
        # Center points to reduce drift
        pts_opt = pts_opt - pts_opt.mean(axis=0)
        
        score_opt = _true_score(pts_opt)
        
        # Check time
        if time.time() - t_start > max_time:
            return pts, _true_score(pts)
            
        return pts_opt, score_opt
        
    except Exception:
        return pts, _true_score(pts)

def construct():
    global _t0
    if _t0 is None:
        _t0 = time.time()
    
    rng = np.random.default_rng(42)
    best_pts = None
    best_score = np.inf
    
    # Initial population: Fibonacci sphere and random points
    def get_fib_sphere():
        i = np.arange(N)
        phi = np.arccos(1 - 2 * (i + 0.5) / N)
        theta = np.pi * (1 + 5 ** 0.5) * i
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        return np.column_stack([x, y, z])
    
    # Create initial population
    pop_size = 50
    population = []
    
    # Add Fibonacci sphere
    fib_pts = get_fib_sphere()
    population.append(fib_pts.copy())
    
    # Add random points on unit sphere
    for _ in range(pop_size - 1):
        pts = rng.normal(size=(N, D))
        # Normalize to unit sphere
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / norms
        # Add some randomness in radius
        radii = rng.uniform(0.8, 1.2, size=(N, 1))
        pts = pts * radii
        population.append(pts)
    
    # Evaluate initial population
    for i, pts in enumerate(population):
        score = _true_score(pts)
        if score < best_score:
            best_score = score
            best_pts = pts.copy()
    
    # Evolutionary loop
    max_generations = 200
    mutation_scale = 0.5
    
    for gen in range(max_generations):
        if time.time() - _t0 > DEADLINE - 10:
            break
        
        # Create new population
        new_population = []
        
        for i in range(pop_size):
            if time.time() - _t0 > DEADLINE - 10:
                break
            
            # Select parents (tournament selection)
            if i == 0:
                # Elitism: keep the best point
                new_population.append(best_pts.copy())
                continue
            
            # Differential evolution: x_new = x_rand1 + F * (x_rand2 - x_rand3)
            idx1, idx2, idx3 = rng.choice(pop_size, size=3, replace=False)
            
            # Ensure idx1, idx2, idx3 are distinct and not i (to avoid self-selection issues in DE)
            while len(set([idx1, idx2, idx3, i])) < 4:
                idx1, idx2, idx3 = rng.choice(pop_size, size=3, replace=False)
            
            x1 = population[idx1]
            x2 = population[idx2]
            x3 = population[idx3]
            
            # Mutation
            x_trial = x1 + mutation_scale * (x2 - x3)
            
            # Crossover: binomial crossover
            crossover_mask = rng.random(size=(N, D)) < 0.5
            x_trial = np.where(crossover_mask, x_trial, population[i])
            
            # Boundary handling: clip to reasonable bounds (e.g., [-2, 2])
            x_trial = np.clip(x_trial, -2, 2)
            
            # Evaluate trial
            score_trial = _true_score(x_trial)
            
            # Selection: keep the better of the trial and the original
            score_orig = _true_score(population[i])
            if score_trial < score_orig:
                new_population.append(x_trial)
                if score_trial < best_score:
                    best_score = score_trial
                    best_pts = x_trial.copy()
            else:
                new_population.append(population[i])
        
        if len(new_population) < pop_size:
            break
            
        population = new_population
        
        # Periodic local refinement on the best point
        if gen % 5 == 0 and time.time() - _t0 < DEADLINE - 15:
            refined_pts, refined_score = _local_refine(best_pts, max_time=3.0)
            if refined_score < best_score:
                best_score = refined_score
                best_pts = refined_pts
                # Replace the worst point in the population with the refined best
                worst_idx = np.argmax([_true_score(p) for p in population])
                population[worst_idx] = best_pts.copy()
        
        # Occasionally add random new points to maintain diversity
        if gen % 10 == 0 and time.time() - _t0 < DEADLINE - 15:
            pts = rng.normal(size=(N, D))
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            pts = pts / norms
            radii = rng.uniform(0.5, 1.5, size=(N, 1))
            pts = pts * radii
            score_new = _true_score(pts)
            if score_new < best_score:
                best_score = score_new
                best_pts = pts.copy()
            # Replace a random point
            idx = rng.randint(pop_size)
            population[idx] = pts
    
    # Final local refinement on the best point
    if best_pts is not None and time.time() - _t0 < DEADLINE - 5:
        refined_pts, refined_score = _local_refine(best_pts, max_time=10.0)
        if refined_score < best_score:
            best_score = refined_score
            best_pts = refined_pts
    
    # Fallback
    if best_pts is None:
        best_pts = get_fib_sphere()

    return best_pts
