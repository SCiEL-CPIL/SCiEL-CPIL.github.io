"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13

# EVOLVE-BLOCK-START

def _compute_score(points):
    """Compute the minimum triangle area divided by the convex hull area."""
    n = len(points)
    
    # Compute convex hull area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume  # In 2D, volume is area
    except Exception:
        return 0.0
    
    if hull_area <= 1e-12:
        return 0.0
    
    # Compute minimum triangle area
    min_tri_area = float('inf')
    for i, j, k in combinations(range(n), 3):
        p1 = points[i]
        p2 = points[j]
        p3 = points[k]
        area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - 
                         (p3[0] - p1[0]) * (p2[1] - p1[1]))
        if area < min_tri_area:
            min_tri_area = area
    
    return min_tri_area / hull_area


def _params_to_points(params):
    """
    Convert 13 parameters to 13 points using 1+12 ring symmetry.
    params[0]: central x offset
    params[1]: central y offset  
    params[2:14]: 12 radial values for the ring points
    The 12 ring points have fixed angular spacing (every 30 degrees)
    """
    cx = params[0]
    cy = params[1]
    radii = params[2:14]
    
    # Fixed angles: 12 points at 30-degree intervals
    angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    
    # Build the 12 ring points
    ring_x = radii * np.cos(angles)
    ring_y = radii * np.sin(angles)
    
    # Central point
    center = np.array([cx, cy])
    
    # Stack all points
    points = np.vstack([center.reshape(1, 2), 
                        np.column_stack((ring_x, ring_y))])
    
    return points


def _score_from_params(params):
    """Score a parameter vector by converting to points and computing score."""
    points = _params_to_points(params)
    return _compute_score(points)


def _differential_evolution(params_bounds, max_iter=2000, pop_size=50, 
                            mutation=0.8, crossover=0.9, seed=42):
    """
    Custom differential evolution on the 13D parameter space.
    params_bounds: list of (low, high) for each of 13 parameters
    """
    rng = np.random.RandomState(seed)
    n_params = len(params_bounds)
    
    # Initialize population
    pop = np.zeros((pop_size, n_params))
    for i in range(n_params):
        pop[:, i] = rng.uniform(params_bounds[i][0], params_bounds[i][1], pop_size)
    
    # Evaluate initial population
    scores = np.array([_score_from_params(pop[i]) for i in range(pop_size)])
    
    best_idx = np.argmax(scores)
    best_params = pop[best_idx].copy()
    best_score = scores[best_idx]
    
    for gen in range(max_iter):
        # Generate new population
        new_pop = np.zeros((pop_size, n_params))
        new_scores = np.zeros(pop_size)
        
        for i in range(pop_size):
            # Select three distinct individuals
            a, b, c = rng.choice(pop_size, size=3, replace=False)
            
            # Differential mutation
            mutant = pop[a] + mutation * (pop[b] - pop[c])
            
            # Clip to bounds
            for j in range(n_params):
                mutant[j] = np.clip(mutant[j], params_bounds[j][0], params_bounds[j][1])
            
            # Crossover
            trial = pop[i].copy()
            j_rand = rng.randint(n_params)
            for j in range(n_params):
                if rng.random() < crossover or j == j_rand:
                    trial[j] = mutant[j]
            
            # Evaluate
            trial_score = _score_from_params(trial)
            
            # Selection
            if trial_score >= scores[i]:
                new_pop[i] = trial
                new_scores[i] = trial_score
            else:
                new_pop[i] = pop[i]
                new_scores[i] = scores[i]
        
        # Update population
        pop = new_pop
        scores = new_scores
        
        # Update best
        gen_best_idx = np.argmax(scores)
        if scores[gen_best_idx] > best_score:
            best_score = scores[gen_best_idx]
            best_params = pop[gen_best_idx].copy()
    
    return best_params, best_score


def _local_search_params(params, max_iter=200, step_size=0.01, seed=42):
    """Local search in parameter space."""
    rng = np.random.RandomState(seed)
    n_params = len(params)
    current = params.copy()
    current_score = _score_from_params(current)
    best = current.copy()
    best_score = current_score
    
    for i in range(max_iter):
        # Pick a random parameter to perturb
        idx = rng.randint(n_params)
        
        improved = False
        for _ in range(30):
            new_params = current.copy()
            new_params[idx] = current[idx] + rng.normal(0, step_size)
            
            # Clip to reasonable bounds
            new_params[0] = np.clip(new_params[0], -0.5, 0.5)  # center x
            new_params[1] = np.clip(new_params[1], -0.5, 0.5)  # center y
            new_params[2:14] = np.clip(new_params[2:14], 0.3, 1.5)  # radii
            
            new_score = _score_from_params(new_params)
            
            if new_score > current_score:
                current = new_params
                current_score = new_score
                improved = True
                if current_score > best_score:
                    best = current.copy()
                    best_score = current_score
                break
        
        if not improved:
            step_size *= 0.9
            if step_size < 1e-6:
                break
    
    return best, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Parameter bounds for 1+12 ring symmetry
    # [center_x, center_y, r1, r2, ..., r12]
    params_bounds = [
        (-0.3, 0.3),   # center x
        (-0.3, 0.3),   # center y
    ] + [(0.5, 1.2) for _ in range(12)]  # 12 radii
    
    best_params = None
    best_score = -float('inf')
    best_points = None
    
    # Run differential evolution with multiple seeds
    n_de_runs = 3
    
    for run in range(n_de_runs):
        if time.time() > deadline - 30:
            break
        
        seed = 100 + run * 10
        de_params, de_score = _differential_evolution(
            params_bounds, 
            max_iter=500, 
            pop_size=30,
            mutation=0.7 + 0.1 * run,
            crossover=0.8,
            seed=seed
        )
        
        if de_score > best_score:
            best_score = de_score
            best_params = de_params.copy()
            best_points = _params_to_points(de_params).copy()
        
        # Local search on DE result
        if time.time() > deadline - 15:
            break
        
        ls_params, ls_score = _local_search_params(
            de_params, 
            max_iter=100, 
            step_size=0.02, 
            seed=seed + 100
        )
        
        if ls_score > best_score:
            best_score = ls_score
            best_params = ls_params.copy()
            best_points = _params_to_points(ls_params).copy()
    
    # If we have a good result, return it
    if best_points is not None:
        return best_points
    
    # Fallback: generate a reasonable symmetric configuration
    # 1 center + 12 on a circle
    angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    center = np.array([0.0, 0.0])
    ring = np.column_stack((np.cos(angles), np.sin(angles)))
    points = np.vstack([center.reshape(1, 2), ring])
    
    return points

# EVOLVE-BLOCK-END
