"""Packing 50 equal spheres in a unit cube via simulated annealing + local refinement."""
import time
import numpy as np
from scipy.optimize import minimize

N = 50
DEADLINE = 115.0  # seconds, safety margin before 120s limit


def _score(centers):
    """Compute the maximum radius for given centers."""
    # Wall distances: min distance from each center to the boundary
    wall_dists = 0.5 - np.max(np.abs(centers), axis=1)
    min_wall = np.min(wall_dists)
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    return min(min_pair / 2.0, min_wall)


def _init_cubic():
    """Initial cubic grid placement."""
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N].copy()


def _init_random():
    """Random initialization inside the cube."""
    return (np.random.rand(N, 3) - 0.5)


def _simulated_annealing(centers, start_time, time_budget):
    """Simulated annealing to maximize the minimum clearance."""
    best_centers = centers.copy()
    best_score = _score(centers)
    current_centers = centers.copy()
    current_score = best_score
    
    T = 0.01  # initial temperature
    cooling = 0.995
    step_size = 0.05
    
    while T > 1e-6 and (time.time() - start_time) < time_budget:
        # Perturb one or two random centers
        n_perturb = 1 if np.random.rand() < 0.7 else 2
        idx = np.random.choice(N, size=n_perturb, replace=False)
        
        new_centers = current_centers.copy()
        for i in idx:
            new_centers[i] += np.random.randn(3) * step_size
            # Clamp to cube with small margin
            new_centers[i] = np.clip(new_centers[i], -0.49, 0.49)
        
        new_score = _score(new_centers)
        delta = new_score - current_score
        
        if delta > 0 or np.random.rand() < np.exp(delta / T):
            current_centers = new_centers
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_centers = current_centers.copy()
        
        T *= cooling
        step_size = max(0.005, step_size * 0.999)
    
    return best_centers, best_score


def _soft_min(values, beta=100.0):
    """
    Compute a differentiable approximation of the minimum of values.
    Uses log-sum-exp: -1/beta * log(sum(exp(-beta * v)))
    """
    # Shift for numerical stability
    min_val = np.min(values)
    exps = np.exp(-beta * (values - min_val))
    log_sum_exp = np.log(np.sum(exps))
    return min_val - np.log(np.sum(exps)) / beta


def _objective_smooth(x, beta=100.0):
    """
    Objective function for L-BFGS-B.
    Minimizes the negative soft-min of clearances.
    Clearances are: pairwise distances / 2, and wall distances.
    """
    centers = x.reshape(N, 3)
    
    # Wall distances
    wall_dists = 0.5 - np.max(np.abs(centers), axis=1)
    
    # Pairwise distances
    # To avoid O(N^2) memory blowup in gradients if N was larger, we chunk, 
    # but for N=50, 50x50x3 is small.
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # We want to maximize min(min_pair/2, min_wall)
    # So we minimize -min(min_pair/2, min_wall)
    # Using soft-min:
    # We treat all pairwise distances/2 and all wall distances as a single pool of constraints?
    # No, the radius r is limited by min(all pair dists / 2, all wall dists).
    # So we form a vector of all clearances:
    # clearances = [d_ij / 2 for i<j, w_i for i]
    
    # Efficiently gather clearances
    # Pairwise
    iu = np.triu_indices(N, k=1)
    pair_dists = dists[iu]
    pair_clears = pair_dists / 2.0
    
    # Combine
    all_clears = np.concatenate([pair_clears, wall_dists])
    
    # Soft min
    sm = _soft_min(all_clears, beta)
    
    # We want to maximize sm, so minimize -sm
    return -sm


def _local_refinement(centers, start_time, time_budget):
    """Local optimization using scipy minimize on negative score."""
    best_centers = centers.copy()
    best_score = _score(centers)
    
    # Try a few random restarts of local optimization
    n_restarts = 5
    beta = 100.0
    
    for restart in range(n_restarts):
        if (time.time() - start_time) > time_budget:
            break
        
        if restart == 0:
            x0 = centers.ravel()
        else:
            # Add small random perturbation
            x0 = centers.ravel() + np.random.randn(N * 3) * 0.01
            x0 = x0.reshape(N, 3)
            x0 = np.clip(x0, -0.49, 0.49).ravel()
        
        try:
            # Use L-BFGS-B with smooth objective
            result = minimize(_objective_smooth, x0, method='L-BFGS-B', 
                             args=(beta,),
                             options={'maxiter': 1000, 'maxls': 50, 'ftol': 1e-10, 'gtol': 1e-8})
            
            new_centers = result.x.reshape(N, 3)
            new_centers = np.clip(new_centers, -0.49, 0.49)
            new_score = _score(new_centers)
            
            if new_score > best_score:
                best_score = new_score
                best_centers = new_centers.copy()
                
        except Exception:
            pass
    
    return best_centers, best_score


def construct():
    start_time = time.time()
    
    # Start with cubic grid
    centers = _init_cubic()
    best_centers = centers.copy()
    best_score = _score(centers)
    
    # Phase 1: Simulated annealing
    time_budget_sa = 40.0
    centers, score = _simulated_annealing(centers, start_time, time_budget_sa)
    if score > best_score:
        best_score = score
        best_centers = centers.copy()
    
    # Phase 2: Local refinement with smooth objective
    time_budget_lr = 60.0
    centers, score = _local_refinement(centers, start_time, time_budget_lr)
    if score > best_score:
        best_score = score
        best_centers = centers.copy()
    
    # Check deadline
    elapsed = time.time() - start_time
    if elapsed > DEADLINE:
        return best_centers
    
    # Phase 3: More SA with different initial temperature
    if elapsed < DEADLINE - 10:
        centers = _init_random()
        centers, score = _simulated_annealing(centers, start_time, DEADLINE - 20)
        if score > best_score:
            best_score = score
            best_centers = centers.copy()
        
        # Local refinement on the new best
        if time.time() - start_time < DEADLINE - 5:
            centers, score = _local_refinement(centers, start_time, DEADLINE - 5)
            if score > best_score:
                best_score = score
                best_centers = centers.copy()
    
    return best_centers
