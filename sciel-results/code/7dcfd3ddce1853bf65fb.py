# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

# Global deadline guard
_GLOBAL_START_TIME = None
_DEADLINE = 325.0  # 325s wall clock limit (evaluator kills at 330s)

def _get_remaining_time():
    global _GLOBAL_START_TIME
    if _GLOBAL_START_TIME is None:
        _GLOBAL_START_TIME = time.time()
    return _DEADLINE - (time.time() - _GLOBAL_START_TIME)

def _morse_energy(x, rho):
    """Compute Morse energy for a set of positions (vectorized)."""
    n = x.shape[0]
    if n < 2:
        return 0.0
    # Compute pairwise distances
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    # Mask diagonal
    np.fill_diagonal(dist, np.inf)
    
    # Morse term
    # E = exp(rho(1-r)) * (exp(rho(1-r)) - 2)
    # We only need i < j, so we can sum upper triangle
    er = rho * (1.0 - dist)
    # Clip to avoid overflow for very small r (though min_dist check usually prevents this)
    # If r is very small, er is large, exp(er) is huge. 
    # However, we should return a large penalty if r < 0.1
    if np.any(dist < 0.1):
         return 1e10
         
    exp_er = np.exp(er)
    energy = np.sum(exp_er * (exp_er - 2.0)) / 2.0 # Divide by 2 because upper and lower triangle
    return energy

def _min_distance(x):
    """Compute minimum distance between any pair of atoms (vectorized)."""
    n = x.shape[0]
    if n < 2:
        return np.inf
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    return np.min(dist)

def _is_valid(x):
    """Check if the configuration is valid (atoms at least 0.5 apart)."""
    return _min_distance(x) >= 0.5

def _generate_random_cluster(n, seed=42, min_dist=0.55):
    """Generate a random cluster with minimum distance constraint (vectorized check)."""
    rng = np.random.default_rng(seed)
    
    # Estimate radius for dense packing
    # Volume ~ n * (4/3) * pi * (0.55/2)^3 ... approx radius ~ 1.2 * n^(1/3)
    radius = 1.2 * (n ** (1.0/3.0)) + 0.5
    points = []
    attempts = 0
    max_attempts = n * 50
    
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if not points:
                points.append(p)
            else:
                # Vectorized distance check
                dists = np.linalg.norm(np.array(points) - p, axis=1)
                if np.min(dists) >= min_dist:
                    points.append(p)
        attempts += 1
    
    # If we couldn't place all atoms, fill with random points (less constrained)
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    
    return np.array(points)

def _relax_and_validate(positions, potential, rho):
    """Relax positions and check validity."""
    n = positions.shape[0]
    try:
        # Check initial validity to avoid wasting time on obviously bad starts
        if _min_distance(positions) < 0.4: # Slightly looser check before relax
            return float('inf'), None
            
        energy, relaxed = relax(positions, potential, rho)
        
        # Verify energy and validity
        if np.isinf(energy) or np.isnan(energy):
            return float('inf'), None
            
        min_d = _min_distance(relaxed)
        if min_d >= 0.5:
            # Recompute energy exactly to be safe? relax returns energy, but let's trust it 
            # unless it looks suspicious. The task says "Energies are recomputed exactly".
            # The evaluator does this. We just need to provide coordinates.
            return energy, relaxed
        else:
            return float('inf'), None
    except Exception:
        return float('inf'), None

def _basin_hop(start_positions, potential, rho, max_iterations=50, initial_step=0.5, initial_temp=2.0):
    """
    Perform basin hopping optimization.
    Returns (best_energy, best_positions).
    """
    n = start_positions.shape[0]
    current_energy, current_positions = _relax_and_validate(start_positions, potential, rho)
    
    if current_positions is None:
        # Try with slightly perturbed start
        perturbed = start_positions + np.random.normal(0, 0.1, start_positions.shape)
        current_energy, current_positions = _relax_and_validate(perturbed, potential, rho)
        if current_positions is None:
            return float('inf'), None
    
    best_energy = current_energy
    best_positions = current_positions.copy()
    
    step_size = initial_step
    temp = initial_temp
    
    for it in range(max_iterations):
        if _get_remaining_time() < 5.0:
            break
        
        # Perturb
        noise = np.random.normal(0, step_size, current_positions.shape)
        new_positions = current_positions + noise
        
        # Relax
        new_energy, new_relaxed = _relax_and_validate(new_positions, potential, rho)
        
        if new_relaxed is None:
            # Try smaller perturbation
            noise = np.random.normal(0, step_size * 0.5, current_positions.shape)
            new_positions = current_positions + noise
            new_energy, new_relaxed = _relax_and_validate(new_positions, potential, rho)
        
        if new_relaxed is None:
            # Skip this iteration
            continue
        
        # Accept/reject
        delta_e = new_energy - current_energy
        if delta_e < 0 or (temp > 1e-6 and np.random.rand() < math.exp(-delta_e / temp)):
            current_energy = new_energy
            current_positions = new_relaxed
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_positions = current_positions.copy()
        
        # Anneal
        temp *= 0.95
        step_size *= 0.98
        
        # If stuck, increase step size occasionally
        if it > 0 and it % 20 == 0:
            step_size *= 1.2
    
    return best_energy, best_positions

def construct(instance):
    global _GLOBAL_START_TIME
    if _GLOBAL_START_TIME is None:
        _GLOBAL_START_TIME = time.time()
        
    n = instance["n"]
    rho = instance.get("rho", instance.get("kind"))
    potential = instance["potential"]
    
    if rho is None:
        rho = 6  # Default
    
    best_energy = float('inf')
    best_positions = None
    
    # Generate diverse initial guesses
    # We focus on random packing as it's more robust for these sizes than perfect lattices
    # which might relax into different local minima or be too symmetric.
    initial_guesses = []
    
    # 1. Random clusters with different seeds
    # Generate 5-10 random starts. Relaxation is the expensive part, but vectorized checks help.
    for seed in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        if _get_remaining_time() < 10.0:
            break
        try:
            initial_guesses.append(_generate_random_cluster(n, seed=seed))
        except Exception:
            pass
    
    # Relax all initial guesses and keep the best ones
    relaxed_results = []
    for guess in initial_guesses:
        if _get_remaining_time() < 10.0:
            break
        
        energy, positions = _relax_and_validate(guess, potential, rho)
        if positions is not None and energy < float('inf'):
            relaxed_results.append((energy, positions))
    
    # Sort by energy
    relaxed_results.sort(key=lambda x: x[0])
    
    # Update best
    if relaxed_results:
        best_energy = relaxed_results[0][0]
        best_positions = relaxed_results[0][1].copy()
    
    # Perform basin hopping from the top few starts
    # Use more starts if time permits, but prioritize the best one with longer runs
    num_bh_starts = min(len(relaxed_results), 5)
    
    # Adaptive BH parameters based on rho
    if rho <= 3:
        initial_step = 0.8
        initial_temp = 5.0
    elif rho <= 6:
        initial_step = 0.5
        initial_temp = 3.0
    else:
        initial_step = 0.3
        initial_temp = 2.0
        
    # More iterations for smaller n (faster per iteration)
    if n <= 40:
        max_iter = 100
    elif n <= 60:
        max_iter = 60
    else:
        max_iter = 40
        
    for idx in range(num_bh_starts):
        if _get_remaining_time() < 15.0:
            break
        
        start_energy, start_pos = relaxed_results[idx]
        
        # Run BH
        # For the first (best) start, run longer
        if idx == 0:
             bh_energy, bh_positions = _basin_hop(
                start_pos, potential, rho,
                max_iterations=max_iter,
                initial_step=initial_step,
                initial_temp=initial_temp
            )
        else:
             # For subsequent starts, run shorter to allow more restarts
             bh_energy, bh_positions = _basin_hop(
                start_pos, potential, rho,
                max_iterations=max_iter // 2,
                initial_step=initial_step,
                initial_temp=initial_temp
            )
        
        if bh_positions is not None and bh_energy < best_energy:
            best_energy = bh_energy
            best_positions = bh_positions.copy()
    
    # Final fallback
    if best_positions is None:
        # Absolute fallback: generate a simple valid configuration
        best_positions = _generate_random_cluster(n, seed=99)
        # Try to relax it one last time
        energy, relaxed = _relax_and_validate(best_positions, potential, rho)
        if relaxed is not None:
            best_positions = relaxed
        elif _min_distance(best_positions) < 0.5:
            # Spread out if invalid
            best_positions = best_positions * 1.5
    
    return best_positions
# EVOLVE-BLOCK-END
