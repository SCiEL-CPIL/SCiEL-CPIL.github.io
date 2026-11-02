"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_radius(centers):
    """Compute the radius: min of half min pairwise distance and min distance to boundary."""
    n = len(centers)
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    # Distance to boundary
    dists_to_wall = np.min(0.5 - np.abs(centers), axis=1)
    min_wall = np.min(dists_to_wall)
    
    return min(min_pair, min_wall)


def _energy(centers, r_target):
    """Energy function: penalize overlaps and boundary violations."""
    n = len(centers)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # Overlap penalty: pairs closer than 2*r_target
    overlap = np.maximum(0, 2 * r_target - dists)
    overlap_penalty = np.sum(overlap ** 2)
    
    # Boundary penalty
    wall_viol = np.maximum(0, np.abs(centers) - (0.5 - r_target))
    wall_penalty = np.sum(wall_viol ** 2)
    
    return overlap_penalty + wall_penalty


def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin
    
    best_centers = None
    best_radius = 0.0
    
    # Multiple restarts with different initializations
    n_restarts = 8
    
    for restart in range(n_restarts):
        if time.time() > deadline - 5:
            break
        
        rng = np.random.default_rng(seed=42 + restart * 1000)
        
        # Initialize with perturbed grid
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        centers = grid[:N].copy()
        
        # Add random perturbation
        perturbation_scale = 0.02 * (1 + 0.5 * restart)
        centers += rng.uniform(-perturbation_scale, perturbation_scale, size=centers.shape)
        
        # Clip to valid range
        centers = np.clip(centers, -0.48, 0.48)
        
        current_radius = _compute_radius(centers)
        if current_radius > best_radius:
            best_radius = current_radius
            best_centers = centers.copy()
        
        # Simulated annealing with momentum
        T = 0.005  # Initial temperature
        T_min = 1e-8
        cooling_rate = 0.995
        momentum = np.zeros_like(centers)
        learning_rate = 0.01
        n_steps = 5000
        
        for step in range(n_steps):
            if time.time() > deadline - 2:
                break
            
            # Compute gradient of energy (approximate via finite differences)
            # Use a simpler approach: random direction with acceptance criterion
            
            # Generate random perturbation
            delta = rng.normal(0, learning_rate, size=centers.shape)
            new_centers = centers + delta
            
            # Keep within bounds
            new_centers = np.clip(new_centers, -0.49, 0.49)
            
            # Compute radii
            new_radius = _compute_radius(new_centers)
            current_r = _compute_radius(centers)
            
            # Accept if better, or with probability based on temperature
            if new_radius >= current_r:
                centers = new_centers
                momentum = 0.9 * momentum + 0.1 * delta
            else:
                delta_E = current_r - new_radius
                if delta_E < 0:
                    centers = new_centers
                    momentum = 0.9 * momentum + 0.1 * delta
                else:
                    # Metropolis acceptance
                    if rng.random() < np.exp(-delta_E / max(T, 1e-10)):
                        centers = new_centers
                        momentum = 0.9 * momentum + 0.1 * delta
                    else:
                        momentum = 0.9 * momentum
        
            # Adaptive learning rate
            learning_rate = max(0.001, 0.01 * np.exp(-step / 2000))
            T = max(T_min, T * cooling_rate)
        
        # Local refinement: greedy pairwise optimization
        for iter_pass in range(3):
            if time.time() > deadline - 2:
                break
            for i in range(N):
                if time.time() > deadline - 2:
                    break
                # Try to move center i to maximize its minimum distance to others and walls
                best_pos = centers[i]
                best_local_r = _compute_radius(centers)
                
                # Try random perturbations around current position
                for _ in range(20):
                    trial = centers[i] + rng.uniform(-0.01, 0.01, size=3)
                    trial = np.clip(trial, -0.49, 0.49)
                    centers[i] = trial
                    r = _compute_radius(centers)
                    if r > best_local_r:
                        best_local_r = r
                        best_pos = trial
                centers[i] = best_pos
        
        current_radius = _compute_radius(centers)
        if current_radius > best_radius:
            best_radius = current_radius
            best_centers = centers.copy()
    
    # Final validation
    if best_centers is not None:
        final_radius = _compute_radius(best_centers)
        if final_radius > best_radius:
            best_radius = final_radius
    
    # Ensure valid output
    result = best_centers if best_centers is not None else np.zeros((N, 3))
    result = np.clip(result, -0.5, 0.5)
    
    return result
# EVOLVE-BLOCK-END
