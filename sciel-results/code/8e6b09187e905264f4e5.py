"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree


def construct():
    """
    Use simulated annealing to maximize the minimum distance between circles
    and from walls for 50 equal circles in a unit square [-0.5, 0.5]^2.
    
    The score is r = min(min_{i<j} |c_i - c_j| / 2, min_i min_k (0.5 - |c_ik|)).
    We want to maximize r.
    
    Approach:
    1. Initialize with a hexagonal-like grid.
    2. Use simulated annealing with random perturbations.
    3. Objective: maximize the minimum of pairwise distances (divided by 2) and wall distances.
    4. Use a cooling schedule and accept worse moves with probability exp(delta/T).
    5. Use KD-tree for fast nearest-neighbor distance computation.
    6. Multiple restarts with different random seeds.
    """
    import time
    start_time = time.time()
    max_time = 115.0  # Leave buffer for overhead
    
    def compute_score_fast(centers, tree=None):
        """Compute the radius r for the given centers using KD-tree."""
        n = len(centers)
        if n < 2:
            return 0.5
        
        # Wall distances
        min_wall = np.min(0.5 - np.abs(centers))
        
        # If wall distance is already small, we can early exit
        # but still need to check pair distances
        
        # Use KD-tree for nearest neighbor
        if tree is None:
            tree = cKDTree(centers)
        
        # Find nearest neighbor for each point
        dists, _ = tree.query(centers, k=2)
        min_pair = np.min(dists[:, 1]) / 2.0
        
        return min(min_pair, min_wall)
    
    def hexagonal_init(seed):
        """Create a hexagonal grid initialization."""
        np.random.seed(seed)
        centers = np.zeros((N, 2))
        
        # Hexagonal packing initialization
        spacing = 0.142
        row_spacing = spacing * np.sqrt(3) / 2
        
        idx = 0
        y = -0.47
        row = 0
        while idx < N and y <= 0.47:
            x_offset = 0.0 if row % 2 == 0 else spacing / 2
            x = -0.47 + x_offset
            while x <= 0.47 and idx < N:
                centers[idx] = [x, y]
                idx += 1
                x += spacing
            y += row_spacing
            row += 1
        
        # Fill remaining if any
        if idx < N:
            remaining = N - idx
            side = int(np.ceil(np.sqrt(remaining)))
            ticks = (np.arange(side) + 0.5) / side - 0.5
            for i in range(remaining):
                centers[idx + i] = [ticks[i % side], ticks[i // side]]
        
        centers = np.clip(centers, -0.4999, 0.4999)
        return centers
    
    def anneal(centers, seed, max_iterations=50000):
        """Run simulated annealing from given initial centers."""
        np.random.seed(seed)
        
        current_score = compute_score_fast(centers)
        best_centers = centers.copy()
        best_score = current_score
        
        # Simulated annealing parameters
        T_start = 0.02
        T_end = 1e-8
        n_iterations = max_iterations
        step_size = 0.03
        
        # Cooling schedule
        alpha = (T_end / T_start) ** (1.0 / n_iterations)
        
        tree = cKDTree(centers)
        
        for it in range(n_iterations):
            # Check time
            if (it % 1000 == 0) and (time.time() - start_time > max_time - 5):
                break
            
            T = T_start * (alpha ** it)
            
            # Pick a random center to perturb
            i = np.random.randint(N)
            old_pos = centers[i].copy()
            
            # Random perturbation
            centers[i] += np.random.normal(0, step_size, 2)
            
            # Clip to stay within bounds
            centers[i] = np.clip(centers[i], -0.4999, 0.4999)
            
            # Update tree and compute new score
            tree = cKDTree(centers)
            new_score = compute_score_fast(centers, tree)
            
            if new_score >= current_score:
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_centers = centers.copy()
            else:
                # Accept with probability exp((new_score - current_score) / T)
                delta = new_score - current_score
                if delta < -T * 10:  # Very unlikely to accept
                    centers[i] = old_pos
                elif np.random.rand() < np.exp(delta / T):
                    current_score = new_score
                else:
                    # Reject: revert
                    centers[i] = old_pos
                    tree = cKDTree(centers)
            
            # Adaptive step size: reduce gradually
            if it % 500 == 0:
                step_size *= 0.97
        
        return best_centers, best_score
    
    # Try multiple restarts with different seeds
    best_overall = None
    best_overall_score = 0.0
    
    seeds = [42, 123, 456, 789, 101112, 202021, 303031, 404041]
    
    for seed in seeds:
        if time.time() - start_time > max_time - 10:
            break
        
        centers = hexagonal_init(seed)
        best_centers, best_score = anneal(centers, seed, max_iterations=40000)
        
        if best_score > best_overall_score:
            best_overall_score = best_score
            best_overall = best_centers.copy()
        
        # If we're close to the known best, do more iterations
        if best_score > 0.0713:
            # Do additional refinement
            best_centers2, best_score2 = anneal(best_centers, seed + 1000, max_iterations=30000)
            if best_score2 > best_overall_score:
                best_overall_score = best_score2
                best_overall = best_centers2.copy()
        
        if time.time() - start_time > max_time - 15:
            break
    
    if best_overall is None:
        # Fallback to a simple grid
        side = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(side) + 0.5) / side - 0.5
        best_overall = np.array([[ticks[i % side], ticks[i // side]] for i in range(N)])
    
    return best_overall
# EVOLVE-BLOCK-END
