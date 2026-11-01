"""Packing 50 equal spheres in a unit cube using simulated annealing + local refinement."""
import numpy as np
import time

N = 50

def _compute_radius(centers):
    """Compute the largest common radius for the given centers."""
    centers = np.asarray(centers, dtype=np.float64)
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Wall distance
    wall = 0.5 - np.max(np.abs(centers), axis=1)
    min_wall = np.min(wall)
    if min_wall <= 0:
        return 0.0
    # Pairwise distances
    # Use broadcasting for N=50 (2500 pairs)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    return min(min_wall, min_pair / 2.0)


def _random_initial():
    """Generate a random initial configuration."""
    return np.random.uniform(-0.5 + 0.01, 0.5 - 0.01, size=(N, 3))


def _lloyd_step(centers, n_iter=30):
    """Apply Lloyd's algorithm: move each center toward centroid of its Voronoi cell (approximated by nearest-neighbor averaging)."""
    centers = centers.copy()
    for _ in range(n_iter):
        # For each point, find its 6 nearest neighbors
        # Compute pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        new_centers = centers.copy()
        for i in range(N):
            # Get 6 nearest neighbors
            nn_idx = np.argsort(dists[i])[:6]
            nn_centers = centers[nn_idx]
            
            # Centroid of nearest neighbors
            centroid = np.mean(nn_centers, axis=0)
            
            # Also consider wall constraint: if center is close to a wall, pull it inward
            # Simple approach: move toward centroid but clip to cube
            new_c = 0.5 * centers[i] + 0.5 * centroid
            # Clip to cube interior with small margin
            new_c = np.clip(new_c, -0.5 + 0.001, 0.5 - 0.001)
            new_centers[i] = new_c
        
        centers = new_centers
    
    return centers


def _simulated_annealing(centers, max_time=100.0):
    """Run simulated annealing to maximize the minimum radius."""
    start_time = time.time()
    deadline = start_time + max_time
    
    current = centers.copy()
    current_r = _compute_radius(current)
    
    best = current.copy()
    best_r = current_r
    
    T = 0.05  # Initial temperature
    T_min = 1e-8
    cooling = 0.995
    
    while T > T_min:
        if time.time() > deadline - 5.0:
            break
        
        # Generate a perturbation
        candidate = current.copy()
        # Move a random subset of points
        n_move = np.random.randint(1, 6)
        idx = np.random.choice(N, size=n_move, replace=False)
        step_size = T * 0.5
        candidate[idx] += np.random.normal(0, step_size, size=(n_move, 3))
        # Clip to cube
        candidate = np.clip(candidate, -0.5 + 0.001, 0.5 - 0.001)
        
        cand_r = _compute_radius(candidate)
        
        if cand_r >= current_r or np.random.random() < np.exp((cand_r - current_r) / T):
            current = candidate
            current_r = cand_r
            if current_r > best_r:
                best = current.copy()
                best_r = current_r
        
        T *= cooling
    
    return best, best_r


def _local_refinement(centers, max_time=20.0):
    """Local refinement: iteratively move the most constrained point."""
    start_time = time.time()
    deadline = start_time + max_time
    
    centers = centers.copy()
    current_r = _compute_radius(centers)
    
    best = centers.copy()
    best_r = current_r
    
    for _ in range(10000):
        if time.time() > deadline - 2.0:
            break
        
        # Find the point with the smallest clearance
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair_dists = np.min(dists, axis=1)
        
        wall = 0.5 - np.max(np.abs(centers), axis=1)
        
        # Clearance for each point
        clearance = np.minimum(min_pair_dists / 2.0, wall)
        
        # Find the most constrained point
        i = np.argmin(clearance)
        
        # Try to move point i to improve its clearance
        # Direction: away from nearest neighbor and toward center
        nn = np.argmin(dists[i])
        away = centers[i] - centers[nn]
        away_norm = np.linalg.norm(away)
        if away_norm > 1e-10:
            away = away / away_norm
        
        # Also move toward center if close to wall
        toward_center = -centers[i]
        tc_norm = np.linalg.norm(toward_center)
        if tc_norm > 1e-10:
            toward_center = toward_center / tc_norm
        
        # Blend direction
        direction = 0.6 * away + 0.4 * toward_center
        dir_norm = np.linalg.norm(direction)
        if dir_norm > 1e-10:
            direction = direction / dir_norm
        
        # Try several step sizes
        improved = False
        for step in [0.005, 0.002, 0.001, 0.0005]:
            candidate = centers.copy()
            candidate[i] = centers[i] + step * direction
            candidate[i] = np.clip(candidate[i], -0.5 + 0.001, 0.5 - 0.001)
            
            cand_r = _compute_radius(candidate)
            if cand_r > current_r:
                centers = candidate
                current_r = cand_r
                improved = True
                break
        
        if not improved:
            # Try random small moves for the constrained point
            for _ in range(10):
                candidate = centers.copy()
                candidate[i] += np.random.normal(0, 0.003, size=3)
                candidate[i] = np.clip(candidate[i], -0.5 + 0.001, 0.5 - 0.001)
                cand_r = _compute_radius(candidate)
                if cand_r > current_r:
                    centers = candidate
                    current_r = cand_r
                    break
        
        if current_r > best_r:
            best = centers.copy()
            best_r = current_r
    
    return best, best_r


def construct():
    """Construct the best packing of 50 equal spheres in a unit cube."""
    start_time = time.time()
    total_deadline = start_time + 140.0  # Safety margin
    
    np.random.seed(42)
    
    # Phase 1: Multiple random restarts with simulated annealing
    best_centers = None
    best_r = 0.0
    
    n_restarts = 5
    sa_time_each = 12.0
    
    for restart in range(n_restarts):
        if time.time() > total_deadline - 25.0:
            break
        
        # Initialize
        if restart == 0:
            # Start with a grid-like perturbation
            m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
            ticks = (np.arange(m) + 0.5) / m - 0.5
            grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
            centers = grid[:N].copy()
            centers += np.random.normal(0, 0.01, size=centers.shape)
        else:
            centers = _random_initial()
        
        # Simulated annealing
        result, r = _simulated_annealing(centers, max_time=sa_time_each)
        
        if r > best_r:
            best_r = r
            best_centers = result.copy()
    
    # Phase 2: Local refinement on the best result
    if best_centers is not None:
        if time.time() < total_deadline - 10.0:
            refined, refined_r = _local_refinement(best_centers, max_time=15.0)
            if refined_r > best_r:
                best_r = refined_r
                best_centers = refined
    
    # Phase 3: One more SA round from the refined solution if time permits
    if time.time() < total_deadline - 15.0 and best_centers is not None:
        final, final_r = _simulated_annealing(best_centers, max_time=10.0)
        if final_r > best_r:
            best_r = final_r
            best_centers = final
    
    if best_centers is None:
        # Fallback: simple grid
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        return grid[:N]
    
    return best_centers
