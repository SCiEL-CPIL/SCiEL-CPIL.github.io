import numpy as np

N = 100

def construct():
    np.random.seed(42)
    
    # 1. Greedy Initialization
    # Build the set of centers one by one, choosing positions that maximize
    # the minimum distance to existing centers and boundaries.
    
    centers = np.zeros((N, 3))
    centers[0] = [0.0, 0.0, 0.0]
    
    num_candidates = 2000
    
    for i in range(1, N):
        candidates = np.random.uniform(-0.5, 0.5, (num_candidates, 3))
        
        # Calculate min distance to existing centers
        diff = candidates[:, np.newaxis, :] - centers[:i][np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        min_dist_existing = np.min(dists, axis=1)
        
        # Calculate min distance to boundaries
        max_abs = np.max(np.abs(candidates), axis=1)
        min_dist_boundary = 0.5 - max_abs
        
        # Score is the max possible radius for this candidate
        scores = np.minimum(min_dist_existing / 2.0, min_dist_boundary)
        
        best_idx = np.argmax(scores)
        centers[i] = candidates[best_idx]
        
    # 2. Simulated Annealing
    def calc_radius(c):
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists)
        
        max_abs_coord = np.max(np.abs(c), axis=1)
        min_boundary = 0.5 - np.max(max_abs_coord)
        
        return min(min_pair / 2.0, min_boundary)

    r_current = calc_radius(centers)
    r_best = r_current
    best_centers = centers.copy()
    
    T = 0.02
    cooling_rate = 0.9995
    max_steps = 100000
    step_size = 0.03
    
    for step in range(max_steps):
        i = np.random.randint(N)
        delta = np.random.normal(0, step_size, 3)
        
        new_centers = centers.copy()
        new_centers[i] = np.clip(centers[i] + delta, -0.5, 0.5)
        
        r_new = calc_radius(new_centers)
        
        if r_new > r_current:
            centers = new_centers
            r_current = r_new
            if r_current > r_best:
                r_best = r_current
                best_centers = new_centers.copy()
        else:
            if T > 1e-9:
                prob = np.exp((r_new - r_current) / T)
                if np.random.rand() < prob:
                    centers = new_centers
                    r_current = r_new
                    if r_current > r_best:
                        r_best = r_current
                        best_centers = new_centers.copy()
        
        T *= cooling_rate
        step_size *= 0.99995

    return best_centers
