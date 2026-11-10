import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.special import logsumexp

N = 50

def _pairwise_distances(pos):
    """Compute all pairwise distances efficiently."""
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    return dists

def compute_radius(positions):
    """
    Compute the maximum common radius for given circle centers.
    positions: array of shape (N, 2)
    """
    # Boundary constraint: distance to nearest wall
    # Square is [-0.5, 0.5] x [-0.5, 0.5]
    # Distance from x_i to wall is 0.5 - |x_i|
    min_bound = 0.5 - np.max(np.abs(positions))
    if min_bound <= 0:
        return 0.0
    
    # Pairwise distance constraint
    dists = cdist(positions, positions)
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    
    return min(min_bound, min_dist / 2.0)

def soft_min_radius(x, temperature=0.01):
    """
    Smooth approximation of the min radius using log-sum-exp.
    Maximizing the soft-min is equivalent to maximizing the worst-case constraint.
    """
    pos = x.reshape(N, 2)
    
    # Boundary terms: 0.5 - |x_i| and 0.5 - |y_i|
    bound_terms = np.concatenate([
        0.5 - np.abs(pos[:, 0]),
        0.5 - np.abs(pos[:, 1])
    ])
    
    # Pairwise distance terms / 2
    dists = cdist(pos, pos)
    np.fill_diagonal(dists, np.inf)
    iu = np.triu_indices(N, k=1)
    pair_terms = dists[iu] / 2.0
    
    # Combine all constraints
    all_terms = np.concatenate([bound_terms, pair_terms])
    
    # Stability check
    if np.any(all_terms < 0):
        return -10.0

    neg_terms = -all_terms / temperature
    # logsumexp handles stability
    soft_min = -temperature * logsumexp(neg_terms)
    
    return soft_min

def generate_grid_init(jitter=0.0, seed=0):
    """Generate a hexagonal grid initialization with optional jitter."""
    np.random.seed(seed)
    positions = []
    sqrt3 = np.sqrt(3.0)
    
    best_pos = None
    best_r = 0
    
    # Try multiple grid variations
    for n_rows in range(5, 10):
        n_cols = int(np.ceil(N / n_rows))
        
        for offset_pattern in [0, 1]:
            s_x = 1.0 / (n_cols + 0.5)
            s_y = s_x * sqrt3 / 2.0
            
            total_height = (n_rows - 1) * s_y + s_x
            if total_height > 1.0:
                scale = 1.0 / total_height
                s_x *= scale
                s_y *= scale
            
            pos_list = []
            for row in range(n_rows):
                y = -0.5 + s_x/2 + row * s_y
                if y > 0.5 - s_x/2:
                    continue
                n_in_row = n_cols if row % 2 == offset_pattern else n_cols - 1
                if row % 2 == 1:
                    x_start = -0.5 + s_x
                else:
                    x_start = -0.5 + s_x/2
                
                for col in range(n_in_row):
                    x = x_start + col * s_x
                    if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                        pos_list.append([x, y])
                    if len(pos_list) >= N:
                        break
                if len(pos_list) >= N:
                    break
            
            if len(pos_list) >= N:
                pos_arr = np.array(pos_list[:N])
                if jitter > 0:
                    pos_arr += np.random.uniform(-jitter, jitter, pos_arr.shape)
                    pos_arr = np.clip(pos_arr, -0.5, 0.5)
                r = compute_radius(pos_arr)
                if r > best_r:
                    best_r = r
                    best_pos = pos_arr.copy()
    
    # Also try a simple square grid
    for n_grid in range(5, 8):
        s = 1.0 / (n_grid + 0.5)
        pos_list = []
        for i in range(n_grid):
            for j in range(n_grid):
                x = -0.5 + s/2 + i * s
                y = -0.5 + s/2 + j * s
                if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                    pos_list.append([x, y])
                if len(pos_list) >= N:
                    break
            if len(pos_list) >= N:
                break
        
        if len(pos_list) >= N:
            pos_arr = np.array(pos_list[:N])
            if jitter > 0:
                pos_arr += np.random.uniform(-jitter, jitter, pos_arr.shape)
                pos_arr = np.clip(pos_arr, -0.5, 0.5)
            r = compute_radius(pos_arr)
            if r > best_r:
                best_r = r
                best_pos = pos_arr.copy()
    
    if best_pos is not None:
        return best_pos
    # Fallback to random
    return np.random.uniform(-0.4, 0.4, (N, 2))

def polish(positions, time_limit=5.0):
    """
    Local optimization using scipy minimize with annealing temperature.
    """
    start_time = time.time()
    positions = positions.copy()
    current_r = compute_radius(positions)
    
    best_pos = positions.copy()
    best_r = current_r
    
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    # Annealing schedule for temperature
    temps = [0.02, 0.01, 0.005, 0.002, 0.001, 0.0005]
    
    for T in temps:
        if time.time() - start_time > time_limit:
            break
            
        def objective(x):
            if time.time() - start_time > time_limit:
                return 0
            pos = x.reshape(N, 2)
            if np.any(np.abs(pos) > 0.5):
                return 0
            return -soft_min_radius(x, temperature=T)
        
        try:
            result = minimize(objective, best_pos.flatten(), method='L-BFGS-B', 
                              bounds=bounds, options={'maxiter': 200, 'maxls': 50, 'ftol': 1e-10})
            new_pos = result.x.reshape(N, 2)
            if np.all(np.abs(new_pos) <= 0.5):
                new_r = compute_radius(new_pos)
                if new_r > best_r:
                    best_r = new_r
                    best_pos = new_pos.copy()
        except Exception:
            pass
            
    return best_pos

def hybrid_polish(positions, time_limit=10.0, seed=0):
    """
    Hybrid polishing: Gradient-based polish followed by final high-precision pass.
    """
    start_time = time.time()
    best_pos = positions.copy()
    best_r = compute_radius(best_pos)
    
    # Phase 1: Standard polish
    t1 = min(6.0, time_limit * 0.6)
    best_pos = polish(best_pos, time_limit=t1)
    best_r = compute_radius(best_pos)
    
    # Phase 2: Final high-precision polish
    remaining = time_limit - (time.time() - start_time)
    if remaining > 2.0:
        best_pos = polish(best_pos, time_limit=remaining)
        best_r = compute_radius(best_pos)
    
    return best_pos

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    best_pos = None
    best_r = 0.0
    
    # 1. Start with Grid (no jitter) - strong baseline
    pos = generate_grid_init(jitter=0.0, seed=0)
    pos = hybrid_polish(pos, time_limit=min(20.0, deadline - time.time()), seed=0)
    r = compute_radius(pos)
    if r > best_r:
        best_r = r
        best_pos = pos.copy()
    
    # 2. Try Jittered Grids to escape symmetric local minima
    for i in range(5):
        if time.time() > deadline - 20.0:
            break
        seed = 100 + i
        jitter = 0.01 * (i + 1)
        pos_init = generate_grid_init(jitter=jitter, seed=seed)
        
        current_time = time.time()
        alloc_time = min(15.0, (deadline - 20.0 - current_time) / max(1, 6 - i))
        
        if alloc_time < 3.0:
            break
            
        pos_polished = hybrid_polish(pos_init, time_limit=alloc_time, seed=seed)
        r = compute_radius(pos_polished)
        
        if r > best_r:
            best_r = r
            best_pos = pos_polished.copy()
    
    # 3. Try Random Starts with Repulsion-like initialization
    for i in range(3):
        if time.time() > deadline - 10.0:
            break
        seed = 200 + i
        np.random.seed(seed)
        pos_init = np.random.uniform(-0.4, 0.4, (N, 2))
        
        current_time = time.time()
        alloc_time = min(10.0, (deadline - 10.0 - current_time) / max(1, 4 - i))
        
        if alloc_time < 2.0:
            break
            
        pos_polished = hybrid_polish(pos_init, time_limit=alloc_time, seed=seed)
        r = compute_radius(pos_polished)
        
        if r > best_r:
            best_r = r
            best_pos = pos_polished.copy()
    
    # 4. Final Polish on the Best Result
    if time.time() < deadline - 2.0:
        remaining = deadline - 2.0 - time.time()
        if remaining > 5.0:
            best_pos = hybrid_polish(best_pos, time_limit=remaining, seed=42)
            best_r = compute_radius(best_pos)
            
    # Ensure valid output
    if best_pos is None:
        best_pos = generate_grid_init(jitter=0.0, seed=0)
        
    best_pos = np.clip(best_pos, -0.5, 0.5)
    
    return best_pos

if __name__ == "__main__":
    # Quick self-test
    t0 = time.time()
    res = construct()
    t1 = time.time()
    r = compute_radius(res)
    print(f"Time: {t1-t0:.2f}s, Radius: {r:.10f}")
