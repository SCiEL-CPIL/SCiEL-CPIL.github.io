"""Circle packing in perimeter-4 rectangle using greedy placement with local search."""
import numpy as np
import time

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    best_value = 0.0
    best_result = None
    
    # Search over aspect ratios
    aspect_ratios = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0]
    
    for ar in aspect_ratios:
        if time.time() > deadline:
            break
        
        # w + h = 2, h/w = ar => w(1+ar) = 2 => w = 2/(1+ar)
        w = 2.0 / (1.0 + ar)
        h = 2.0 - w
        
        # Try multiple random seeds for greedy placement
        for seed in range(5):
            if time.time() > deadline:
                break
            
            rng = np.random.default_rng(seed * 1000 + int(ar * 100))
            
            # Generate initial configuration using greedy placement
            centers, radii = _greedy_place(w, h, rng)
            
            if centers is None:
                continue
            
            value = _evaluate_fast(centers, radii, w, h)
            if value is not None and value > best_value:
                best_value = value
                best_result = np.column_stack([centers, radii])
            
            # Local search: perturb and re-optimize radii
            result = _local_search(centers, radii, w, h, rng, 
                                  time_budget=min(5.0, deadline - time.time()))
            
            if result is not None:
                val = _evaluate_fast(result[:, :2], result[:, 2], w, h)
                if val is not None and val > best_value:
                    best_value = val
                    best_result = result
    
    if best_result is None:
        # Fallback: simple grid in 1x1 rectangle
        w, h = 1.0, 1.0
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) < N:
                    x = (i + 0.5) / side * w
                    y = (j + 0.5) / side * h
                    r = min(w, h) / (2 * side) * 0.9
                    cells.append([x, y, r])
        return np.array(cells[:N])
    
    return best_result


def _greedy_place(w, h, rng):
    """Place circles greedily at maximal radius that fits."""
    centers = []
    radii = []
    
    # Create a grid of candidate positions
    n_grid = 20
    xs = np.linspace(0.05, w - 0.05, n_grid)
    ys = np.linspace(0.05, h - 0.05, n_grid)
    
    # Shuffle candidate positions
    candidates = []
    for x in xs:
        for y in ys:
            candidates.append((x, y))
    
    rng.shuffle(candidates)
    
    # Add center as a high-priority candidate
    candidates.insert(0, (w / 2, h / 2))
    
    for cx, cy in candidates:
        if len(centers) >= N:
            break
        
        # Find max radius that fits at this position
        max_r = min(cx, w - cx, cy, h - cy)
        
        # Check against existing circles
        for j, (ex, ey) in enumerate(centers):
            dist = np.sqrt((cx - ex)**2 + (cy - ey)**2)
            max_r = min(max_r, dist - radii[j] - 1e-9)
        
        if max_r > 0.01:
            centers.append([cx, cy])
            radii.append(max_r)
    
    if len(centers) < N:
        return None, None
    
    return np.array(centers[:N]), np.array(radii[:N])


def _local_search(centers, radii, w, h, rng, time_budget):
    """Local search to improve packing by adjusting positions and radii."""
    start_time = time.time()
    n = len(centers)
    
    cur_centers = centers.copy()
    cur_radii = radii.copy()
    cur_value = _evaluate_fast(cur_centers, cur_radii, w, h)
    
    if cur_value is None:
        return None
    
    best_centers = cur_centers.copy()
    best_radii = cur_radii.copy()
    best_value = cur_value
    
    # Simple hill climbing with random restarts
    for step in range(1000):
        if time.time() > start_time + time_budget:
            break
        
        # Perturb 1-2 circles
        n_perturb = rng.integers(1, 3)
        idx = rng.choice(n, size=n_perturb, replace=False)
        
        new_centers = cur_centers.copy()
        new_radii = cur_radii.copy()
        
        for i in idx:
            new_centers[i, 0] += rng.normal(0, 0.03)
            new_centers[i, 1] += rng.normal(0, 0.03)
            new_radii[i] += rng.normal(0, 0.01)
        
        # Clip to valid range
        for i in range(n):
            new_centers[i, 0] = np.clip(new_centers[i, 0], 0.01, w - 0.01)
            new_centers[i, 1] = np.clip(new_centers[i, 1], 0.01, h - 0.01)
            new_radii[i] = np.clip(new_radii[i], 0.01, min(w, h) / 2)
        
        # Try to increase radii where possible
        for i in range(n):
            max_r = min(new_centers[i, 0], w - new_centers[i, 0], 
                       new_centers[i, 1], h - new_centers[i, 1])
            for j in range(n):
                if i == j:
                    continue
                dist = np.sqrt((new_centers[i, 0] - new_centers[j, 0])**2 + 
                              (new_centers[i, 1] - new_centers[j, 1])**2)
                max_r = min(max_r, dist - new_radii[j] - 1e-9)
            new_radii[i] = max(new_radii[i], min(new_radii[i] * 1.02, max_r))
        
        new_value = _evaluate_fast(new_centers, new_radii, w, h)
        
        if new_value is not None and new_value > cur_value:
            cur_centers = new_centers
            cur_radii = new_radii
            cur_value = new_value
            
            if cur_value > best_value:
                best_value = cur_value
                best_centers = cur_centers.copy()
                best_radii = cur_radii.copy()
    
    if best_value <= 0:
        return None
    
    return np.column_stack([best_centers, best_radii])


def _evaluate_fast(centers, radii, w, h):
    """Vectorized evaluation. Returns sum of radii if valid, None if invalid."""
    n = len(centers)
    
    # Check all circles are inside rectangle
    if np.any(centers[:, 0] - radii < -1e-9) or np.any(centers[:, 0] + radii > w + 1e-9):
        return None
    if np.any(centers[:, 1] - radii < -1e-9) or np.any(centers[:, 1] + radii > h + 1e-9):
        return None
    if np.any(radii < 1e-6):
        return None
    
    # Check no overlaps using vectorized distance computation
    # Compute pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    # Get upper triangle
    iu = np.triu_indices(n, k=1)
    pair_dists = dists[iu]
    pair_radii_sum = radii[iu[0]] + radii[iu[1]]
    
    if np.any(pair_dists < pair_radii_sum - 1e-9):
        return None
    
    return np.sum(radii)
