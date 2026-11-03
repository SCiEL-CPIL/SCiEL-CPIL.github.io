import numpy as np
import time

N = 200

def _run_greedy(K=25, seed=0):
    np.random.seed(seed)
    
    lattice_coords = []
    for a in range(-K, K + 1):
        for b in range(-K, K + 1):
            if abs(a + b) <= K:
                lattice_coords.append((a, b))
    
    available = set(lattice_coords)
    
    def get_cartesian(axial):
        a, b = axial
        x = a + b / 2.0
        y = b * np.sqrt(3) / 2.0
        return x, y

    def compute_cost_from_stats(sum_x, sum_y, min_x, min_y, max_x, max_y, n, cand_x, cand_y):
        new_sum_x = sum_x + cand_x
        new_sum_y = sum_y + cand_y
        new_n = n + 1
        new_center_x = new_sum_x / new_n
        new_center_y = new_sum_y / new_n
        
        new_min_x = min(min_x, cand_x)
        new_max_x = max(max_x, cand_x)
        new_min_y = min(min_y, cand_y)
        new_max_y = max(max_y, cand_y)
        
        diff_x = max(new_max_x - new_center_x, new_center_x - new_min_x)
        diff_y = max(new_max_y - new_center_y, new_center_y - new_min_y)
        
        return max(diff_x, diff_y)

    current_set = {(0, 0)}
    available.remove((0, 0))
    
    x0, y0 = get_cartesian((0, 0))
    sum_x, sum_y = x0, y0
    min_x, min_y = x0, y0
    max_x, max_y = x0, y0
    n = 1
    
    neighbors_dirs = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]
    
    while n < N:
        candidates = set()
        for p in current_set:
            for d in neighbors_dirs:
                np_ = (p[0] + d[0], p[1] + d[1])
                if np_ in available:
                    candidates.add(np_)
        
        if not candidates:
            break
            
        best_candidate = None
        min_cost = np.inf
        
        # Add small random tie-breaking to explore different shapes
        for c in candidates:
            cx, cy = get_cartesian(c)
            cost = compute_cost_from_stats(sum_x, sum_y, min_x, min_y, max_x, max_y, n, cx, cy)
            # Add small noise
            cost += np.random.random() * 1e-6
            if cost < min_cost:
                min_cost = cost
                best_candidate = c
        
        if best_candidate is not None:
            current_set.add(best_candidate)
            available.remove(best_candidate)
            cx, cy = get_cartesian(best_candidate)
            sum_x += cx
            sum_y += cy
            min_x = min(min_x, cx)
            max_x = max(max_x, cx)
            min_y = min(min_y, cy)
            max_y = max(max_y, cy)
            n += 1
        else:
            break

    if n < N:
        return None, 0.0
        
    coords = np.array([get_cartesian(p) for p in current_set])
    center = np.mean(coords, axis=0)
    coords = coords - center
    
    max_x = np.max(np.abs(coords[:, 0]))
    max_y = np.max(np.abs(coords[:, 1]))
    max_extent = max(max_x, max_y)
    
    scale = 0.5 / max_extent
    coords = coords * scale
    
    # Calculate radius
    # Min distance between points
    # For a lattice, min dist is scale * 1.0 (if neighbors are included)
    # But we must check if any pair is closer due to scaling? No, scaling is uniform.
    # The min distance in lattice is 1.0.
    # So r = scale * 1.0 / 2.
    # However, we must ensure no points are closer than 1.0 in lattice.
    # By construction, we only add neighbors, so min dist is 1.0.
    # But wait, what if we skip a neighbor? No, we add points.
    # The min distance in the final set is at least 1.0?
    # Yes, because all points are from the lattice.
    # So r = scale / 2.
    
    radius = scale / 2.0
    return coords, radius

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    best_coords = None
    best_radius = 0.0
    
    # Run multiple restarts
    for i in range(20):
        if time.time() > deadline - 5:
            break
        coords, radius = _run_greedy(K=25, seed=i)
        if coords is not None and radius > best_radius:
            best_radius = radius
            best_coords = coords
            
    if best_coords is None:
        # Fallback
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
        
    return best_coords
