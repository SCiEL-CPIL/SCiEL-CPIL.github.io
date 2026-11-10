import numpy as np
import itertools
import time

def no_5_on_a_sphere8() -> np.ndarray:
    n = 8
    deadline = time.time() + 570.0
    
    grid = np.array(list(itertools.product(range(n), repeat=3)), dtype=np.int64)
    num_points = len(grid)
    
    def can_add_point(arr, p):
        m = len(arr)
        if m < 4:
            return True
        
        diff = arr - p
        sq_diff = (arr ** 2).sum(axis=1) - (p ** 2).sum()
        w = np.column_stack((diff, sq_diff))
        
        if m == 4:
            d = np.linalg.det(w)
            return abs(d) >= 0.5
        
        # Vectorized check for all 4-combinations
        indices = range(m)
        combos = list(itertools.combinations(indices, 4))
        if not combos:
            return True
        combo_array = np.array(combos)
        sub_w = w[combo_array]
        dets = np.linalg.det(sub_w)
        return not np.any(np.abs(dets) < 0.5)

    def greedy_fill(order):
        arr = np.empty((0, 3), dtype=np.int64)
        for idx in order:
            if time.time() > deadline:
                break
            p = grid[idx]
            if can_add_point(arr, p):
                arr = np.vstack([arr, p]) if len(arr) > 0 else p[None, :]
        return arr

    def local_search(points):
        current = points.copy()
        current_set = set(map(tuple, current))
        
        # Precompute indices for current points
        curr_indices = set()
        for p in current:
            # Find index in grid
            # Since grid is unique, we can use a dictionary or search
            # For small grid, search is ok, but let's use a map for speed
            pass
        # Better: maintain indices
        curr_idx_list = []
        for p in current:
            # Linear search in grid
            idx = np.where((grid == p).all(axis=1))[0][0]
            curr_idx_list.append(idx)
        curr_idx_set = set(curr_idx_list)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try to add any point from outside
            # Iterate over all grid points not in current
            for idx in range(num_points):
                if time.time() > deadline:
                    return current
                if idx in curr_idx_set:
                    continue
                
                p = grid[idx]
                if can_add_point(current, p):
                    current = np.vstack([current, p])
                    curr_idx_set.add(idx)
                    improved = True
                    break # Restart loop with new set
            
        return current

    # Diverse Orderings
    rng = np.random.RandomState(0)
    orderings = []
    
    # 1. Random
    orderings.append(rng.permutation(num_points))
    
    # 2. Manhattan Distance from Center
    center = np.array([3.5, 3.5, 3.5])
    manhattan_dists = np.sum(np.abs(grid - center), axis=1)
    orderings.append(np.argsort(manhattan_dists))
    orderings.append(np.argsort(-manhattan_dists))
    
    # 3. Coordinate Sum
    coord_sums = grid.sum(axis=1)
    orderings.append(np.argsort(coord_sums))
    orderings.append(np.argsort(-coord_sums))
    
    # 4. Corner Bias
    corners = np.array([[0,0,0], [7,7,7], [0,7,0], [7,0,0], [0,0,7], [7,7,0], [0,7,7], [7,0,7]])
    min_corner_dist = np.min([np.sum((grid - c)**2, axis=1) for c in corners], axis=0)
    orderings.append(np.argsort(min_corner_dist))
    orderings.append(np.argsort(-min_corner_dist))
    
    # 5. More Randoms
    for _ in range(5):
        orderings.append(rng.permutation(num_points))

    best_points = None
    best_len = 0
    
    for order in orderings:
        if time.time() > deadline:
            break
            
        points = greedy_fill(order)
        points = local_search(points)
        
        if len(points) > best_len:
            best_len = len(points)
            best_points = points

    if best_points is None:
        return grid[:4]
        
    return best_points.astype(np.int64)
