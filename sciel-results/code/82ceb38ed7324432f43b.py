import numpy as np
import time

def kissing_number11() -> np.ndarray:
    """
    Finds a large set of integer points in Z^11 such that 
    min pairwise squared distance >= max squared norm.
    """
    start_time = time.time()
    deadline = 590.0
    
    d = 11
    
    # Fallback: Norm 2 points (220 points)
    def get_norm2_points():
        points = []
        for i in range(d):
            for j in range(i + 1, d):
                for si in [-1, 1]:
                    for sj in [-1, 1]:
                        p = [0] * d
                        p[i] = si
                        p[j] = sj
                        points.append(p)
        return np.array(points, dtype=np.int64)
    
    # Generate all norm-2 points
    points_list = []
    for i in range(d):
        for j in range(i + 1, d):
            for si in [-1, 1]:
                for sj in [-1, 1]:
                    p = [0] * d
                    p[i] = si
                    p[j] = sj
                    points_list.append(p)
    
    points = np.array(points_list, dtype=np.int64)
    n = len(points)  # 220
    
    # Compute dot products
    dot_prods = points @ points.T
    
    # conflict[i,j] = True if dot[i,j] >= 1 (incompatible, dist^2 <= 2)
    conflict = (dot_prods >= 1).astype(np.int8)
    np.fill_diagonal(conflict, 0)
    
    best_set = []
    best_size = 0
    
    search_deadline = deadline - 10.0
    np.random.seed(42)
    
    for attempt in range(10000):
        if time.time() > search_deadline:
            break
        
        # Random greedy
        order = np.random.permutation(n)
        selected = []
        in_set = np.zeros(n, dtype=bool)
        
        for idx in order:
            if not selected:
                compatible = True
            else:
                sel_arr = np.array(selected)
                compatible = conflict[idx, sel_arr].sum() == 0
            
            if compatible:
                selected.append(idx)
                in_set[idx] = True
        
        current_size = len(selected)
        
        # Try to add any remaining compatible node
        if current_size > 0:
            remaining = np.where(~in_set)[0]
            if len(remaining) > 0:
                sel_arr = np.array(selected)
                for u in remaining:
                    if conflict[u, sel_arr].sum() == 0:
                        selected.append(u)
                        in_set[u] = True
                        current_size += 1
        
        if current_size > best_size:
            best_size = current_size
            best_set = selected.copy()
        
        # Swap local search (limited)
        if current_size > 3 and attempt % 10 == 0:
            for i in range(min(len(selected), 20)):
                if time.time() > search_deadline:
                    break
                saved = selected[i]
                selected.pop(i)
                in_set[saved] = False
                
                added = []
                remaining = np.where(~in_set)[0]
                if len(remaining) > 0 and len(selected) > 0:
                    sel_arr = np.array(selected)
                    for u in remaining:
                        if conflict[u, sel_arr].sum() == 0:
                            selected.append(u)
                            in_set[u] = True
                            added.append(u)
                            sel_arr = np.append(sel_arr, u)
                
                if len(added) > 1:
                    current_size = len(selected)
                    if current_size > best_size:
                        best_size = current_size
                        best_set = selected.copy()
                else:
                    for u in added:
                        in_set[u] = False
                        selected.remove(u)
                    selected.insert(i, saved)
                    in_set[saved] = True
    
    # Verify best set
    if best_size > 2:
        clique_points = points[best_set]
        norms_sq = np.sum(clique_points**2, axis=1)
        max_norm_sq = int(np.max(norms_sq))
        
        min_dist_sq = float('inf')
        n_points = len(clique_points)
        for i in range(n_points):
            for j in range(i + 1, n_points):
                diff = clique_points[i] - clique_points[j]
                dist_sq = int(np.sum(diff**2))
                if dist_sq < min_dist_sq:
                    min_dist_sq = dist_sq
        
        if min_dist_sq >= max_norm_sq:
            return clique_points
    
    # Fallback
    return get_norm2_points()
