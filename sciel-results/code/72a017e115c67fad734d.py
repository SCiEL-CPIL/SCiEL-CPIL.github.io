"""Optimized search for minimum maximal no-isosceles set on N x N grid."""
N = 16
import time
import random
import itertools
import numpy as np


def construct():
    start_time = time.time()
    deadline = start_time + 290  # safety margin before 300s limit
    
    # Precompute all points and index mapping
    all_points = list(itertools.product(range(N), repeat=2))
    n_pts = len(all_points)
    pt_to_idx = {p: i for i, p in enumerate(all_points)}
    
    # Precompute squared distances between all pairs
    coords = np.array(all_points, dtype=np.int32)
    dx = coords[:, 0:1] - coords[0:1, 0]
    dy = coords[:, 1:2] - coords[0:1, 1]
    dist2 = (dx * dx + dy * dy).astype(np.int32)  # shape (n_pts, n_pts)
    
    def _can_add(p_idx, chosen_mask):
        """Check if point p_idx can be added to the set indicated by chosen_mask."""
        if not np.any(chosen_mask):
            return True
        
        chosen_indices = np.where(chosen_mask)[0]
        if len(chosen_indices) < 2:
            return True
        
        d_to_p = dist2[p_idx, chosen_indices]
        d_chosen = dist2[np.ix_(chosen_indices, chosen_indices)]
        
        k = len(chosen_indices)
        i_idx, j_idx = np.triu_indices(k, k=1)
        if len(i_idx) == 0:
            return True
        
        d1 = d_to_p[i_idx]
        d2 = d_to_p[j_idx]
        d3 = d_chosen[i_idx, j_idx]
        
        if np.any(d1 == d2) or np.any(d1 == d3) or np.any(d2 == d3):
            return False
        return True
    
    def _greedy(order_indices, chosen_mask=None):
        if chosen_mask is None:
            chosen_mask = np.zeros(n_pts, dtype=bool)
        else:
            chosen_mask = chosen_mask.copy()
        
        for p_idx in order_indices:
            if not chosen_mask[p_idx]:
                if _can_add(p_idx, chosen_mask):
                    chosen_mask[p_idx] = True
        
        return chosen_mask
    
    def _construct_maximal(chosen_mask):
        chosen_mask = chosen_mask.copy()
        changed = True
        while changed:
            changed = False
            for p_idx in range(n_pts):
                if not chosen_mask[p_idx]:
                    if _can_add(p_idx, chosen_mask):
                        chosen_mask[p_idx] = True
                        changed = True
        return chosen_mask
    
    def _mask_to_list(chosen_mask):
        return [all_points[i] for i in np.where(chosen_mask)[0]]
    
    def _is_maximal(chosen_mask):
        for p_idx in range(n_pts):
            if not chosen_mask[p_idx]:
                if _can_add(p_idx, chosen_mask):
                    return False
        return True
    
    def _is_valid_no_iso(chosen_mask):
        chosen_indices = np.where(chosen_mask)[0]
        n = len(chosen_indices)
        if n < 3:
            return True
        
        d = dist2[np.ix_(chosen_indices, chosen_indices)]
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    d_ij = d[i, j]
                    d_ik = d[i, k]
                    d_jk = d[j, k]
                    if d_ij == d_ik or d_ij == d_jk or d_ik == d_jk:
                        return False
        return True
    
    def _remove_and_remaximize(chosen_mask, remove_idx):
        new_mask = chosen_mask.copy()
        new_mask[remove_idx] = False
        return _construct_maximal(new_mask)
    
    def _local_search(start_mask, max_iters=10000, deadline=None):
        current = start_mask.copy()
        best = start_mask.copy()
        best_len = np.sum(start_mask)
        
        for it in range(max_iters):
            if deadline is not None and time.time() > deadline:
                break
            improved = False
            current_indices = np.where(current)[0]
            for idx in current_indices:
                if deadline is not None and time.time() > deadline:
                    break
                extended = _remove_and_remaximize(current, idx)
                ext_len = np.sum(extended)
                if ext_len < np.sum(current):
                    current = extended
                    improved = True
                    break
            
            if not improved:
                break
            
            if np.sum(current) < best_len:
                best = current.copy()
                best_len = np.sum(current)
        
        return best
    
    def _simulated_annealing(start_mask, max_iters=20000, deadline=None):
        current = start_mask.copy()
        best = start_mask.copy()
        best_len = np.sum(start_mask)
        
        T = 2.0
        T_min = 0.01
        cooling = 0.999
        
        rng = random.Random(12345)
        
        for it in range(max_iters):
            if deadline is not None and time.time() > deadline:
                break
            
            if np.sum(current) < best_len:
                best = current.copy()
                best_len = np.sum(current)
            
            if T < T_min:
                break
            
            current_indices = np.where(current)[0]
            if len(current_indices) > 0:
                idx = current_indices[rng.randint(0, len(current_indices) - 1)]
                extended = _remove_and_remaximize(current, idx)
                
                delta = np.sum(extended) - np.sum(current)
                if delta <= 0 or rng.random() < (1.0 - delta) / T:
                    current = extended
        
            T *= cooling
        
        return best
    
    # Phase 1: Quick greedy constructions
    orderings = [
        list(range(n_pts)),
        sorted(range(n_pts), key=lambda i: (all_points[i][1], all_points[i][0])),
        sorted(range(n_pts), key=lambda i: (all_points[i][0] + all_points[i][1], all_points[i][0])),
        [i for i in range(n_pts) if (all_points[i][0] + all_points[i][1]) % 2 == 0] +
        [i for i in range(n_pts) if (all_points[i][0] + all_points[i][1]) % 2 == 1],
        sorted(range(n_pts), key=lambda i: (abs(all_points[i][0] - all_points[i][1]), all_points[i][0])),
    ]
    
    best_mask = None
    best_len = float('inf')
    
    for order in orderings:
        if time.time() > deadline:
            break
        chosen = _greedy(order)
        extended = _construct_maximal(chosen)
        ext_len = np.sum(extended)
        if ext_len < best_len:
            best_mask = extended
            best_len = ext_len
    
    # Phase 2: Local search
    if best_mask is not None and time.time() < deadline - 30:
        local_best = _local_search(best_mask, max_iters=10000, deadline=deadline - 30)
        if np.sum(local_best) < best_len:
            best_mask = local_best
            best_len = np.sum(best_mask)
    
    # Phase 3: Simulated annealing
    if best_mask is not None and time.time() < deadline - 15:
        sa_best = _simulated_annealing(best_mask, max_iters=15000, deadline=deadline - 15)
        if np.sum(sa_best) < best_len:
            best_mask = sa_best
            best_len = np.sum(best_mask)
    
    # Phase 4: Randomized restarts
    rng = random.Random(42)
    while time.time() < deadline - 5:
        order = list(range(n_pts))
        rng.shuffle(order)
        chosen = _greedy(order)
        extended = _construct_maximal(chosen)
        ext_len = np.sum(extended)
        
        if ext_len < best_len:
            best_mask = extended
            best_len = ext_len
        
        if np.sum(extended) < best_len + 2 and time.time() < deadline - 10:
            local = _local_search(extended, max_iters=2000, deadline=deadline - 10)
            if np.sum(local) < best_len:
                best_mask = local
                best_len = np.sum(best_mask)
    
    # Phase 5: Final polish
    if best_mask is not None and time.time() < deadline - 2:
        final = _local_search(best_mask, max_iters=5000, deadline=deadline - 2)
        if np.sum(final) < best_len:
            best_mask = final
    
    # Final verification and fallback
    if best_mask is None:
        best_mask = _construct_maximal(_greedy(list(range(n_pts))))
        
    if not _is_valid_no_iso(best_mask):
        best_mask = _construct_maximal(_greedy(list(range(n_pts))))
        
    if not _is_maximal(best_mask):
        best_mask = _construct_maximal(best_mask)
    
    return _mask_to_list(best_mask)
