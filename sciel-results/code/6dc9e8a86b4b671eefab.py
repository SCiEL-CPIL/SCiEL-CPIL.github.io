import numpy as np
import itertools
import time

def construct():
    D = 11
    candidates = []
    
    # Type 1: two ±1's, rest 0 (norm^2 = 2)
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[i] = si
                v[j] = sj
                candidates.append(v)
                
    # Type 2: one ±2, rest 0 (norm^2 = 4)
    for i in range(D):
        for s in (1, -1):
            v = np.zeros(D, dtype=np.int64)
            v[i] = s * 2
            candidates.append(v)
            
    # Type 3: four ±1's, rest 0 (norm^2 = 4)
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for pos, sign in zip(positions, signs):
                v[pos] = sign
            candidates.append(v)
            
    candidates = np.array(candidates, dtype=np.int64)
    n = len(candidates)
    
    norms_sq = np.sum(candidates**2, axis=1)
    
    # Conflict condition for mixed set (max norm 2): dist_sq < 4
    dots = candidates @ candidates.T
    dist_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * dots
    conflict = (dist_sq < 4)
    np.fill_diagonal(conflict, False)
    
    # Baseline: All Type 1 vectors (indices 0 to 219)
    best_size = 220
    best_selected = list(range(220))
    
    start_time = time.time()
    deadline = start_time + 140.0
    
    np.random.seed(42)
    
    # Helper function: Greedy completion given a set of pre-selected nodes
    def greedy_completion(pre_selected_indices):
        selected_mask = np.zeros(n, dtype=bool)
        for idx in pre_selected_indices:
            selected_mask[idx] = True
            
        # Calculate blocked nodes
        if len(pre_selected_indices) > 0:
            pre_sel_arr = np.array(pre_selected_indices)
            blocked = conflict[:, pre_sel_arr].any(axis=1)
        else:
            blocked = np.zeros(n, dtype=bool)
            
        # Available nodes
        available = ~blocked & ~selected_mask
        
        # Greedy selection from available
        # We use min-degree heuristic on the available subgraph
        # To speed up, we can just use original degrees as a proxy, 
        # or compute degrees in available subgraph.
        # Computing degrees in available subgraph:
        # avail_indices = np.where(available)[0]
        # sub_conflict = conflict[np.ix_(avail_indices, avail_indices)]
        # degrees = sub_conflict.sum(axis=1)
        
        # This is O(N_avail^2). N_avail is large.
        # Let's use original degrees for speed, it's a good heuristic.
        orig_degrees = conflict.sum(axis=1)
        
        current_selected = list(pre_selected_indices)
        
        while available.any():
            avail_indices = np.where(available)[0]
            if len(avail_indices) == 0:
                break
            degs = orig_degrees[avail_indices]
            min_deg = np.min(degs)
            min_mask = degs == min_deg
            min_local = np.where(min_mask)[0]
            chosen_local = np.random.choice(min_local)
            chosen_global = avail_indices[chosen_local]
            
            current_selected.append(chosen_global)
            selected_mask[chosen_global] = True
            
            # Update available: remove chosen and its neighbors
            neigh = conflict[chosen_global]
            to_remove = neigh | selected_mask # Selected are already False in available, but neigh might include them
            # Actually, we just need to set available[chosen_global]=False and available[neigh]=False
            available[chosen_global] = False
            available[neigh] = False
            
        return current_selected

    n_attempts = 20
    for attempt in range(n_attempts):
        if time.time() > deadline:
            break
            
        # Initial Greedy
        remaining = np.ones(n, dtype=bool)
        degrees = conflict.sum(axis=1).copy()
        selected = []
        
        while remaining.any():
            rem_indices = np.where(remaining)[0]
            if len(rem_indices) == 0:
                break
            rem_degrees = degrees[rem_indices]
            min_deg = np.min(rem_degrees)
            min_mask = rem_degrees == min_deg
            min_indices_local = np.where(min_mask)[0]
            chosen_local = np.random.choice(min_indices_local)
            chosen_global = rem_indices[chosen_local]
            selected.append(chosen_global)
            
            neigh_mask = conflict[chosen_global]
            to_remove_mask = np.zeros(n, dtype=bool)
            to_remove_mask[chosen_global] = True
            to_remove_mask[neigh_mask] = True
            actual_remove = to_remove_mask & remaining
            if np.any(actual_remove):
                deg_decrease = conflict[actual_remove][:, remaining].sum(axis=0)
                degrees[remaining] -= deg_decrease
            remaining[actual_remove] = False
        
        current_selected = set(selected)
        current_size = len(current_selected)
        
        # Local Search: Perturb and Complete
        for pert in range(10):
            if time.time() > deadline:
                break
            
            # Perturb: Remove 5% of nodes
            num_remove = max(1, int(current_size * 0.05))
            to_remove = np.random.choice(list(current_selected), num_remove, replace=False)
            for idx in to_remove:
                current_selected.remove(idx)
                
            # Complete
            new_selected = greedy_completion(list(current_selected))
            new_size = len(new_selected)
            
            if new_size > current_size:
                current_selected = set(new_selected)
                current_size = new_size
        
        if current_size > best_size:
            best_size = current_size
            best_selected = list(current_selected)
            
    result = candidates[best_selected]
    return result
