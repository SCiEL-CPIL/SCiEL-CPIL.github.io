import time
import random
import numpy as np

N = 64
MAX_DIST_SQ = 2 * (N - 1) ** 2

def construct():
    start_time = time.time()
    deadline = start_time + 270.0
    
    all_points = [(x, y) for x in range(N) for y in range(N)]
    M = len(all_points)
    
    # Precompute distance matrix
    coords = np.array(all_points)
    x_coords = coords[:, 0]
    y_coords = coords[:, 1]
    
    dx = x_coords[:, np.newaxis] - x_coords[np.newaxis, :]
    dy = y_coords[:, np.newaxis] - y_coords[np.newaxis, :]
    D = (dx * dx + dy * dy).astype(np.uint16)
    
    best_set = []
    best_size = 0
    
    def is_valid_set(indices):
        # Check if a given set of indices is valid
        # This is for verification, not for search
        if len(indices) < 3:
            return True
        S = np.array(indices)
        for i in range(len(S)):
            for j in range(i + 1, len(S)):
                for k in range(j + 1, len(S)):
                    d1 = D[S[i], S[j]]
                    d2 = D[S[j], S[k]]
                    d3 = D[S[k], S[i]]
                    if d1 == d2 or d2 == d3 or d3 == d1:
                        return False
        return True

    def greedy_construct(seed):
        random.seed(seed)
        current_set = []
        in_set = np.zeros(M, dtype=bool)
        used_dist = np.zeros(MAX_DIST_SQ + 1, dtype=bool)
        
        remaining = list(range(M))
        random.shuffle(remaining)
        
        for p_idx in remaining:
            if time.time() > deadline:
                break
            
            # Check if p_idx can be added
            if not current_set:
                current_set.append(p_idx)
                in_set[p_idx] = True
                continue
            
            # Get distances from p_idx to all points in current_set
            S_arr = np.array(current_set)
            d_vals = D[p_idx, S_arr]
            
            # Check for duplicates in d_vals
            # and check if any d_val is in used_dist
            
            # Fast check:
            # 1. Are all d_vals unique?
            # 2. Are none of d_vals in used_dist?
            
            if len(np.unique(d_vals)) != len(d_vals):
                continue
                
            if np.any(used_dist[d_vals]):
                continue
                
            # Valid, add it
            current_set.append(p_idx)
            in_set[p_idx] = True
            
            # Update used_dist
            # Distances between p_idx and all existing points are now "used"
            # Also, distances between existing points were already used.
            # Wait, used_dist contains distances between ALL pairs in S.
            # When adding p_idx, we add distances d(p_idx, q) for all q in S.
            used_dist[d_vals] = True
            
        return current_set

    def local_search(initial_set, max_iterations=1000):
        current_set = list(initial_set)
        in_set = np.zeros(M, dtype=bool)
        for idx in current_set:
            in_set[idx] = True
            
        used_dist = np.zeros(MAX_DIST_SQ + 1, dtype=bool)
        # Rebuild used_dist
        S_arr = np.array(current_set)
        if len(S_arr) > 1:
            for i in range(len(S_arr)):
                for j in range(i + 1, len(S_arr)):
                    used_dist[D[S_arr[i], S_arr[j]]] = True
        
        best_set = list(current_set)
        best_size = len(current_set)
        
        for _ in range(max_iterations):
            if time.time() > deadline:
                break
            
            S_arr = np.array(current_set)
            C_arr = np.array([i for i in range(M) if not in_set[i]])
            
            # Try to add a point
            if len(C_arr) > 0:
                # Vectorized check for all candidates
                d_cs = D[C_arr][:, S_arr] # Shape (|C|, |S|)
                
                # Check unique in each row
                # This is tricky to vectorize perfectly without loops or sorting
                # But |S| is small.
                # Let's just loop over candidates if |C| is small, or use a simple check.
                # For |S| ~ 100, sorting 100 elements is fast.
                
                valid_mask = np.zeros(len(C_arr), dtype=bool)
                for i, c_idx in enumerate(C_arr):
                    d_vals = d_cs[i]
                    if len(np.unique(d_vals)) != len(d_vals):
                        continue
                    if np.any(used_dist[d_vals]):
                        continue
                    valid_mask[i] = True
                    
                if np.any(valid_mask):
                    # Pick a random valid candidate
                    valid_indices = np.where(valid_mask)[0]
                    chosen_local = random.choice(valid_indices)
                    chosen_global = C_arr[chosen_local]
                    
                    # Add it
                    current_set.append(chosen_global)
                    in_set[chosen_global] = True
                    d_vals = D[chosen_global, S_arr]
                    used_dist[d_vals] = True
                    
                    if len(current_set) > best_size:
                        best_set = list(current_set)
                        best_size = len(current_set)
                        # Reset S_arr for next iteration? It's updated at start of loop
                else:
                    # Try a swap
                    # Pick a random point to remove
                    if len(current_set) > 0:
                        rem_idx = random.choice(current_set)
                        # Remove rem_idx
                        current_set.remove(rem_idx)
                        in_set[rem_idx] = False
                        
                        # Update used_dist
                        # This is hard to do efficiently without rebuilding.
                        # Rebuild used_dist for the new set
                        S_new = np.array(current_set)
                        used_dist[:] = False
                        if len(S_new) > 1:
                            for i in range(len(S_new)):
                                for j in range(i + 1, len(S_new)):
                                    used_dist[D[S_new[i], S_new[j]]] = True
                        
                        # Try to add a candidate
                        C_new = np.array([i for i in range(M) if not in_set[i]])
                        if len(C_new) > 0:
                            d_cs_new = D[C_new][:, S_new]
                            added = False
                            for i, c_idx in enumerate(C_new):
                                d_vals = d_cs_new[i]
                                if len(np.unique(d_vals)) != len(d_vals):
                                    continue
                                if np.any(used_dist[d_vals]):
                                    continue
                                
                                # Add it
                                current_set.append(c_idx)
                                in_set[c_idx] = True
                                used_dist[d_vals] = True
                                added = True
                                
                                if len(current_set) > best_size:
                                    best_set = list(current_set)
                                    best_size = len(current_set)
                                break
                            
                            if not added:
                                # Revert
                                current_set.append(rem_idx)
                                in_set[rem_idx] = True
                                # Rebuild used_dist again? Or just keep the state?
                                # We modified used_dist. We should revert it.
                                # Rebuilding is safest.
                                S_revert = np.array(current_set)
                                used_dist[:] = False
                                if len(S_revert) > 1:
                                    for i in range(len(S_revert)):
                                        for j in range(i + 1, len(S_revert)):
                                            used_dist[D[S_revert[i], S_revert[j]]] = True
        
        return best_set

    # Run multiple restarts
    num_restarts = 10
    for i in range(num_restarts):
        if time.time() > deadline:
            break
        random.seed(i * 1000 + 42)
        g_set = greedy_construct(i)
        if len(g_set) > best_size:
            best_set = g_set
            best_size = len(g_set)
        
        if time.time() < deadline - 30:
            l_set = local_search(g_set, max_iterations=500)
            if len(l_set) > best_size:
                best_set = l_set
                best_size = len(l_set)
                
    return [all_points[i] for i in best_set]
