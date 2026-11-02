import numpy as np
import itertools
import time

D = 11

def _generate_all_vectors(max_M=6):
    """Generate all nonzero integer vectors in R^D with norm squared <= max_M."""
    vectors = set()
    
    for M in range(1, max_M + 1):
        for w in range(1, min(D, M) + 1):
            # Generate all ways to write M as sum of w squares
            def gen_sq_list(target, max_sq, count):
                if count == 0:
                    if target == 0:
                        yield []
                    return
                if target < 0:
                    return
                v = 1
                while v * v <= target:
                    for rest in gen_sq_list(target - v * v, v, count - 1):
                        yield [v * v] + rest
                    v += 1
            
            patterns = set()
            for sq_list in gen_sq_list(M, int(np.sqrt(M)), w):
                patterns.add(tuple(sorted(sq_list)))
            
            for pat in patterns:
                mags = [int(np.sqrt(x)) for x in pat]
                for indices in itertools.combinations(range(D), w):
                    for perm in set(itertools.permutations(mags)):
                        for signs in itertools.product([1, -1], repeat=w):
                            v = [0] * D
                            for i, idx in enumerate(indices):
                                v[idx] = perm[i] * signs[i]
                            vectors.add(tuple(v))
    
    return list(vectors)

def _greedy_construct(vectors, max_M):
    """Greedy construction with conflict checking."""
    if not vectors:
        return np.array([], dtype=np.int64).reshape(0, D)
    
    np.random.shuffle(vectors)
    
    selected = []
    selected_arr = np.array([], dtype=np.int64).reshape(0, D)
    
    for v in vectors:
        v_arr = np.array(v, dtype=np.int64)
        M = int(np.dot(v_arr, v_arr))
        
        if not selected:
            selected.append(v)
            selected_arr = v_arr.reshape(1, -1)
            continue
        
        # Check conflict: for each selected vector u, need ||v-u||^2 >= max(M_v, M_u)
        # ||v-u||^2 = ||v||^2 + ||u||^2 - 2*v·u >= max(||v||^2, ||u||^2)
        # So: min(||v||^2, ||u||^2) - 2*v·u >= 0
        # i.e., 2*v·u <= min(||v||^2, ||u||^2)
        
        dots = selected_arr @ v_arr
        norms_sq = np.sum(selected_arr ** 2, axis=1)
        min_norms = np.minimum(norms_sq, M)
        
        if np.all(2 * dots <= min_norms):
            selected.append(v)
            selected_arr = np.vstack((selected_arr, v_arr.reshape(1, -1)))
    
    return np.array(selected, dtype=np.int64)

def _local_search(selected_arr, all_vectors, max_iter=5000):
    """Local search: try to swap one selected vector for two non-conflicting ones."""
    if len(selected_arr) == 0:
        return selected_arr
    
    selected_set = set(map(tuple, selected_arr))
    current_size = len(selected_arr)
    improved = True
    iterations = 0
    
    while improved and iterations < max_iter:
        improved = False
        iterations += 1
        
        # Try each pair of non-selected vectors that are mutually compatible
        # and check if removing one selected vector allows adding both
        candidate_indices = []
        for i, v in enumerate(all_vectors):
            if tuple(v) not in selected_set:
                candidate_indices.append(i)
        
        np.random.shuffle(candidate_indices)
        
        for ci in candidate_indices[:200]:  # Limit per iteration
            v1 = all_vectors[ci]
            v1_arr = np.array(v1, dtype=np.int64)
            M1 = int(np.dot(v1_arr, v1_arr))
            
            # Check v1 is compatible with all selected except possibly one
            conflicts_with_v1 = []
            for si, sv in enumerate(selected_arr):
                dot = int(np.dot(v1_arr, sv))
                Msq = int(np.dot(sv, sv))
                min_norm = min(M1, Msq)
                if 2 * dot > min_norm:
                    conflicts_with_v1.append(si)
            
            if len(conflicts_with_v1) > 1:
                continue
            
            # Try to find a second vector v2 that is compatible with v1 and all selected
            # except for the same conflict(s)
            for cj in candidate_indices[:200]:
                if cj == ci:
                    continue
                v2 = all_vectors[cj]
                v2_arr = np.array(v2, dtype=np.int64)
                M2 = int(np.dot(v2_arr, v2_arr))
                
                # Check v1-v2 compatibility
                dot12 = int(np.dot(v1_arr, v2_arr))
                min12 = min(M1, M2)
                if 2 * dot12 > min12:
                    continue
                
                # Check v2 conflicts with selected
                conflicts_with_v2 = []
                for si, sv in enumerate(selected_arr):
                    dot = int(np.dot(v2_arr, sv))
                    Msq = int(np.dot(sv, sv))
                    min_norm = min(M2, Msq)
                    if 2 * dot > min_norm:
                        conflicts_with_v2.append(si)
                
                # The union of conflicts must be a single vector we can remove
                all_conflicts = set(conflicts_with_v1) | set(conflicts_with_v2)
                
                if len(all_conflicts) <= 1:
                    # We can remove at most one vector and add two
                    if len(all_conflicts) == 1:
                        remove_idx = list(all_conflicts)[0]
                        # Verify: after removing selected[remove_idx], both v1 and v2 fit
                        new_selected = list(selected_arr)
                        new_selected.pop(remove_idx)
                        
                        # Re-verify v1 and v2 against new_selected
                        ok = True
                        for v_new in [v1_arr, v2_arr]:
                            M_new = int(np.dot(v_new, v_new))
                            for sv in new_selected:
                                dot = int(np.dot(v_new, sv))
                                Msq = int(np.dot(sv, sv))
                                min_norm = min(M_new, Msq)
                                if 2 * dot > min_norm:
                                    ok = False
                                    break
                            if not ok:
                                break
                        
                        if ok:
                            # Perform the swap
                            selected_arr = np.array(new_selected + [v1, v2], dtype=np.int64)
                            selected_set = set(map(tuple, selected_arr))
                            current_size = len(selected_arr)
                            improved = True
                            break
                    elif len(all_conflicts) == 0:
                        # Both v1 and v2 fit without removing anything
                        selected_arr = np.vstack((selected_arr, v1_arr, v2_arr))
                        selected_set = set(map(tuple, selected_arr))
                        current_size = len(selected_arr)
                        improved = True
                        break
                
                if improved:
                    break
            
            if improved:
                break
    
    return selected_arr

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    best_vectors = None
    best_k = 0
    
    # Generate all candidate vectors
    all_vectors = _generate_all_vectors(max_M=6)
    
    # Initial greedy construction
    initial = _greedy_construct(all_vectors, 6)
    best_vectors = initial
    best_k = len(initial)
    
    # Local search with multiple restarts
    restarts = 0
    while time.time() < deadline and restarts < 20:
        restarts += 1
        
        # Shuffle and re-greedy
        shuffled = list(all_vectors)
        np.random.shuffle(shuffled)
        candidate = _greedy_construct(shuffled, 6)
        
        if len(candidate) > best_k:
            best_vectors = candidate
            best_k = len(candidate)
        
        # Local search on the candidate
        if time.time() < deadline - 30:
            candidate = _local_search(candidate, all_vectors, max_iter=3000)
            if len(candidate) > best_k:
                best_vectors = candidate
                best_k = len(candidate)
    
    return best_vectors
