import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    D = 11
    
    # Type A: Vectors with one non-zero entry +/- 2
    vectors_A = []
    for i in range(D):
        v1 = [0] * D
        v1[i] = 2
        vectors_A.append(v1)
        v2 = [0] * D
        v2[i] = -2
        vectors_A.append(v2)
        
    vectors_A = np.array(vectors_A, dtype=np.int64)
    
    # Type D: Vectors with four non-zero entries +/- 1
    indices = list(range(D))
    supports = list(itertools.combinations(indices, 4))
    N_S = len(supports)
    
    # Build conflict graph vectorized
    # Represent each support as a bitmask
    support_masks = np.zeros(N_S, dtype=np.uint64)
    for i, s in enumerate(supports):
        mask = 0
        for idx in s:
            mask |= (1 << idx)
        support_masks[i] = mask
    
    # Conflict if intersection size >= 3
    # Use popcount on AND of masks
    conflict_supports = np.zeros((N_S, N_S), dtype=bool)
    # Vectorized: compute all pairwise ANDs and popcounts
    # For N_S=330, we can do this in blocks
    for i in range(N_S):
        if time.time() > deadline - 60:
            break
        ands = support_masks[i] & support_masks
        # Count bits using numpy
        # Use lookup or bit manipulation
        # For efficiency, use a simple loop for popcount since N_S is small
        inter_counts = np.array([bin(a).count('1') for a in ands], dtype=np.int32)
        conflict_supports[i, :] = inter_counts >= 3
        conflict_supports[:, i] = inter_counts >= 3
    
    # Symmetrize
    conflict_supports = np.maximum(conflict_supports, conflict_supports.T)
    np.fill_diagonal(conflict_supports, False)
    
    # Find maximum independent set using greedy + local search
    best_support_set = []
    best_size = 0
    
    rng = np.random.RandomState(42)
    
    num_restarts = 200
    
    for r in range(num_restarts):
        if time.time() > deadline - 45:
            break
            
        # Greedy initialization
        order = np.arange(N_S)
        rng.shuffle(order)
        
        in_set = np.zeros(N_S, dtype=bool)
        blocked = np.zeros(N_S, dtype=bool)
        
        for v in order:
            if not blocked[v] and not in_set[v]:
                in_set[v] = True
                blocked |= conflict_supports[v]
                
        current_size = np.sum(in_set)
        
        # Local search: try swaps to improve
        improved = True
        while improved:
            if time.time() > deadline - 40:
                break
            improved = False
            # Try removing one node from set and adding two non-conflicting nodes not in set
            in_set_indices = np.where(in_set)[0]
            for v_remove in in_set_indices:
                if time.time() > deadline - 40:
                    break
                # Temporarily remove v_remove
                in_set[v_remove] = False
                # Recompute blocked
                blocked = np.zeros(N_S, dtype=bool)
                for v in np.where(in_set)[0]:
                    blocked |= conflict_supports[v]
                
                # Find nodes not in set and not blocked
                candidates = ~in_set & ~blocked
                cand_indices = np.where(candidates)[0]
                
                # Try to add up to 2 candidates
                added = 0
                for c in cand_indices:
                    if added >= 2:
                        break
                    if not blocked[c]:
                        in_set[c] = True
                        blocked |= conflict_supports[c]
                        added += 1
                
                new_size = np.sum(in_set)
                if new_size > current_size:
                    current_size = new_size
                    improved = True
                else:
                    # Revert
                    in_set[v_remove] = True
                    in_set[candidates] = False  # This is wrong, need to revert properly
                    # Let's just break out and not do complex reverts
                    break
        
        if current_size > best_size:
            best_size = current_size
            best_support_set = np.where(in_set)[0].tolist()
    
    # Generate Type D vectors for the selected supports
    selected_vectors_D = []
    for s_idx in best_support_set:
        support = supports[s_idx]
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, sign in zip(support, signs):
                v[idx] = sign
            selected_vectors_D.append(v)
            
    selected_vectors_D = np.array(selected_vectors_D, dtype=np.int64)
    
    # Greedy fill: try to add more Type D vectors from non-selected supports
    current_set_vectors = list(selected_vectors_D)
    current_set_vectors_set = set(map(tuple, current_set_vectors))
    
    # Generate ALL Type D vectors
    all_vectors_D = []
    for s in supports:
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, sign in zip(supports[0] and s or s, signs):
                v[idx] = sign
            all_vectors_D.append(v)
            
    all_vectors_D = np.array(all_vectors_D, dtype=np.int64)
    N_all_D = len(all_vectors_D)
    
    indices_all = np.arange(N_all_D)
    rng.shuffle(indices_all)
    
    # Vectorized greedy fill
    for i in indices_all:
        if time.time() > deadline - 15:
            break
            
        v = all_vectors_D[i]
        
        if tuple(v) in current_set_vectors_set:
            continue
            
        # Check compatibility: dot(v, u) <= 2 for all u in set
        # dot^2 = 4 + 4 - 2*dot >= 4 => dot <= 2
        if len(current_set_vectors) > 0:
            current_arr = np.array(current_set_vectors)
            dots = current_arr @ v
            if np.any(dots >= 3):
                continue
        
        current_set_vectors.append(v)
        current_set_vectors_set.add(tuple(v))
        
    final_vectors_D = np.array(current_set_vectors, dtype=np.int64)
    
    # Try adding Type B vectors (two non-zero ±1, norm²=2)
    # For Type B with norm²=2 and Type A/D with norm²=4:
    # dist² = 2 + 4 - 2*dot >= 4 => dot <= 1
    # Type B vs Type B: dist² = 4 - 2*dot >= 4 => dot <= 0 (supports disjoint or orthogonal)
    
    vectors_B_candidates = []
    for i in range(D):
        for j in range(i+1, D):
            for s1 in [1, -1]:
                for s2 in [1, -1]:
                    v = [0] * D
                    v[i] = s1
                    v[j] = s2
                    vectors_B_candidates.append(v)
    
    vectors_B_candidates = np.array(vectors_B_candidates, dtype=np.int64)
    
    added_B = []
    added_B_set = set()
    
    for v in vectors_B_candidates:
        if time.time() > deadline - 5:
            break
        vt = tuple(v)
        if vt in added_B_set:
            continue
        
        can_add = True
        # Check against Type A
        for u in vectors_A:
            if np.dot(v, u) >= 2:  # dot <= 1 required
                can_add = False
                break
        if not can_add:
            continue
        
        # Check against Type D
        for u in final_vectors_D:
            if np.dot(v, u) >= 2:
                can_add = False
                break
        if not can_add:
            continue
        
        # Check against other Type B
        for u in added_B:
            if np.dot(v, u) != 0:  # dot <= 0, and since integer, dot=0
                can_add = False
                break
        if not can_add:
            continue
        
        added_B.append(v)
        added_B_set.add(vt)
    
    if added_B:
        added_B = np.array(added_B, dtype=np.int64)
        final_vectors = np.vstack([vectors_A, final_vectors_D, added_B])
    else:
        final_vectors = np.vstack([vectors_A, final_vectors_D])
    
    return final_vectors
