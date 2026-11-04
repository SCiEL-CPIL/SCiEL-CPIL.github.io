import numpy as np
import itertools
import time
import random

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

    # Build conflict graph: conflict if intersection size >= 3
    # Represent supports as bitmasks for fast intersection check
    support_masks = []
    for s in supports:
        mask = 0
        for i in s:
            mask |= (1 << i)
        support_masks.append(mask)
    support_masks = np.array(support_masks, dtype=np.int64)

    # Precompute conflict matrix using vectorized bit operations
    # We want to count bits in (mask_i & mask_j)
    # N_S is 330. 330x330 matrix is small.
    # Use a lookup table for bit counting on 11-bit values
    bit_count_lut = np.array([bin(i).count('1') for i in range(2048)], dtype=np.int8)
    
    # Compute all pairwise intersections and count bits
    # support_masks shape: (N_S,)
    # We need (N_S, N_S) matrix of bit counts of (mask_i & mask_j)
    
    # Vectorized approach: for each i, compute inter = support_masks[i] & support_masks
    # Then use LUT to count bits
    conflict = np.zeros((N_S, N_S), dtype=bool)
    for i in range(N_S):
        inter = support_masks[i] & support_masks
        counts = bit_count_lut[inter]
        conflict[i, :] = counts >= 3
        conflict[:, i] = conflict[i, :]

    rng = np.random.RandomState(42)
    py_rng = random.Random(42)

    best_support_set = set()
    best_size = 0

    # Fast Local Search for MIS
    def greedy_init(r):
        order = np.arange(N_S)
        rng.shuffle(order)
        x = np.zeros(N_S, dtype=bool)
        blocked = np.zeros(N_S, dtype=bool)
        for v in order:
            if not blocked[v] and not x[v]:
                x[v] = True
                blocked |= conflict[v]
        return x

    def local_search(x, num_iters=1000):
        current_size = np.sum(x)
        in_set = np.where(x)[0].tolist()
        
        # Precompute blocked set
        if len(in_set) > 0:
            blocked = np.zeros(N_S, dtype=bool)
            blocked[np.array(in_set)] = True
            for v in in_set:
                blocked[conflict[v]] = True
        else:
            blocked = np.zeros(N_S, dtype=bool)
        
        for it in range(num_iters):
            if time.time() > deadline - 15:
                break
            
            # Try to add a node
            candidates_add = np.where(~x & ~blocked)[0]
            
            if len(candidates_add) > 0:
                # Randomly pick one to add
                v = py_rng.choice(candidates_add)
                x[v] = True
                current_size += 1
                in_set.append(v)
                blocked[conflict[v]] = True
            else:
                # Try swap: remove one from in_set, add one that was blocked only by it
                if len(in_set) > 0:
                    # For each candidate in in_set, see if removing it unlocks any new additions
                    # This is expensive, so only do it occasionally
                    if it % 10 == 0 and len(in_set) > 0:
                        v_remove = py_rng.choice(in_set)
                        # Temporarily remove
                        x[v_remove] = False
                        in_set.remove(v_remove)
                        current_size -= 1
                        
                        # Recompute blocked
                        blocked_new = np.zeros(N_S, dtype=bool)
                        for v in in_set:
                            blocked_new[conflict[v]] = True
                        
                        candidates_after_remove = np.where(~x & ~blocked_new)[0]
                        
                        if len(candidates_after_remove) > 0:
                            # Add a random candidate
                            v_add = py_rng.choice(candidates_after_remove)
                            x[v_add] = True
                            in_set.append(v_add)
                            current_size += 1
                            blocked = blocked_new.copy()
                            blocked[conflict[v_add]] = True
                        else:
                            # Revert
                            x[v_remove] = True
                            in_set.append(v_remove)
                            current_size += 1
                            blocked = blocked_new.copy()
                            blocked[conflict[v_remove]] = True
                    else:
                        # Just remove a random node
                        if len(in_set) > 0:
                            v = py_rng.choice(in_set)
                            x[v] = False
                            in_set.remove(v)
                            current_size -= 1
                            # Recompute blocked
                            blocked = np.zeros(N_S, dtype=bool)
                            for v2 in in_set:
                                blocked[conflict[v2]] = True
                        else:
                            break
                else:
                    break
        
        return x, current_size

    for restart in range(200):
        if time.time() > deadline - 15:
            break

        x = greedy_init(restart)
        x, current_size = local_search(x, num_iters=500)
        
        if current_size > best_size:
            best_size = current_size
            best_support_set = set(np.where(x)[0].tolist())

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

    # Combine A and D
    current_set_vectors = list(vectors_A) + list(selected_vectors_D)
    current_set_vectors_set = set(map(tuple, current_set_vectors))
    
    # Current Max Norm is 2.
    # We want to add Type B vectors (two +/- 2s) with norm sqrt(8).
    # If we add them, Max Norm becomes sqrt(8).
    # We must ensure all pairwise distances >= sqrt(8).
    
    # Generate Type B vectors
    type_B_candidates = []
    for i in range(D):
        for j in range(i+1, D):
            for s1 in [2, -2]:
                for s2 in [2, -2]:
                    v = [0] * D
                    v[i] = s1
                    v[j] = s2
                    type_B_candidates.append(v)
    
    type_B_candidates = np.array(type_B_candidates, dtype=np.int64)
    
    # Greedily add Type B vectors
    # Check compatibility with existing set (norms 2) and new Type B vectors (norm sqrt(8))
    # Condition for u (norm 2) and b (norm sqrt(8)): |u-b|^2 >= 8 => 4+8-2u.b >= 8 => u.b <= 2
    # Condition for b1, b2 (norm sqrt(8)): |b1-b2|^2 >= 8 => 8+8-2b1.b2 >= 8 => b1.b2 <= 4
    
    added_B = []
    added_B_set = set()
    
    # Convert current set to numpy for vectorized dot products
    current_set_np = np.array(current_set_vectors, dtype=np.int64)
    
    # Sort candidates to prioritize those with fewer conflicts
    # Compute a heuristic score for each candidate
    indices_B = np.arange(len(type_B_candidates))
    rng.shuffle(indices_B)
    
    for idx in indices_B:
        if time.time() > deadline - 3:
            break
            
        v = type_B_candidates[idx]
        v_tuple = tuple(v)
        if v_tuple in added_B_set:
            continue
            
        # Vectorized check against existing set (A and D)
        dots = current_set_np @ v
        if np.any(dots > 2):
            continue
        
        # Check against added B
        if added_B:
            added_B_np = np.array(added_B, dtype=np.int64)
            dots_b = added_B_np @ v
            if np.any(dots_b > 4):
                continue
        
        added_B.append(v)
        added_B_set.add(v_tuple)
    
    if added_B:
        final_vectors = np.vstack([vectors_A, selected_vectors_D, np.array(added_B, dtype=np.int64)])
    else:
        final_vectors = np.vstack([vectors_A, selected_vectors_D])
    
    return final_vectors
