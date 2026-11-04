import numpy as np
import itertools
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

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
    # We can loop over i and vectorize j.
    conflict = np.zeros((N_S, N_S), dtype=bool)
    for i in range(N_S):
        # Calculate intersection with all j
        inter = support_masks[i] & support_masks
        # Count bits: bin(x).count('1') is slow.
        # Use lookup table or np.unpackbits?
        # For 11 bits, we can just use a small LUT or loop.
        # Since N_S is small, a simple loop with np.bit_count (if available) or bin count is fine.
        # Python 3.10+ has int.bit_count().
        # Let's use a vectorized approach with unpackbits for speed if possible, or just loop.
        # Actually, 330 * 330 = 108,900 operations. Even with Python loop, it's fast.
        # But let's try to vectorize the bit count.
        # np.unpackbits(inter.astype(np.uint16)) -> 16 bits per uint16.
        # Simpler: just use a list comprehension with bit_count.
        counts = np.array([bin(x).count('1') for x in inter])
        conflict[i, :] = counts >= 3
        # Symmetry
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

    def local_search(x, num_iters=500):
        current_size = np.sum(x)
        for _ in range(num_iters):
            # Try to add a node
            in_set = np.where(x)[0]
            if len(in_set) > 0:
                blocked = np.zeros(N_S, dtype=bool)
                # Vectorized OR of conflict rows
                blocked |= conflict[in_set].any(axis=0)
            else:
                blocked = np.zeros(N_S, dtype=bool)
            
            candidates_add = np.where(~x & ~blocked)[0]
            candidates_remove = in_set
            
            if len(candidates_add) > 0:
                # Randomly pick one to add
                v = py_rng.choice(candidates_add)
                x[v] = True
                current_size += 1
            elif len(candidates_remove) > 0:
                # Randomly pick one to remove
                v = py_rng.choice(candidates_remove)
                x[v] = False
                current_size -= 1
            else:
                break
        
        return x, current_size

    for restart in range(100):
        if time.time() > deadline - 20:
            break

        x = greedy_init(restart)
        x, current_size = local_search(x, num_iters=200)
        
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
    
    # Sort candidates to prioritize those with fewer conflicts? 
    # Or just random order.
    indices_B = np.arange(len(type_B_candidates))
    rng.shuffle(indices_B)
    
    for idx in indices_B:
        if time.time() > deadline - 5:
            break
            
        v = type_B_candidates[idx]
        v_tuple = tuple(v)
        if v_tuple in added_B_set:
            continue
            
        # Check against existing set (A and D)
        # We can vectorize this check against the numpy array of current vectors
        # But current_set_vectors is a list. Let's convert to numpy once.
        # Actually, we can just loop. The set is ~250 vectors. 220 candidates. 250*220 = 55k dot products. Fast.
        
        can_add = True
        
        # Check against A and D
        for u in current_set_vectors:
            dot = np.dot(v, u)
            if dot > 2: # dot must be <= 2
                can_add = False
                break
        
        if not can_add:
            continue
            
        # Check against added B
        for b in added_B:
            dot = np.dot(v, b)
            if dot > 4: # dot must be <= 4
                can_add = False
                break
        
        if can_add:
            added_B.append(v)
            added_B_set.add(v_tuple)
            
    final_vectors = np.vstack([vectors_A, selected_vectors_D, np.array(added_B, dtype=np.int64)]) if added_B else np.vstack([vectors_A, selected_vectors_D])
    
    return final_vectors
