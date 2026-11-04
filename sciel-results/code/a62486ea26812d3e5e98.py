import numpy as np
import itertools
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 148.0

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

    # Build conflict graph using bitmask popcount
    support_masks = np.zeros(N_S, dtype=np.int32)
    for s_idx, s in enumerate(supports):
        mask = 0
        for i in s:
            mask |= (1 << i)
        support_masks[s_idx] = mask

    # Efficient conflict computation: two 4-sets conflict if |intersection| >= 3
    # For 11-bit masks, we can use the fact that |A ∩ B| = popcount(A & B)
    # Precompute popcount for all 2^11 = 2048 possible masks
    popcount = np.zeros(2048, dtype=np.int8)
    for i in range(1, 2048):
        popcount[i] = popcount[i >> 1] + (i & 1)

    # Compute conflict matrix in chunks to avoid memory issues
    conflict = np.zeros((N_S, N_S), dtype=bool)
    chunk_size = 64
    for i_start in range(0, N_S, chunk_size):
        i_end = min(i_start + chunk_size, N_S)
        chunk = support_masks[i_start:i_end]
        # For each mask in chunk, compute intersection with all masks
        # Use broadcasting: chunk[:, None] & support_masks[None, :]
        inter = chunk[:, None] & support_masks[None, :]
        # inter is (chunk_size, N_S), values are 11-bit masks
        # Need popcount of each element
        # Reshape to use vectorized popcount
        inter_flat = inter.ravel()
        pc_flat = popcount[inter_flat]
        pc = pc_flat.reshape(i_end - i_start, N_S)
        conflict[i_start:i_end] = (pc >= 3)

    # Make symmetric (should already be, but ensure)
    conflict = conflict | conflict.T

    rng = np.random.RandomState(42)
    py_rng = random.Random(42)

    best_support_set = set()
    best_size = 0

    # Greedy initialization with multiple strategies
    def greedy_init(strategy='degree'):
        x = np.zeros(N_S, dtype=bool)
        if strategy == 'degree':
            order = np.argsort(conflict.sum(axis=1))
        elif strategy == 'random':
            order = np.arange(N_S)
            rng.shuffle(order)
        else:  # 'mixed'
            order = np.arange(N_S)
            # Sort by degree but with some randomness
            deg = conflict.sum(axis=1)
            noise = rng.random(N_S) * 10
            order = np.argsort(deg + noise)
        
        blocked = np.zeros(N_S, dtype=bool)
        for v in order:
            if not blocked[v]:
                x[v] = True
                blocked |= conflict[v]
        return x, int(np.sum(x))

    # Optimized local search with cached blocked set
    def local_search(x, current_size, num_iterations, T0, T_end):
        best_x = x.copy()
        best_size_local = current_size
        T = T0
        cooling = (T_end / T0) ** (1.0 / num_iterations)
        
        # Precompute conflict rows for fast access
        conflict_rows = conflict  # Already a numpy array
        
        for it in range(num_iterations):
            if time.time() > deadline - 15:
                break
            
            # Compute blocked set efficiently
            if current_size > 0:
                selected = np.where(x)[0]
                # blocked[i] = True if any selected node conflicts with i
                blocked = np.zeros(N_S, dtype=bool)
                # Use the fact that conflict is symmetric
                for s in selected:
                    blocked[conflict_rows[s]] = True
                # Selected nodes themselves are not "blocked" in the add sense
                blocked[selected] = False
            else:
                blocked = np.zeros(N_S, dtype=bool)
            
            add_candidates = np.where(~x & ~blocked)[0]
            remove_candidates = np.where(x)[0]
            
            if len(add_candidates) == 0 and len(remove_candidates) == 0:
                break
                
            move_type = py_rng.random()
            if move_type < 0.75 and len(add_candidates) > 0:
                v = add_candidates[py_rng.randint(0, len(add_candidates) - 1)]
                x[v] = True
                current_size += 1
                if current_size > best_size_local:
                    best_size_local = current_size
                    best_x = x.copy()
            elif move_type < 0.90 and len(remove_candidates) > 0:
                v = remove_candidates[py_rng.randint(0, len(remove_candidates) - 1)]
                x[v] = False
                current_size -= 1
                # Simulated annealing: accept with probability based on T
                if T > 0.01:
                    if py_rng.random() < np.exp(-1.0 / T):
                        x[v] = True
                        current_size += 1
            else:
                # Swap move
                if len(add_candidates) > 0 and len(remove_candidates) > 0:
                    v_add = add_candidates[py_rng.randint(0, len(add_candidates) - 1)]
                    v_rem = remove_candidates[py_rng.randint(0, len(remove_candidates) - 1)]
                    if v_add != v_rem:
                        x[v_rem] = False
                        x[v_add] = True
                        # Size stays the same
            
            T *= cooling

        return best_x, best_size_local

    # Multiple restarts with different strategies
    strategies = ['degree', 'random', 'mixed']
    num_restarts = 80
    iterations_per_restart = 10000
    T0 = 1.5
    T_end = 0.01

    for restart in range(num_restarts):
        if time.time() > deadline - 30:
            break
        strategy = strategies[restart % len(strategies)]
        x, size = greedy_init(strategy)
        best_x, best_size_local = local_search(x, size, iterations_per_restart, T0, T_end)
        
        if best_size_local > best_size:
            best_size = best_size_local
            best_support_set = set(np.where(best_x)[0].tolist())

    # Generate ALL Type D vectors upfront
    all_vectors_D = []
    all_support_indices = []
    for s_idx, s in enumerate(supports):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, sign in zip(s, signs):
                v[idx] = sign
            all_vectors_D.append(v)
            all_support_indices.append(s_idx)
    all_vectors_D = np.array(all_vectors_D, dtype=np.int64)
    all_support_indices = np.array(all_support_indices, dtype=np.int64)

    # Start with vectors from selected supports
    in_selected_support = np.isin(all_support_indices, list(best_support_set))
    current_set_vectors = all_vectors_D[in_selected_support].tolist()
    
    # For remaining vectors, do greedy fill
    remaining_indices = np.where(~in_selected_support)[0]
    rng.shuffle(remaining_indices)
    
    if len(current_set_vectors) > 0:
        existing = np.array(current_set_vectors, dtype=np.int64)
        n_existing = len(existing)
    else:
        existing = np.empty((0, D), dtype=np.int64)
        n_existing = 0
    
    # Process in batches, checking compatibility
    batch_size = 300
    for batch_start in range(0, len(remaining_indices), batch_size):
        if time.time() > deadline - 8:
            break
        batch_end = min(batch_start + batch_size, len(remaining_indices))
        batch_idx = remaining_indices[batch_start:batch_end]
        batch_vecs = all_vectors_D[batch_idx]
        
        if n_existing == 0:
            current_set_vectors.extend(batch_vecs.tolist())
            existing = batch_vecs.copy()
            n_existing = len(existing)
        else:
            # Compute dot products with existing set
            dots = batch_vecs @ existing.T
            # Compatible if |dot| <= 2 for all existing vectors
            compatible = (np.abs(dots).max(axis=1) <= 2)
            
            new_vecs = batch_vecs[compatible]
            if len(new_vecs) > 0:
                current_set_vectors.extend(new_vecs.tolist())
                existing = np.vstack([existing, new_vecs])
                n_existing = len(existing)

    final_vectors_D = np.array(current_set_vectors, dtype=np.int64)
    final_vectors = np.vstack([vectors_A, final_vectors_D])
    return final_vectors
