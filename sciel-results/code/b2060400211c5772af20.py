import numpy as np
import itertools
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 135.0

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

    # Precompute popcount for 11-bit integers (0 to 2047)
    popcount = np.zeros(2048, dtype=np.int8)
    for i in range(2048):
        popcount[i] = bin(i).count('1')

    # Vectorized conflict matrix computation
    masks_col = support_masks[:, None]
    masks_row = support_masks[None, :]
    inter_matrix = masks_col & masks_row
    popcounts = popcount[inter_matrix]
    conflict = popcounts >= 3

    # Precompute conflict neighbors for each support
    conflict_neighbors = [np.where(conflict[i])[0] for i in range(N_S)]

    rng = np.random.RandomState(42)
    py_rng = random.Random(42)

    best_support_set = set()
    best_size = 0

    def greedy_init(seed_offset):
        order = np.arange(N_S)
        rng.shuffle(order)
        x = np.zeros(N_S, dtype=bool)
        blocked = np.zeros(N_S, dtype=bool)
        for v in order:
            if not blocked[v]:
                x[v] = True
                blocked[conflict_neighbors[v]] = True
        return x, int(np.sum(x))

    def local_search(x, current_size, num_iterations, T0, T_end):
        best_x = x.copy()
        best_size_local = current_size
        T = T0
        cooling = (T_end / T0) ** (1.0 / num_iterations)
        
        for it in range(num_iterations):
            if time.time() > deadline - 15:
                break
            
            # Compute blocked nodes using conflict neighbors
            if current_size > 0:
                blocked = np.zeros(N_S, dtype=bool)
                for v in np.where(x)[0]:
                    blocked[conflict_neighbors[v]] = True
            else:
                blocked = np.zeros(N_S, dtype=bool)
            
            add_candidates = np.where(~x & ~blocked)[0]
            remove_candidates = np.where(x)[0]
            
            if len(add_candidates) == 0 and len(remove_candidates) == 0:
                break
                
            move_type = py_rng.random()
            if move_type < 0.65 and len(add_candidates) > 0:
                # Add
                v = add_candidates[py_rng.randint(0, len(add_candidates)-1)]
                if not x[v]:
                    x[v] = True
                    new_size = current_size + 1
                    if new_size > best_size_local:
                        best_size_local = new_size
                        best_x = x.copy()
                    current_size = new_size
            elif move_type < 0.85 and len(remove_candidates) > 0:
                # Remove
                v = remove_candidates[py_rng.randint(0, len(remove_candidates)-1)]
                if x[v]:
                    x[v] = False
                    new_size = current_size - 1
                    if py_rng.random() < np.exp(-1.0 / T):
                        current_size = new_size
                    else:
                        x[v] = True
            else:
                # Swap: remove one, add one that becomes unblocked
                if len(add_candidates) > 0 and len(remove_candidates) > 0:
                    v_rem = remove_candidates[py_rng.randint(0, len(remove_candidates)-1)]
                    # Find candidates that would be unblocked by removing v_rem
                    # These are nodes that conflict only with v_rem among selected nodes
                    temp_blocked = np.zeros(N_S, dtype=bool)
                    for u in np.where(x)[0]:
                        if u != v_rem:
                            temp_blocked[conflict_neighbors[u]] = True
                    unblocked_by_removal = np.where(~x & ~temp_blocked)[0]
                    if len(unblocked_by_removal) > 0:
                        v_add = unblocked_by_removal[py_rng.randint(0, len(unblocked_by_removal)-1)]
                        x[v_rem] = False
                        x[v_add] = True
                        # Size unchanged, but structure improved
                        pass
            
            T *= cooling

        return best_x, best_size_local

    num_restarts = 200
    iterations_per_restart = 8000
    T0 = 1.0
    T_end = 0.005

    for restart in range(num_restarts):
        if time.time() > deadline - 25:
            break
        x, size = greedy_init(restart)
        best_x, best_size_local = local_search(x, size, iterations_per_restart, T0, T_end)
        
        if best_size_local > best_size:
            best_size = best_size_local
            best_support_set = set(np.where(best_x)[0].tolist())

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
            for idx, sign in zip(s, signs):
                v[idx] = sign
            all_vectors_D.append(v)
    all_vectors_D = np.array(all_vectors_D, dtype=np.int64)
    N_all_D = len(all_vectors_D)

    indices_all = np.arange(N_all_D)
    rng.shuffle(indices_all)

    for i in indices_all:
        if time.time() > deadline - 10:
            break
        v = all_vectors_D[i]
        if tuple(v) in current_set_vectors_set:
            continue
        can_add = True
        for u in current_set_vectors:
            dot = np.dot(v, u)
            if dot >= 3:
                can_add = False
                break
        if can_add:
            current_set_vectors.append(v)
            current_set_vectors_set.add(tuple(v))

    final_vectors_D = np.array(current_set_vectors, dtype=np.int64)
    final_vectors = np.vstack([vectors_A, final_vectors_D])
    return final_vectors
