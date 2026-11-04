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

    # Type C: Vectors with two non-zero entries +/- 1
    # These are valid if they don't conflict with Type A (dot product must be < 2)
    # Type A has norm 2, Type C has norm sqrt(2)
    # For valid kissing: |dot| <= min(norm_A, norm_C) = sqrt(2) ~ 1.414
    # So dot product must be in {-1, 0, 1} for Type C vs Type A
    # Type C vs Type C: dot must be in {-2, -1, 0, 1} (norm sqrt(2), so |dot| <= 2)
    # Actually for kissing: distance >= max_norm. If we normalize to norm 2, 
    # then all vectors should have norm 2 after scaling.
    # Let me reconsider: the task says "minimum pairwise distance is at least the maximum vector norm"
    # If we have vectors v_i, we need ||v_i - v_j|| >= max_i ||v_i|| for all i != j
    # This is equivalent to: for normalized vectors u_i = v_i / ||v_i||, we need ||u_i - u_j|| >= 1
    # which means 2 - 2*u_i . u_j >= 1, so u_i . u_j <= 0.5
    # For integer vectors, we need: (v_i . v_j) / (||v_i|| * ||v_j||) <= 0.5
    
    # Type A: norm = 2, so ||v_A|| = 2
    # Type C: norm = sqrt(2), so ||v_C|| = sqrt(2)
    # v_A . v_C: if they share an index, dot = +/-1 or +/-2
    # If same index with same sign: dot = 2, ratio = 2/(2*sqrt(2)) = 1/sqrt(2) > 0.5 INVALID
    # If same index with diff sign: dot = -2, ratio = -1/sqrt(2) < 0.5 OK
    # If adjacent indices: dot = 0, OK
    # So Type C vectors are valid as long as they don't have the same sign as a Type A vector on shared indices
    # But we're constructing a set, so we just need pairwise distances within our set
    
    # Let's generate Type C: all pairs (i,j) with i<j, signs (+1,+1) and (+1,-1)
    # Actually, for maximum packing, we want to be careful
    # Let's focus on Type D first and see if we can add Type C later
    
    # Type D: Vectors with four non-zero entries +/- 1
    indices = list(range(D))
    supports = list(itertools.combinations(indices, 4))
    N_S = len(supports)

    # Precompute popcount for 11-bit integers
    popcount = np.zeros(2048, dtype=np.int8)
    for i in range(2048):
        popcount[i] = bin(i).count('1')

    # Build support masks
    support_masks = []
    for s in supports:
        mask = 0
        for i in s:
            mask |= (1 << i)
        support_masks.append(mask)
    support_masks = np.array(support_masks, dtype=np.int64)

    # Vectorized conflict matrix: two Type D vectors conflict if their supports intersect in >= 3 positions
    # (because then the dot product could be as large as 3 in absolute value, and norms are 2,
    #  so ratio could be 3/4 = 0.75 > 0.5)
    # Actually: if supports intersect in k positions, max |dot| = k
    # Norm of each = 2, so ratio = k/4
    # We need k/4 <= 0.5, so k <= 2
    # So conflict when intersection size >= 3
    masks_col = support_masks[:, None]
    masks_row = support_masks[None, :]
    inter_matrix = masks_col & masks_row
    popcounts = popcount[inter_matrix]
    conflict = popcounts >= 3

    # Degrees for greedy heuristic
    degrees = conflict.sum(axis=1)

    py_rng = random.Random(42)

    best_support_set = set()
    best_size = 0

    def greedy_init_heuristic():
        x = np.zeros(N_S, dtype=bool)
        available = np.ones(N_S, dtype=bool)
        current_size = 0
        avail_deg = degrees.copy().astype(np.int32)

        for _ in range(N_S):
            if not available.any():
                break

            avail_indices = np.where(available)[0]
            min_deg = avail_deg[avail_indices].min()
            candidates = avail_indices[avail_deg[avail_indices] == min_deg]

            if len(candidates) == 0:
                break
            v = py_rng.choice(candidates)

            x[v] = True
            current_size += 1
            neighbors = np.where(conflict[v] & available)[0]
            avail_deg[neighbors] -= 1
            available[neighbors] = False
            available[v] = False

        return x, current_size

    def local_search_1swap(x, current_size, max_iters=5000):
        best_x = x.copy()
        best_size_local = current_size

        conflict_count = np.zeros(N_S, dtype=np.int32)
        if current_size > 0:
            conflict_count = conflict[x].sum(axis=0).astype(np.int32)

        for it in range(max_iters):
            if time.time() > deadline - 10:
                break

            add_candidates = np.where((~x) & (conflict_count == 0))[0]

            if len(add_candidates) > 0:
                v = py_rng.choice(add_candidates)
                x[v] = True
                current_size += 1
                conflict_count += conflict[v]
                if current_size > best_size_local:
                    best_size_local = current_size
                    best_x = x.copy()
            else:
                remove_candidates = np.where(x)[0]
                if len(remove_candidates) == 0:
                    break

                improved = False
                not_in_x = np.where(~x)[0]
                if len(not_in_x) == 0:
                    break

                swap_candidates = np.where((~x) & (conflict_count == 1))[0]

                if len(swap_candidates) > 0:
                    for _ in range(30):
                        if time.time() > deadline - 10:
                            break
                        v_add = py_rng.choice(swap_candidates)
                        conflicting_selected = np.where(conflict[v_add] & x)[0]
                        if len(conflicting_selected) == 0:
                            continue
                        v_rem = conflicting_selected[0]
                        x[v_rem] = False
                        x[v_add] = True
                        conflict_count -= conflict[v_rem]
                        conflict_count += conflict[v_add]
                        improved = True
                        break

                if not improved:
                    for _ in range(20):
                        if time.time() > deadline - 10:
                            break
                        v_rem = py_rng.choice(remove_candidates)
                        v_add = py_rng.choice(not_in_x)

                        if v_add == v_rem:
                            continue

                        conflicts_with_x = conflict[v_add] & x
                        n_conflicts = conflicts_with_x.sum()

                        if n_conflicts == 0 or (n_conflicts == 1 and conflicts_with_x[v_rem]):
                            x[v_rem] = False
                            x[v_add] = True
                            conflict_count -= conflict[v_rem]
                            conflict_count += conflict[v_add]
                            improved = True
                            break

                if not improved:
                    if current_size > 0:
                        v_rem = py_rng.choice(remove_candidates)
                        x[v_rem] = False
                        current_size -= 1
                        conflict_count -= conflict[v_rem]
                    else:
                        break

        return best_x, best_size_local

    num_restarts = 300
    iters_per_restart = 2000

    for restart in range(num_restarts):
        if time.time() > deadline - 20:
            break

        x, size = greedy_init_heuristic()
        best_x, best_size_local = local_search_1swap(x, size, iters_per_restart)

        if best_size_local > best_size:
            best_size = best_size_local
            best_support_set = set(np.where(best_x)[0].tolist())

    # Generate ALL Type D vectors (all sign combinations)
    all_vectors_D = []
    for s in supports:
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, sign in zip(s, signs):
                v[idx] = sign
            all_vectors_D.append(v)
    all_vectors_D = np.array(all_vectors_D, dtype=np.int64)
    N_all_D = len(all_vectors_D)

    # Generate selected Type D vectors
    selected_vectors_D = []
    for s_idx in best_support_set:
        support = supports[s_idx]
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, sign in zip(support, signs):
                v[idx] = sign
            selected_vectors_D.append(v)
    selected_vectors_D = np.array(selected_vectors_D, dtype=np.int64)

    # Greedy fill with all Type D vectors
    current_set_vectors = list(selected_vectors_D)
    current_set_vectors_set = set(map(tuple, current_set_vectors))

    if len(current_set_vectors) > 0:
        current_matrix = np.array(current_set_vectors, dtype=np.int64)
    else:
        current_matrix = np.zeros((0, D), dtype=np.int64)

    rng = np.random.RandomState(42)
    indices_all = np.arange(N_all_D)
    rng.shuffle(indices_all)

    for i in indices_all:
        if time.time() > deadline - 5:
            break
        v = all_vectors_D[i]
        if tuple(v) in current_set_vectors_set:
            continue

        if len(current_matrix) > 0:
            dots = current_matrix @ v
            # For Type D vectors (norm 2), need |dot| <= 1 (since 2 - 2*(dot/4) >= 2*2/4 = 2... 
            # Actually: distance^2 = ||v_i||^2 + ||v_j||^2 - 2*v_i.v_j = 4 + 4 - 2*dot = 8 - 2*dot
            # Need distance >= max_norm = 2, so distance^2 >= 4
            # 8 - 2*dot >= 4 => dot <= 2
            # Also need distance >= 2 for all pairs, and max norm is 2
            # So we need ||v_i - v_j|| >= 2, i.e., ||v_i - v_j||^2 >= 4
            # 8 - 2*dot >= 4 => dot <= 2
            # And by symmetry, we also need to check the other direction, but dot is symmetric
            # So condition is: dot <= 2 AND -dot <= 2, i.e., |dot| <= 2
            # But wait, we also need to check against Type A vectors (norm 2)
            # For Type A (norm 2) and Type D (norm 2): distance^2 = 4 + 4 - 2*dot = 8 - 2*dot >= 4 => dot <= 2
            # So |dot| <= 2 for all pairs
            if np.any(dots > 2) or np.any(dots < -2):
                continue

        current_set_vectors.append(v)
        current_set_vectors_set.add(tuple(v))
        current_matrix = np.vstack([current_matrix, v.reshape(1, -1)])

    final_vectors_D = np.array(current_set_vectors, dtype=np.int64)
    final_vectors = np.vstack([vectors_A, final_vectors_D])
    return final_vectors
