import numpy as np
import itertools
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin

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
    support_masks = []
    for s in supports:
        mask = 0
        for i in s:
            mask |= (1 << i)
        support_masks.append(mask)
    support_masks = np.array(support_masks, dtype=np.int64)

    # Precompute popcount for 11-bit integers
    popcount = np.zeros(2048, dtype=np.int8)
    for i in range(2048):
        popcount[i] = bin(i).count('1')

    # Vectorized conflict matrix
    masks_col = support_masks[:, None]
    masks_row = support_masks[None, :]
    inter_matrix = masks_col & masks_row
    popcounts = popcount[inter_matrix]
    conflict = popcounts >= 3

    # Degrees for greedy heuristic
    degrees = conflict.sum(axis=1)

    rng = np.random.RandomState(42)
    py_rng = random.Random(42)

    best_support_set = set()
    best_size = 0

    def greedy_init_heuristic():
        """
        Greedy initialization using minimum degree heuristic with random tie-breaking.
        """
        x = np.zeros(N_S, dtype=bool)
        blocked = np.zeros(N_S, dtype=bool)
        available = np.ones(N_S, dtype=bool)
        current_size = 0

        for _ in range(N_S):
            if not available.any():
                break

            avail_indices = np.where(available)[0]
            static_degrees = degrees[avail_indices]
            min_deg = static_degrees.min()
            candidates = avail_indices[static_degrees == min_deg]

            if len(candidates) == 0:
                break
            v = py_rng.choice(candidates)

            x[v] = True
            current_size += 1
            blocked |= conflict[v]
            available &= ~blocked
            available[v] = False

        return x, current_size

    def local_search_1swap(x, current_size, max_iters=5000):
        """
        Iterative 1-swap local search with vectorized operations.
        """
        best_x = x.copy()
        best_size_local = current_size

        for _ in range(max_iters):
            if time.time() > deadline - 15:
                break

            # Compute blocked vectorized
            if current_size > 0:
                blocked = conflict[x].any(axis=0)
            else:
                blocked = np.zeros(N_S, dtype=bool)

            add_candidates = np.where(~x & ~blocked)[0]

            if len(add_candidates) > 0:
                # Greedily add a random candidate
                v = py_rng.choice(add_candidates)
                x[v] = True
                current_size += 1
                if current_size > best_size_local:
                    best_size_local = current_size
                    best_x = x.copy()
            else:
                # No add possible. Try swaps.
                remove_candidates = np.where(x)[0]
                if len(remove_candidates) == 0:
                    break

                improved = False
                not_in_x = np.where(~x)[0]
                if len(not_in_x) == 0:
                    break

                # Try a few random swaps
                for _ in range(20):
                    if time.time() > deadline - 15:
                        break
                    v_rem = py_rng.choice(remove_candidates)
                    v_add = py_rng.choice(not_in_x)

                    if v_add == v_rem:
                        continue

                    # Check if v_add conflicts with any node in x, excluding v_rem
                    conflicts_with_x = conflict[v_add] & x
                    n_conflicts = conflicts_with_x.sum()

                    if n_conflicts == 0 or (n_conflicts == 1 and conflicts_with_x[v_rem]):
                        x[v_rem] = False
                        x[v_add] = True
                        improved = True
                        break

                if not improved:
                    # Perturb: randomly remove one node
                    if current_size > 0:
                        v_rem = py_rng.choice(remove_candidates)
                        x[v_rem] = False
                        current_size -= 1
                    else:
                        break

        return best_x, best_size_local

    num_restarts = 300
    iters_per_restart = 5000

    for restart in range(num_restarts):
        if time.time() > deadline - 25:
            break

        x, size = greedy_init_heuristic()
        best_x, best_size_local = local_search_1swap(x, size, iters_per_restart)

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

    # Vectorized greedy fill
    current_set_vectors = list(selected_vectors_D)
    current_set_vectors_set = set(map(tuple, current_set_vectors))

    if len(current_set_vectors) > 0:
        current_matrix = np.array(current_set_vectors, dtype=np.int64)
    else:
        current_matrix = np.zeros((0, D), dtype=np.int64)

    indices_all = np.arange(N_all_D)
    rng.shuffle(indices_all)

    for i in indices_all:
        if time.time() > deadline - 10:
            break
        v = all_vectors_D[i]
        if tuple(v) in current_set_vectors_set:
            continue

        if len(current_matrix) > 0:
            dots = current_matrix @ v
            if np.any(dots >= 3):
                continue

        current_set_vectors.append(v)
        current_set_vectors_set.add(tuple(v))
        current_matrix = np.vstack([current_matrix, v.reshape(1, -1)])

    final_vectors_D = np.array(current_set_vectors, dtype=np.int64)
    final_vectors = np.vstack([vectors_A, final_vectors_D])
    return final_vectors
