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

    # Build support masks
    support_masks = np.zeros(N_S, dtype=np.uint64)
    for i, s in enumerate(supports):
        mask = 0
        for idx in s:
            mask |= (1 << idx)
        support_masks[i] = mask

    # Precompute conflict matrix for supports (intersection >= 3)
    conflict_supports = np.zeros((N_S, N_S), dtype=bool)
    popcount_lookup = np.array([bin(i).count('1') for i in range(2048)], dtype=np.int8)
    for i in range(N_S):
        ands = support_masks[i] & support_masks
        counts = popcount_lookup[ands]
        conflict_supports[i, :] = counts >= 3
    conflict_supports = np.maximum(conflict_supports, conflict_supports.T)
    np.fill_diagonal(conflict_supports, False)

    # Phase 1: Select Supports using Simulated Annealing
    rng = np.random.RandomState(42)
    py_rng = random.Random(42)

    best_support_set = set()
    best_size = 0

    def greedy_init_supports():
        order = np.arange(N_S)
        rng.shuffle(order)
        x = np.zeros(N_S, dtype=bool)
        blocked = np.zeros(N_S, dtype=bool)
        for v in order:
            if not blocked[v] and not x[v]:
                x[v] = True
                blocked |= conflict_supports[v]
        return x

    def sa_supports():
        x = greedy_init_supports()
        current_size = np.sum(x)
        best_x = x.copy()
        best_current_size = current_size

        T = 1.0
        T_end = 0.01
        num_iterations = 5000
        cooling = (T_end / T) ** (1.0 / num_iterations)

        for it in range(num_iterations):
            if time.time() > deadline - 60:
                break

            v = py_rng.randint(0, N_S)

            if x[v]:
                x[v] = False
                delta = -1
                if py_rng.random() < np.exp(delta / T):
                    current_size += delta
                else:
                    x[v] = True
            else:
                if np.any(conflict_supports[v, x]):
                    T *= cooling
                    continue
                x[v] = True
                delta = 1
                current_size += delta
                if current_size > best_current_size:
                    best_current_size = current_size
                    best_x = x.copy()

            T *= cooling

        return best_x, best_current_size

    num_restarts = 10
    for r in range(num_restarts):
        if time.time() > deadline - 60:
            break
        x, size = sa_supports()
        if size > best_size:
            best_size = size
            best_support_set = set(np.where(x)[0].tolist())

    # Phase 2: Generate Vectors
    all_vectors_D = []
    support_to_vectors = {i: [] for i in range(N_S)}
    for s_idx, s in enumerate(supports):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, sign in zip(s, signs):
                v[idx] = sign
            v_arr = np.array(v, dtype=np.int64)
            all_vectors_D.append(v_arr)
            support_to_vectors[s_idx].append(v_arr)

    all_vectors_D = np.array(all_vectors_D, dtype=np.int64)
    N_all_D = len(all_vectors_D)

    # Initialize set with Type A
    current_vectors = [vectors_A]
    current_set = set(map(tuple, vectors_A))
    current_arr = vectors_A.copy()

    # Add vectors from selected supports greedily
    for s_idx in best_support_set:
        for v in support_to_vectors[s_idx]:
            current_vectors.append(v)
            current_set.add(tuple(v))

    current_arr = np.array(current_vectors, dtype=np.int64)

    # Phase 3: Greedy Fill with Non-Selected Supports
    indices_all = np.arange(N_all_D)
    rng.shuffle(indices_all)

    for i in indices_all:
        if time.time() > deadline - 20:
            break
        v = all_vectors_D[i]
        vt = tuple(v)
        if vt in current_set:
            continue

        dots = current_arr @ v
        if np.any(dots > 2):
            continue

        current_vectors.append(v)
        current_set.add(vt)
        current_arr = np.vstack([current_arr, v])

    # Phase 4: Local Search (1-Swap) on Vectors
    num_swaps = 500
    for _ in range(num_swaps):
        if time.time() > deadline - 10:
            break

        if len(current_vectors) <= 1:
            break

        idx_remove = py_rng.randint(0, len(current_vectors) - 1)
        v_remove = current_vectors[idx_remove]
        vt_remove = tuple(v_remove)

        # Temporarily remove
        current_vectors.pop(idx_remove)
        current_set.remove(vt_remove)
        current_arr = np.delete(current_arr, idx_remove, axis=0)

        # Try to add up to 3 candidates
        added_any = False
        for _ in range(3):
            i_cand = py_rng.randint(0, N_all_D - 1)
            v_cand = all_vectors_D[i_cand]
            vt_cand = tuple(v_cand)
            if vt_cand in current_set:
                continue

            dots = current_arr @ v_cand
            if np.any(dots > 2):
                continue

            current_vectors.append(v_cand)
            current_set.add(vt_cand)
            current_arr = np.vstack([current_arr, v_cand])
            added_any = True

        if not added_any:
            # Revert
            current_vectors.insert(idx_remove, v_remove)
            current_set.add(vt_remove)
            current_arr = np.vstack([current_arr, v_remove])

    # Phase 5: Add Type B Vectors
    vectors_B_candidates = []
    for i in range(D):
        for j in range(i + 1, D):
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

        dots = current_arr @ v
        if np.any(dots > 1):
            continue

        can_add = True
        for u in added_B:
            if np.dot(v, u) > 0:
                can_add = False
                break
        if not can_add:
            continue

        added_B.append(v)
        added_B_set.add(vt)

    if added_B:
        added_B = np.array(added_B, dtype=np.int64)
        final_vectors = np.vstack([current_arr, added_B])
    else:
        final_vectors = current_arr

    return final_vectors
