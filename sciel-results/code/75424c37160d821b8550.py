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

    # Build support masks
    support_masks = []
    for s in supports:
        mask = 0
        for i in s:
            mask |= (1 << i)
        support_masks.append(mask)
    support_masks = np.array(support_masks, dtype=np.int64)

    # Precompute conflict matrix using popcount lookup table
    # Two 4-subsets conflict if they share >= 3 elements
    # For 4-subsets, sharing >= 3 means sharing exactly 3 or 4
    # Sharing 4 means they're the same (not a conflict since we only consider distinct)
    # Sharing 3 means they differ in exactly 1 element

    # Actually, let's think about this differently.
    # For 4-subsets of an 11-element set, two subsets conflict if |A ∩ B| >= 3
    # This means they share 3 or 4 elements
    # If they share 4, they're identical (not a distinct pair)
    # So conflict means they share exactly 3 elements

    # Two 4-subsets share exactly 3 elements if and only if they differ in exactly 1 element
    # i.e., A = B \ {x} ∪ {y} for some x, y

    # Let's precompute the conflict matrix efficiently
    # For each support, find all supports that share exactly 3 elements with it

    # Create a mapping from mask to index
    mask_to_idx = {mask: i for i, mask in enumerate(support_masks)}

    # For each support, generate all 4-subsets that share exactly 3 elements with it
    # If support is {a,b,c,d}, the conflicting supports are:
    # {a,b,c,e}, {a,b,d,e}, {a,c,d,e}, {b,c,d,e} for each e not in support
    # That's 4 * (11-4) = 28 potential conflicts per support

    conflict = np.zeros((N_S, N_S), dtype=bool)
    for i in range(N_S):
        s = supports[i]
        # For each element in s, remove it and add each element not in s
        for k in range(4):
            without_k = set(s) - {s[k]}
            for e in range(D):
                if e not in without_k and e != s[k]:
                    new_s = tuple(sorted(without_k | {e}))
                    # Check if new_s is a valid 4-subset
                    if len(new_s) == 4:
                        # Compute mask for new_s
                        new_mask = 0
                        for idx in new_s:
                            new_mask |= (1 << idx)
                        if new_mask in mask_to_idx:
                            j = mask_to_idx[new_mask]
                            if j != i:
                                conflict[i, j] = True
                                conflict[j, i] = True

    # Build adjacency list
    adj = [set() for _ in range(N_S)]
    for i in range(N_S):
        for j in range(N_S):
            if conflict[i, j]:
                adj[i].add(j)

    rng = np.random.RandomState(42)
    py_rng = random.Random(42)

    best_support_set = set()
    best_size = 0

    # Greedy initialization with random ordering
    def greedy_init():
        order = np.arange(N_S)
        rng.shuffle(order)
        x = np.zeros(N_S, dtype=bool)
        blocked = np.zeros(N_S, dtype=bool)
        for v in order:
            if not blocked[v] and not x[v]:
                x[v] = True
                blocked |= conflict[v]
        return x

    # Simulated Annealing with vectorized operations
    T0 = 1.0
    T_end = 0.005
    num_iterations = 40000
    cooling = (T_end / T0) ** (1.0 / num_iterations)

    for restart in range(100):
        if time.time() > deadline - 25:
            break

        x = greedy_init()
        current_size = int(np.sum(x))
        best_x = x.copy()
        best_current_size = current_size

        T = T0
        for it in range(num_iterations):
            if time.time() > deadline - 25:
                break

            in_set = np.where(x)[0]
            not_in_set = np.where(~x)[0]

            # Compute blocked nodes
            if len(in_set) > 0:
                blocked = np.zeros(N_S, dtype=bool)
                for v in in_set:
                    blocked |= conflict[v]
            else:
                blocked = np.zeros(N_S, dtype=bool)

            add_candidates = np.where(~x & ~blocked)[0]
            remove_candidates = in_set

            if len(add_candidates) == 0 and len(remove_candidates) == 0:
                break

            move_type = py_rng.random()
            if move_type < 0.45 and len(add_candidates) > 0:
                v = py_rng.choice(add_candidates)
                x[v] = True
                current_size += 1
                if current_size > best_current_size:
                    best_current_size = current_size
                    best_x = x.copy()
            elif move_type < 0.75 and len(remove_candidates) > 0:
                v = py_rng.choice(remove_candidates)
                x[v] = False
                current_size -= 1
                if py_rng.random() >= np.exp(-1.0 / T):
                    x[v] = True
                    current_size += 1
            else:
                # Swap
                if len(add_candidates) > 0 and len(remove_candidates) > 0:
                    v_remove = py_rng.choice(remove_candidates)
                    v_add = py_rng.choice(add_candidates)
                    if v_remove != v_add:
                        # Check if v_add conflicts with any in-set node except v_remove
                        in_set_no_remove = [v for v in in_set if v != v_remove]
                        if len(in_set_no_remove) > 0:
                            if np.any(conflict[v_add, in_set_no_remove]):
                                T *= cooling
                                continue
                        x[v_remove] = False
                        x[v_add] = True

            T *= cooling

        if best_current_size > best_size:
            best_size = best_current_size
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
    current_arr = np.array(current_set_vectors, dtype=np.int64)

    # Generate ALL Type D vectors
    all_vectors_D = []
    for s in supports:
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, sign in zip(support, signs):
                v[idx] = sign
            all_vectors_D.append(v)
    all_vectors_D = np.array(all_vectors_D, dtype=np.int64)
    N_all_D = len(all_vectors_D)

    indices_all = np.arange(N_all_D)
    rng.shuffle(indices_all)

    for i in indices_all:
        if time.time() > deadline - 15:
            break
        v = all_vectors_D[i]
        if tuple(v) in current_set_vectors_set:
            continue
        dots = current_arr @ v
        if np.any(dots >= 3):
            continue
        current_set_vectors.append(v)
        current_set_vectors_set.add(tuple(v))
        current_arr = np.vstack([current_arr, v.reshape(1, -1)])

    final_vectors_D = np.array(current_set_vectors, dtype=np.int64)
    final_vectors = np.vstack([vectors_A, final_vectors_D])
    return final_vectors
