import numpy as np
import itertools
import time

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
    support_masks = []
    for s in supports:
        mask = 0
        for i in s:
            mask |= (1 << i)
        support_masks.append(mask)

    # Precompute conflict matrix
    conflict = np.zeros((N_S, N_S), dtype=bool)
    for i in range(N_S):
        for j in range(i + 1, N_S):
            inter = support_masks[i] & support_masks[j]
            if bin(inter).count('1') >= 3:
                conflict[i, j] = True
                conflict[j, i] = True

    # Maximum Independent Set via Branch and Bound with greedy coloring bound
    # This is a structural change from stochastic local search to systematic search

    # Greedy coloring for upper bound
    def greedy_color_bound(candidates, conflict_matrix):
        """
        Given a set of candidate nodes (as a list of indices),
        compute a greedy coloring to get an upper bound on the maximum independent set size.
        """
        if len(candidates) == 0:
            return 0

        # Order candidates by degree (descending) for better coloring
        degrees = [np.sum(conflict_matrix[c, candidates]) for c in candidates]
        order = np.argsort(-degrees)
        ordered_candidates = [candidates[i] for i in order]

        # Greedy coloring: assign colors such that adjacent nodes get different colors
        color_of = {}
        next_color = 0
        for node in ordered_candidates:
            # Find colors used by neighbors in the candidate set
            neighbor_colors = set()
            for other in ordered_candidates:
                if other != node and conflict_matrix[node, other]:
                    if other in color_of:
                        neighbor_colors.add(color_of[other])

            # Assign the smallest available color
            c = 0
            while c in neighbor_colors:
                c += 1
            color_of[node] = c
            next_color = max(next_color, c + 1)

        return next_color

    # Branch and bound
    best_size = 0
    best_set = []

    def branch_and_bound(candidates, current_set, current_size):
        nonlocal best_size, best_set

        if time.time() > deadline - 10:
            return

        if len(candidates) == 0:
            if current_size > best_size:
                best_size = current_size
                best_set = current_set[:]
            return

        # Upper bound: current_size + greedy coloring bound of candidates
        ub = current_size + greedy_color_bound(candidates, conflict)
        if ub <= best_size:
            return

        # Choose a pivot node: the one with highest degree in the candidate subgraph
        # This helps prune more effectively
        degrees = [np.sum(conflict[c, candidates]) for c in candidates]
        pivot_idx = np.argmax(degrees)
        pivot = candidates[pivot_idx]

        # Branch 1: include pivot
        new_set = current_set + [pivot]
        # New candidates: those not conflicting with pivot
        new_candidates = [c for c in candidates if c != pivot and not conflict[pivot, c]]
        branch_and_bound(new_candidates, new_set, current_size + 1)

        # Branch 2: exclude pivot
        new_candidates = [c for c in candidates if c != pivot]
        branch_and_bound(new_candidates, current_set, current_size)

    # Initialize with an empty candidate set
    all_nodes = list(range(N_S))
    branch_and_bound(all_nodes, [], 0)

    best_support_set = best_set

    # Generate Type D vectors for the selected supports
    selected_vectors_D = []
    for s_idx in best_support_set:
        support = supports[s_idx]
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, sign in zip(support, signs):
                v[idx] = sign
            selected_vectors_D.append(v)

    if selected_vectors_D:
        selected_vectors_D = np.array(selected_vectors_D, dtype=np.int64)
    else:
        selected_vectors_D = np.empty((0, D), dtype=np.int64)

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

    rng = np.random.RandomState(42)
    indices_all = np.arange(N_all_D)
    rng.shuffle(indices_all)

    # Vectorized greedy fill with NumPy dot products
    for i in indices_all:
        if time.time() > deadline - 10:
            break

        v = all_vectors_D[i]
        vt = tuple(v)
        if vt in current_set_vectors_set:
            continue

        if len(current_set_vectors) > 0:
            current_arr = np.array(current_set_vectors)
            dots = current_arr @ v
            if np.any(dots >= 3):
                continue

        current_set_vectors.append(v)
        current_set_vectors_set.add(vt)

    if current_set_vectors:
        final_vectors_D = np.array(current_set_vectors, dtype=np.int64)
    else:
        final_vectors_D = np.empty((0, D), dtype=np.int64)

    # Try adding Type B vectors (two non-zero ±1, norm²=2)
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
            if np.dot(v, u) >= 2:
                can_add = False
                break
        if not can_add:
            continue

        # Check against Type D
        if len(final_vectors_D) > 0:
            dots_D = final_vectors_D @ v
            if np.any(dots_D >= 2):
                can_add = False
        if not can_add:
            continue

        # Check against other Type B
        if added_B:
            added_B_arr = np.array(added_B)
            dots_B = added_B_arr @ v
            if np.any(dots_B != 0):
                can_add = False
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
