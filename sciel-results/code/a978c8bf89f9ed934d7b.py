import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    D = 11

    # Type A: 22 axis vectors with one entry ±2
    vectors_A = []
    for i in range(D):
        v1 = [0] * D
        v1[i] = 2
        vectors_A.append(v1)
        v2 = [0] * D
        v2[i] = -2
        vectors_A.append(v2)
    vectors_A = np.array(vectors_A, dtype=np.int64)

    # Type D: Vectors with four non-zero entries ±1
    indices = list(range(D))
    supports = list(itertools.combinations(indices, 4))
    N_S = len(supports)  # 330

    # Build support masks
    support_masks = []
    for s in supports:
        mask = 0
        for i in s:
            mask |= (1 << i)
        support_masks.append(mask)
    support_masks = np.array(support_masks, dtype=np.int64)

    # Conflict: two 4-subsets conflict if they share >= 3 elements
    # Use popcount of (mask_i & mask_j) >= 3

    # Build conflict matrix using vectorized popcount
    popcount_lut = np.zeros(1 << 16, dtype=np.int32)
    for i in range(1, 1 << 16):
        popcount_lut[i] = popcount_lut[i >> 1] + (i & 1)

    conflict = np.zeros((N_S, N_S), dtype=bool)

    # Process in chunks
    chunk_size = 100
    for i_start in range(0, N_S, chunk_size):
        i_end = min(i_start + chunk_size, N_S)
        masks_i = support_masks[i_start:i_end]
        inter = masks_i[:, None] & support_masks[None, :]
        inter_flat = inter.ravel()
        pc = popcount_lut[inter_flat].reshape(inter.shape)
        conflict[i_start:i_end, :] = (pc >= 3)

    conflict = conflict | conflict.T
    np.fill_diagonal(conflict, False)

    # Maximum Independent Set via branch-and-bound with greedy coloring
    # Use a recursive approach with good pruning

    # Greedy coloring bound
    def greedy_color_bound(candidates):
        """
        candidates: array of node indices in the candidate set
        Returns upper bound on the size of an independent set within candidates
        """
        n = len(candidates)
        if n == 0:
            return 0
        # Simple greedy coloring: assign colors in order
        # Color classes are independent sets
        colors = {}
        max_color = 0
        for v in candidates:
            # Find colors used by neighbors of v that are already colored
            neighbor_colors = set()
            for u, c in colors.items():
                if conflict[v, u]:
                    neighbor_colors.add(c)
            # Assign smallest available color
            c = 0
            while c in neighbor_colors:
                c += 1
            colors[v] = c
            max_color = max(max_color, c)
        return max_color + 1  # Colors are 0-indexed, so max_color+1 is the number of colors

    # Branch and bound
    best_size = 0
    best_set = set()

    # Initial greedy solution
    x = np.zeros(N_S, dtype=bool)
    blocked = np.zeros(N_S, dtype=bool)
    degrees = np.array([np.sum(conflict[i]) for i in range(N_S)])
    order = np.argsort(degrees)
    for v in order:
        if not blocked[v] and not x[v]:
            x[v] = True
            blocked |= conflict[v]
    best_size = int(np.sum(x))
    best_set = set(np.where(x)[0].tolist())

    # Branch and bound with stack
    # State: (candidates_list, current_selected_list)
    # We use a depth-first search with pruning

    # To keep it efficient, limit the search
    # Use a time-limited approach

    # Actually, for this specific graph, let's use a more targeted approach
    # The graph has a known structure: it's the complement of a line graph-like structure

    # Let's use a simple but effective local search with multiple restarts
    # and a final polish step

    def local_search(x, max_iter=100000):
        """
        Local search to improve the independent set
        """
        x = x.copy()
        in_set = set(np.where(x)[0].tolist())
        current_size = len(in_set)
        best_local = x.copy()
        best_local_size = current_size

        for it in range(max_iter):
            if time.time() > deadline - 35:
                break

            # Compute blocked nodes
            blocked = np.zeros(N_S, dtype=bool)
            for v in in_set:
                blocked |= conflict[v]

            # Candidates: not in set, not blocked
            candidates = np.where(~x & ~blocked)[0]

            if len(candidates) > 0:
                # Try adding a random candidate
                v = candidates[np.random.randint(len(candidates))]
                x[v] = True
                in_set.add(v)
                current_size += 1
                if current_size > best_local_size:
                    best_local_size = current_size
                    best_local = x.copy()
            else:
                # No candidates, try a swap: remove one, add others
                if len(in_set) > 0:
                    # Remove a random node
                    remove_list = list(in_set)
                    v_remove = remove_list[np.random.randint(len(remove_list))]
                    x[v_remove] = False
                    in_set.discard(v_remove)
                    current_size -= 1

                    # Recompute blocked
                    blocked2 = np.zeros(N_S, dtype=bool)
                    for v in in_set:
                        blocked2 |= conflict[v]
                    candidates2 = np.where(~x & ~blocked2)[0]

                    # Try to add multiple nodes
                    for v in candidates2:
                        if np.random.random() < 0.5:
                            x[v] = True
                            in_set.add(v)
                            current_size += 1

        return best_local, best_local_size

    # Multiple restarts
    rng = np.random.RandomState(42)
    np.random.seed(42)

    for restart in range(300):
        if time.time() > deadline - 35:
            break

        # Greedy initialization with random ordering
        x = np.zeros(N_S, dtype=bool)
        blocked = np.zeros(N_S, dtype=bool)
        order = np.arange(N_S)
        rng.shuffle(order)
        for v in order:
            if not blocked[v] and not x[v]:
                x[v] = True
                blocked |= conflict[v]

        # Local search
        x, current_size = local_search(x, max_iter=50000)

        if current_size > best_size:
            best_size = current_size
            best_set = set(np.where(x)[0].tolist())

    # Generate Type D vectors for the selected supports
    selected_vectors_D = []
    for s_idx in best_set:
        support = supports[s_idx]
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, sign in zip(support, signs):
                v[idx] = sign
            selected_vectors_D.append(v)
    selected_vectors_D = np.array(selected_vectors_D, dtype=np.int64)

    final_vectors = np.vstack([vectors_A, selected_vectors_D])
    return final_vectors
