import numpy as np
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    D = 11

    # Precomputation
    supports = list(itertools.combinations(range(D), 4))
    N_S = len(supports)  # 330

    support_masks = np.array([sum(1 << i for i in s) for s in supports], dtype=np.int64)

    # Build conflict matrix vectorized
    iu = np.triu_indices(N_S, k=1)
    pairs = (support_masks[iu[0]] & support_masks[iu[1]])
    popcount_lut = np.array([bin(i).count('1') for i in range(2048)], dtype=np.int8)
    intersections = popcount_lut[pairs]
    conflict_pairs = np.where(intersections >= 3)[0]

    conflict = np.zeros((N_S, N_S), dtype=bool)
    conflict[iu[0][conflict_pairs], iu[1][conflict_pairs]] = True
    conflict[iu[1][conflict_pairs], iu[0][conflict_pairs]] = True

    # Precompute adjacency lists for greedy init
    neighbors = [np.where(conflict[i])[0] for i in range(N_S)]
    degree = np.sum(conflict, axis=1)

    rng = np.random.default_rng(42)

    def greedy_init():
        """Greedy MIS with random tie-breaking based on degree."""
        in_set = np.zeros(N_S, dtype=bool)
        remaining = np.ones(N_S, dtype=bool)
        while np.any(remaining):
            if time.time() > deadline:
                break
            rem_idx = np.where(remaining)[0]
            if len(rem_idx) == 0:
                break
            degs = degree[rem_idx].astype(float)
            noise = rng.exponential(0.5, size=len(rem_idx))
            scores = degs + noise
            min_idx = np.argmin(scores)
            chosen = rem_idx[min_idx]
            in_set[chosen] = True
            remaining[chosen] = False
            remaining[neighbors[chosen]] = False
        return in_set

    def sa_optimize(in_set, n_steps=3000):
        """Simulated annealing to maximize independent set size."""
        current = in_set.copy()
        current_size = int(np.sum(current))
        best = current.copy()
        best_size = current_size

        T_start = 2.0
        T_end = 0.01
        alpha = (T_end / T_start) ** (1.0 / n_steps)
        T = T_start

        set_members = np.where(current)[0]

        for step in range(n_steps):
            if time.time() > deadline - 5.0:
                break
            T *= alpha

            move_type = rng.integers(0, 3)

            if move_type == 0:  # Try to add a node
                not_in_set = np.where(~current)[0]
                if len(not_in_set) == 0:
                    continue
                cand = not_in_set[rng.integers(0, len(not_in_set))]
                conflicts_with_set = np.sum(conflict[cand] & current)
                if conflicts_with_set == 0:
                    current[cand] = True
                    current_size += 1
                    set_members = np.where(current)[0]
                    if current_size > best_size:
                        best_size = current_size
                        best = current.copy()

            elif move_type == 1:  # Try to remove a node
                if len(set_members) == 0:
                    continue
                idx = rng.integers(0, len(set_members))
                node = set_members[idx]
                if rng.random() < np.exp(-1.0 / max(T, 1e-10)):
                    current[node] = False
                    current_size -= 1
                    set_members = np.where(current)[0]

            else:  # Swap: remove one, add one
                if len(set_members) == 0:
                    continue
                idx = rng.integers(0, len(set_members))
                remove_node = set_members[idx]

                # Temporarily remove
                current[remove_node] = False
                current_size -= 1
                set_members = np.where(current)[0]

                # Find valid candidates: not in set, no conflict with current set
                not_in_set = np.where(~current)[0]
                if len(not_in_set) > 0:
                    # Vectorized: count conflicts for all candidates
                    cand_conflicts = np.sum(conflict[not_in_set] & current[None, :], axis=1)
                    valid_cands = not_in_set[cand_conflicts == 0]

                    if len(valid_cands) > 0:
                        # Pick a random valid candidate
                        add_node = valid_cands[rng.integers(0, len(valid_cands))]
                        current[add_node] = True
                        current_size += 1
                        set_members = np.where(current)[0]
                        if current_size > best_size:
                            best_size = current_size
                            best = current.copy()
                    else:
                        # No valid candidate: put back
                        current[remove_node] = True
                        current_size += 1
                        set_members = np.where(current)[0]

        return best, best_size

    def greedy_add_pass(in_set):
        """Try to add any node that doesn't conflict with the current set."""
        current = in_set.copy()
        current_size = int(np.sum(current))
        not_in_set = np.where(~current)[0]
        if len(not_in_set) == 0:
            return current, current_size
        cand_conflicts = np.sum(conflict[not_in_set] & current[None, :], axis=1)
        valid_cands = not_in_set[cand_conflicts == 0]
        for v in valid_cands:
            current[v] = True
            current_size += 1
        return current, current_size

    best_size = 0
    best_selection = np.zeros(N_S, dtype=bool)

    # Run multiple restarts
    num_restarts = 80
    for r in range(num_restarts):
        if time.time() > deadline - 15.0:
            break

        in_set = greedy_init()
        init_size = int(np.sum(in_set))

        # SA optimization
        sa_result, sa_size = sa_optimize(in_set, n_steps=2000)

        # Greedy add pass
        final_result, final_size = greedy_add_pass(sa_result)

        if final_size > best_size:
            best_size = final_size
            best_selection = final_result.copy()

        # If we found something good, try to improve further
        if final_size > best_size - 2 and time.time() < deadline - 25.0:
            improved, imp_size = sa_optimize(final_result, n_steps=3000)
            improved, imp_size2 = greedy_add_pass(improved)
            if imp_size2 > best_size:
                best_size = imp_size2
                best_selection = improved.copy()

    # Construct vectors
    vectors = []

    # Type A: ±2e_i for i in range(11)
    for i in range(D):
        v1 = [0] * D
        v1[i] = 2
        vectors.append(v1)
        v2 = [0] * D
        v2[i] = -2
        vectors.append(v2)

    # Type D: All 16 sign vectors for each selected support
    selected_indices = np.where(best_selection)[0]
    for s_idx in selected_indices:
        support = supports[s_idx]
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, sign in zip(support, signs):
                v[idx] = sign
            vectors.append(v)

    if not vectors:
        return np.array([], dtype=np.int8).reshape(0, D)

    final_vectors = np.array(vectors, dtype=np.int8)

    return final_vectors
