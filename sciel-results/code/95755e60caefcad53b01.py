import time
import numpy as np
import random

def construct(q, n, radius):
    t_start = time.time()
    deadline = 115.0  # Safety margin before 120s limit

    space_size = q ** n
    q_powers = [1] * (n + 1)
    for i in range(1, n + 1):
        q_powers[i] = q_powers[i - 1] * q

    def word_to_idx(w):
        idx = 0
        for d in w:
            idx = idx * q + d
        return idx

    def idx_to_word(idx):
        w = [0] * n
        for pos in range(n):
            w[pos] = (idx // q_powers[n - 1 - pos]) % q
        return w

    # Precompute neighbors for radius=1
    nb_count = 1 + n * (q - 1)
    neighbors = np.empty((space_size, nb_count), dtype=np.int32)
    for idx in range(space_size):
        w = idx_to_word(idx)
        nb_list = [idx]
        for pos in range(n):
            for new_d in range(q):
                if new_d == w[pos]:
                    continue
                delta = (new_d - w[pos]) * q_powers[n - 1 - pos]
                nb_list.append(idx + delta)
        neighbors[idx] = nb_list

    # Build incidence matrix A[v, u] = 1 if u is within distance radius of v
    A = np.zeros((space_size, space_size), dtype=np.int32)
    for v in range(space_size):
        A[v, neighbors[v]] = 1

    rng = np.random.default_rng(42)

    # --- Phase 1: Greedy construction to get a valid starting cover ---
    uncovered = np.ones(space_size, dtype=np.int32)
    total_uncovered = space_size
    code_indices = []
    selected_mask = np.zeros(space_size, dtype=bool)

    while total_uncovered > 0 and (time.time() - t_start) < deadline - 60.0:
        scores = A @ uncovered.astype(np.float64)
        scores[selected_mask] = -np.inf
        best_idx = int(np.argmax(scores))
        if scores[best_idx] <= 0:
            break
        code_indices.append(best_idx)
        selected_mask[best_idx] = True
        nb = neighbors[best_idx]
        gain = int(np.sum(uncovered[nb]))
        uncovered[nb] = 0
        total_uncovered -= gain

    # Ensure full coverage
    while total_uncovered > 0 and (time.time() - t_start) < deadline - 55.0:
        greedy_scores = A @ uncovered.astype(np.float64)
        greedy_scores[selected_mask] = -np.inf
        if np.max(greedy_scores) <= 0:
            break
        best_idx = int(np.argmax(greedy_scores))
        code_indices.append(best_idx)
        selected_mask[best_idx] = True
        nb = neighbors[best_idx]
        gain = int(np.sum(uncovered[nb]))
        uncovered[nb] = 0
        total_uncovered -= gain

    # Phase 1b: Greedy removal to tighten
    cov = np.zeros(space_size, dtype=np.int32)
    for c_idx in code_indices:
        cov[neighbors[c_idx]] += 1

    active = [True] * len(code_indices)
    changed = True
    while changed and (time.time() - t_start) < deadline - 50.0:
        changed = False
        for i in range(len(code_indices)):
            if not active[i]:
                continue
            c_idx = code_indices[i]
            if np.min(cov[neighbors[c_idx]]) >= 2:
                active[i] = False
                cov[neighbors[c_idx]] -= 1
                changed = True

    current_indices = [code_indices[i] for i in range(len(code_indices)) if active[i]]
    best_code = current_indices[:]
    best_size = len(best_code)

    # --- Phase 2: Simulated Annealing on the set cover ---
    # State: a set of codeword indices (as a boolean mask)
    # Moves:
    #   1-swap: remove one codeword, add one random codeword
    #   2-swap: remove two codewords, add two random codewords
    #   deletion: remove one codeword (if still valid)
    #   insertion: add one random codeword (to escape bad states)
    #
    # Fitness: size of the set (lower is better), with penalty for uncovered points
    # We track: coverage array, number of uncovered points

    def compute_coverage(indices):
        cov = np.zeros(space_size, dtype=np.int32)
        for c in indices:
            cov[neighbors[c]] += 1
        return cov

    def is_valid(cov):
        return np.all(cov >= 1)

    def get_uncovered_count(cov):
        return int(np.sum(cov == 0))

    # Current state
    cur_indices = set(best_code)
    cur_cov = compute_coverage(list(cur_indices))
    cur_uncovered = get_uncovered_count(cur_cov)
    cur_size = len(cur_indices)

    # Temperature schedule
    T_start = 2.0
    T_end = 0.05
    total_sa_time = deadline - (time.time() - t_start) - 10.0  # leave 10s for final cleanup
    if total_sa_time < 5.0:
        total_sa_time = 5.0

    # Precompute list of all indices for random sampling
    all_indices = list(range(space_size))

    # SA loop
    sa_step = 0
    total_sa_steps = 0
    best_seen = cur_size
    best_seen_indices = list(cur_indices)

    while (time.time() - t_start) < deadline - 10.0:
        elapsed_frac = 1.0 - (deadline - 10.0 - (time.time() - t_start)) / total_sa_time
        elapsed_frac = max(0.0, min(1.0, elapsed_frac))
        T = T_start * (T_end / T_start) ** elapsed_frac

        # Choose move type
        r = rng.random()
        if cur_size > 1 and r < 0.40:
            # 1-swap: remove one, add one
            if len(cur_indices) < 2:
                continue
            # Pick a random codeword to remove
            remove_list = list(cur_indices)
            rem = remove_list[int(rng.integers(0, len(remove_list)))]

            # Pick a random codeword to add (not already in set)
            new_c = int(rng.integers(0, space_size))
            if new_c in cur_indices:
                continue

            # Tentative: remove rem, add new_c
            # Effect on coverage:
            #   rem's neighbors lose 1
            #   new_c's neighbors gain 1
            rem_nb = neighbors[rem]
            new_nb = neighbors[new_c]

            # Compute change in uncovered count
            # For each position, old_cov = cur_cov[pos]
            # After: positions in rem_nb lose 1, positions in new_nb gain 1
            # Overlap positions: net 0 change
            # Only rem_nb \ new_nb: -1; only new_nb \ rem_nb: +1

            # Quick check: would this create new uncovered?
            # Positions where cur_cov == 1 and in rem_nb but NOT in new_nb -> become 0
            # Positions where cur_cov == 0 and in new_nb -> become 1 (good)

            # Vectorized:
            delta_cov = np.zeros(space_size, dtype=np.int32)
            delta_cov[rem_nb] -= 1
            delta_cov[new_nb] += 1

            new_cov_temp = cur_cov + delta_cov
            new_uncovered = int(np.sum(new_cov_temp == 0))
            new_size = cur_size  # same size (1 out, 1 in)

            # Acceptance
            delta_cost = (new_uncovered - cur_uncovered) * 10 + (new_size - cur_size)
            if delta_cost <= 0 or rng.random() < np.exp(-delta_cost / T):
                # Accept
                cur_cov = new_cov_temp
                cur_indices.discard(rem)
                cur_indices.add(new_c)
                cur_uncovered = new_uncovered
                cur_size = new_size

                if cur_uncovered == 0 and cur_size < best_seen:
                    best_seen = cur_size
                    best_seen_indices = list(cur_indices)

        elif r < 0.70:
            # Deletion: try to remove a redundant codeword
            if len(cur_indices) < 2:
                continue
            remove_list = list(cur_indices)
            rem = remove_list[int(rng.integers(0, len(remove_list)))]
            rem_nb = neighbors[rem]

            # Check if all neighbors have coverage >= 2
            if np.min(cur_cov[rem_nb]) >= 2:
                cur_cov[rem_nb] -= 1
                cur_indices.discard(rem)
                cur_size -= 1
                cur_uncovered = get_uncovered_count(cur_cov)  # should still be 0

                if cur_uncovered == 0 and cur_size < best_seen:
                    best_seen = cur_size
                    best_seen_indices = list(cur_indices)
            else:
                # Can't delete safely; try with SA acceptance if it creates few uncovered
                delta_cov = np.zeros(space_size, dtype=np.int32)
                delta_cov[rem_nb] -= 1
                new_cov_temp = cur_cov + delta_cov
                new_uncovered = int(np.sum(new_cov_temp == 0))
                new_size = cur_size - 1

                delta_cost = (new_uncovered - cur_uncovered) * 10 + (new_size - cur_size)
                if delta_cost <= 0 or rng.random() < np.exp(-delta_cost / T):
                    cur_cov = new_cov_temp
                    cur_indices.discard(rem)
                    cur_uncovered = new_uncovered
                    cur_size = new_size

                    if cur_uncovered == 0 and cur_size < best_seen:
                        best_seen = cur_size
                        best_seen_indices = list(cur_indices)

        elif r < 0.85:
            # Insertion: add a random codeword (useful if we're in an invalid state)
            new_c = int(rng.integers(0, space_size))
            if new_c in cur_indices:
                continue
            new_nb = neighbors[new_c]
            delta_cov = np.zeros(space_size, dtype=np.int32)
            delta_cov[new_nb] += 1
            new_cov_temp = cur_cov + delta_cov
            new_uncovered = int(np.sum(new_cov_temp == 0))
            new_size = cur_size + 1

            delta_cost = (new_uncovered - cur_uncovered) * 10 + (new_size - cur_size)
            if delta_cost <= 0 or rng.random() < np.exp(-delta_cost / T):
                cur_cov = new_cov_temp
                cur_indices.add(new_c)
                cur_uncovered = new_uncovered
                cur_size = new_size

                if cur_uncovered == 0 and cur_size < best_seen:
                    best_seen = cur_size
                    best_seen_indices = list(cur_indices)

        else:
            # 2-swap: remove two, add two
            if len(cur_indices) < 2:
                continue
            remove_list = list(cur_indices)
            idx_a = int(rng.integers(0, len(remove_list)))
            rem_a = remove_list[idx_a]
            remove_list.pop(idx_a)
            idx_b = int(rng.integers(0, len(remove_list)))
            rem_b = remove_list[idx_b]

            new_a = int(rng.integers(0, space_size))
            new_b = int(rng.integers(0, space_size))
            if new_a in cur_indices or new_b in cur_indices or new_a == new_b:
                continue

            delta_cov = np.zeros(space_size, dtype=np.int32)
            delta_cov[neighbors[rem_a]] -= 1
            delta_cov[neighbors[rem_b]] -= 1
            delta_cov[neighbors[new_a]] += 1
            delta_cov[neighbors[new_b]] += 1

            new_cov_temp = cur_cov + delta_cov
            new_uncovered = int(np.sum(new_cov_temp == 0))
            new_size = cur_size  # 2 out, 2 in

            delta_cost = (new_uncovered - cur_uncovered) * 10 + (new_size - cur_size)
            if delta_cost <= 0 or rng.random() < np.exp(-delta_cost / T):
                cur_cov = new_cov_temp
                cur_indices.discard(rem_a)
                cur_indices.discard(rem_b)
                cur_indices.add(new_a)
                cur_indices.add(new_b)
                cur_uncovered = new_uncovered
                cur_size = new_size

                if cur_uncovered == 0 and cur_size < best_seen:
                    best_seen = cur_size
                    best_seen_indices = list(cur_indices)

        total_sa_steps += 1

    # --- Phase 3: Final cleanup - restore validity and tighten ---
    # If best_seen_indices is valid, use it; otherwise fall back
    final_indices = best_seen_indices

    # Verify validity
    if len(final_indices) > 0:
        cov_check = np.zeros(space_size, dtype=np.int32)
        for c in final_indices:
            cov_check[neighbors[c]] += 1
        if np.any(cov_check == 0):
            # Try to repair: add codewords to cover holes
            unc = cov_check == 0
            while np.any(unc) and (time.time() - t_start) < deadline - 3.0:
                scores = A @ unc.astype(np.float64)
                best_idx = int(np.argmax(scores))
                if scores[best_idx] <= 0:
                    break
                final_indices.append(best_idx)
                cov_check[neighbors[best_idx]] += 1
                unc = cov_check == 0

    # Final greedy removal pass
    if len(final_indices) > 0:
        cov_final = np.zeros(space_size, dtype=np.int32)
        for c in final_indices:
            cov_final[neighbors[c]] += 1

        active = [True] * len(final_indices)
        changed = True
        while changed and (time.time() - t_start) < deadline - 2.0:
            changed = False
            for i in range(len(final_indices)):
                if not active[i]:
                    continue
                c = final_indices[i]
                if np.min(cov_final[neighbors[c]]) >= 2:
                    active[i] = False
                    cov_final[neighbors[c]] -= 1
                    changed = True

        final_indices = [final_indices[i] for i in range(len(final_indices)) if active[i]]

    # Final verification
    if len(final_indices) == 0:
        # Absolute fallback: return full space (valid but terrible)
        return [idx_to_word(i) for i in range(space_size)]

    cov_v = np.zeros(space_size, dtype=np.int32)
    for c in final_indices:
        cov_v[neighbors[c]] += 1
    if np.any(cov_v == 0):
        # Fallback to initial greedy code
        cov_init = np.zeros(space_size, dtype=np.int32)
        for i in code_indices:
            cov_init[neighbors[i]] += 1
        if not np.any(cov_init == 0):
            return [idx_to_word(i) for i in code_indices]
        return [idx_to_word(i) for i in range(space_size)]

    # Ensure distinctness
    seen = set()
    unique_indices = []
    for c in final_indices:
        if c not in seen:
            seen.add(c)
            unique_indices.append(c)

    return [idx_to_word(i) for i in unique_indices]
