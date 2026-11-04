import time
import numpy as np

def construct(q, n, radius):
    t_start = time.time()
    deadline = 148.0

    space_size = q ** n
    q_powers = [1] * (n + 1)
    for i in range(1, n + 1):
        q_powers[i] = q_powers[i - 1] * q

    def idx_to_word(idx):
        w = [0] * n
        for pos in range(n):
            w[pos] = (idx // q_powers[n - 1 - pos]) % q
        return w

    # Precompute neighbors (radius=1: self + all 1-change neighbors)
    nb_count = 1 + n * (q - 1)
    neighbors = np.empty((space_size, nb_count), dtype=np.int32)
    for idx in range(space_size):
        w = [0] * n
        for pos in range(n):
            w[pos] = (idx // q_powers[n - 1 - pos]) % q
        nb_list = [idx]
        for pos in range(n):
            for new_d in range(q):
                if new_d == w[pos]:
                    continue
                delta = (new_d - w[pos]) * q_powers[n - 1 - pos]
                nb_list.append(idx + delta)
        neighbors[idx] = nb_list

    def build_cov(code_indices):
        cov = np.zeros(space_size, dtype=np.int32)
        for c in code_indices:
            cov[neighbors[c]] += 1
        return cov

    def greedy_cover(seed=0):
        """Standard greedy set cover."""
        rng = np.random.default_rng(seed)
        uncovered = np.ones(space_size, dtype=bool)
        code = []
        while np.any(uncovered) and (time.time() - t_start) < deadline - 30.0:
            gains = np.sum(uncovered[neighbors], axis=1)
            perturb = rng.random(space_size) * 0.001
            gains_pert = gains + perturb
            best = np.argmax(gains_pert)
            if gains[best] == 0:
                break
            code.append(int(best))
            uncovered[neighbors[best]] = False
        return code

    def greedy_repair(code_list, cov, deadline_time):
        uncovered_mask = cov == 0
        while np.any(uncovered_mask) and (time.time() - t_start) < deadline_time:
            gains = np.sum(uncovered_mask[neighbors], axis=1)
            best = np.argmax(gains)
            if gains[best] == 0:
                break
            code_list.append(int(best))
            cov[neighbors[best]] += 1
            uncovered_mask = cov == 0
        return code_list, cov

    def local_search_removal(code, cov, deadline_time):
        """Remove redundant codewords (those whose removal doesn't uncover any node)."""
        cur_code = code[:]
        cur_cov = cov.copy()
        for _ in range(200):
            if (time.time() - t_start) > deadline_time:
                break
            changed = False
            for i in range(len(cur_code)):
                c = cur_code[i]
                if np.min(cur_cov[neighbors[c]]) >= 2:
                    cur_code.pop(i)
                    cur_cov[neighbors[c]] -= 1
                    changed = True
                    break
            if not changed:
                break
        return cur_code, cur_cov

    def iterated_local_search(initial_code, max_iters=50, deadline_time=None):
        """
        ILS: repeatedly try swap and delete moves.
        Swap: remove one codeword, add another (net size unchanged or -1 if added was already covered).
        Delete: remove one codeword if coverage stays valid.
        Accept any move that reduces size; also accept swaps that keep size but might enable future deletes.
        """
        if deadline_time is None:
            deadline_time = deadline - 5.0

        cur_code = list(initial_code)
        cur_cov = build_cov(cur_code)
        cur_size = len(cur_code)
        best_code = cur_code[:]
        best_size = cur_size

        # Track which codewords are in the set for O(1) lookup
        in_set = np.zeros(space_size, dtype=bool)
        for c in cur_code:
            in_set[c] = True

        for it in range(max_iters):
            if (time.time() - t_start) > deadline_time:
                break

            improved = False

            # --- Delete pass: try removing each codeword ---
            for i in range(len(cur_code)):
                if (time.time() - t_start) > deadline_time:
                    break
                c = cur_code[i]
                if np.min(cur_cov[neighbors[c]]) >= 2:
                    # Safe to remove
                    cur_code.pop(i)
                    cur_cov[neighbors[c]] -= 1
                    in_set[c] = False
                    cur_size -= 1
                    improved = True
                    if cur_size < best_size:
                        best_size = cur_size
                        best_code = cur_code[:]
                    break  # restart delete pass

            if not improved:
                # --- Swap pass: try removing one and adding another ---
                # For each candidate removal, find best addition
                # We sample a subset of removals for speed
                n_removals = min(len(cur_code), 200)
                rng = np.random.default_rng(42 + it)
                removal_candidates = rng.choice(len(cur_code), size=n_removals, replace=False)

                best_swap = None
                best_swap_gain = 0

                for ri in removal_candidates:
                    if (time.time() - t_start) > deadline_time:
                        break
                    c_remove = cur_code[ri]
                    # If we remove c_remove, which nodes become uncovered?
                    # Nodes where cov == 1 and c_remove is the only cover
                    # Actually: nodes in neighbors[c_remove] where cur_cov[node] == 1
                    would_uncover = (cur_cov[neighbors[c_remove]] == 1) & in_set[c_remove]
                    # More precisely: after removal, cov[node] decreases by 1 for nodes in neighbors[c_remove]
                    # A node becomes uncovered if cur_cov[node] == 1 and c_remove covers it
                    # Since in_set[c_remove] is True, it does cover all in neighbors[c_remove]
                    # So nodes with cur_cov == 1 in neighbors[c_remove] would be uncovered
                    uncovered_if_removed = (cur_cov[neighbors[c_remove]] == 1)

                    if not np.any(uncovered_if_removed):
                        # Pure delete (should have been caught above, but just in case)
                        continue

                    # We need to add a codeword that covers all of uncovered_if_removed
                    # Gain of adding candidate x: covers how many of uncovered_if_removed?
                    # But we also want x to not be already in set
                    target_mask = np.zeros(space_size, dtype=bool)
                    target_mask[neighbors[c_remove]] = uncovered_if_removed

                    # For speed, only consider candidates that are not in set
                    # Compute coverage of target for all candidates
                    # This is expensive; instead, sample candidates
                    n_samples = min(space_size, 500)
                    # Prefer candidates outside current set
                    outside_mask = ~in_set
                    outside_indices = np.where(outside_mask)[0]
                    if len(outside_indices) < n_samples:
                        sample_idx = outside_indices
                    else:
                        sample_idx = rng.choice(outside_indices, size=n_samples, replace=False)

                    # For each sample, compute how many target nodes it covers
                    # target_count[x] = sum(target_mask[neighbors[x]])
                    gains = np.sum(target_mask[neighbors[sample_idx]], axis=1)
                    total_target = np.sum(target_mask)

                    best_local = np.argmax(gains)
                    if gains[best_local] >= total_target:
                        # This swap fully covers the gap
                        c_add = int(sample_idx[best_local])
                        # Check if c_add is already in set (shouldn't be, but verify)
                        if not in_set[c_add]:
                            # Perform swap
                            cur_code[ri] = c_add
                            # Update coverage: remove c_remove, add c_add
                            cur_cov[neighbors[c_remove]] -= 1
                            cur_cov[neighbors[c_add]] += 1
                            in_set[c_remove] = False
                            in_set[c_add] = True
                            improved = True
                            break

                if improved:
                    # After a swap, do a quick delete pass
                    for i in range(len(cur_code)):
                        if (time.time() - t_start) > deadline_time:
                            break
                        c = cur_code[i]
                        if np.min(cur_cov[neighbors[c]]) >= 2:
                            cur_code.pop(i)
                            cur_cov[neighbors[c]] -= 1
                            in_set[c] = False
                            cur_size -= 1
                            if cur_size < best_size:
                                best_size = cur_size
                                best_code = cur_code[:]
                            break

            if not improved:
                # No improvement found; try a kick: randomly remove k codewords and repair
                if len(cur_code) > 5:
                    k = min(3, len(cur_code) // 10 + 1)
                    kick_idx = rng.choice(len(cur_code), size=k, replace=False)
                    kick_set = set(int(ri) for ri in kick_idx)
                    new_code = [c for i, c in enumerate(cur_code) if i not in kick_set]
                    new_cov = build_cov(new_code)
                    new_code, new_cov = greedy_repair(new_code, new_cov, deadline_time)
                    # Only accept if size didn't increase
                    if len(new_code) <= cur_size:
                        cur_code = new_code
                        cur_cov = new_cov
                        cur_size = len(cur_code)
                        in_set = np.zeros(space_size, dtype=bool)
                        for c in cur_code:
                            in_set[c] = True
                        if cur_size < best_size:
                            best_size = cur_size
                            best_code = cur_code[:]

        return best_code, best_size

    # Phase 1: Generate multiple greedy initial solutions
    best_code = None
    best_size = space_size + 1

    for seed in range(8):
        if (time.time() - t_start) > deadline - 60.0:
            break

        code = greedy_cover(seed)
        cov = build_cov(code)
        if not np.all(cov > 0):
            code, cov = greedy_repair(code, cov, deadline - 50.0)

        # Local search removal
        code, cov = local_search_removal(code, cov, deadline - 50.0)

        if len(code) < best_size:
            best_code = code[:]
            best_size = len(code)

    # Phase 2: ILS on the best greedy solution
    if best_code is not None and (time.time() - t_start) < deadline - 20.0:
        code, size = iterated_local_search(best_code, max_iters=80, deadline_time=deadline - 10.0)
        if size < best_size:
            best_code = code
            best_size = size

    # Phase 3: ILS on a few other greedy solutions
    for seed in range(3, 8):
        if (time.time() - t_start) > deadline - 25.0:
            break

        code = greedy_cover(seed)
        cov = build_cov(code)
        if not np.all(cov > 0):
            code, cov = greedy_repair(code, cov, deadline - 20.0)

        code, cov = local_search_removal(code, cov, deadline - 20.0)

        if len(code) < best_size:
            best_code = code[:]
            best_size = len(code)

        if (time.time() - t_start) < deadline - 15.0:
            code2, size2 = iterated_local_search(code, max_iters=40, deadline_time=deadline - 10.0)
            if size2 < best_size:
                best_code = code2
                best_size = size2

    # Final aggressive removal
    if (time.time() - t_start) < deadline - 3.0 and best_code is not None:
        cur_code = best_code[:]
        cur_cov = build_cov(cur_code)
        cur_code, cur_cov = local_search_removal(cur_code, cur_cov, deadline - 2.0)
        if len(cur_code) < best_size:
            best_code = cur_code[:]
            best_size = len(cur_code)

    # Final verification
    if best_code is None or len(best_code) == 0:
        best_code = greedy_cover(0)

    final_cov = build_cov(best_code)
    if not np.all(final_cov > 0):
        fc = greedy_cover(0)
        fcov = build_cov(fc)
        if np.all(fcov > 0):
            best_code = fc
        else:
            best_code = list(range(space_size))

    return [idx_to_word(c) for c in best_code]
