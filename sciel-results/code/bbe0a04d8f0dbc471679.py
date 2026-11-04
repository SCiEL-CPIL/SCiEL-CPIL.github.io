import time
import numpy as np
import random

def construct(q, n, radius):
    t_start = time.time()
    deadline = 115.0  # Safety margin before 120s limit

    space_size = q ** n
    q_powers = [1] * (n + 1)
    for i in range(1, n + 1):
        q_powers[i] = q_powers[i-1] * q

    def word_to_idx(w):
        idx = 0
        for d in w:
            idx = idx * q + d
        return idx

    def idx_to_word(idx):
        w = [0] * n
        for pos in range(n):
            w[pos] = (idx // q_powers[n-1-pos]) % q
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
                delta = (new_d - w[pos]) * q_powers[n-1-pos]
                nb_list.append(idx + delta)
        neighbors[idx] = nb_list

    # --- Phase 1: Lagrangian Relaxation with Subgradient Updates ---
    # The covering problem: min sum(x_v) s.t. for all u: sum_{v in N(u)} x_v >= 1, x_v in {0,1}
    # Lagrangian: L(x, lambda) = sum_v x_v + sum_u lambda_u * (1 - sum_{v in N(u)} x_v)
    #             = sum_u lambda_u + sum_v x_v * (1 - sum_{u in N(v)} lambda_u)
    # Reduced cost of v: rc_v = 1 - sum_{u in N(v)} lambda_u
    # We select v with highest rc_v (most negative contribution to Lagrangian), i.e., most beneficial.
    # Actually we want to minimize L, so we pick v that minimizes rc_v (most negative).
    # But for set cover intuition: we want to pick v that covers many uncovered points.
    # Let's use: score_v = sum_{u in N(v)} uncovered_u - sum_{u in N(v)} lambda_u
    # This blends greedy coverage with dual guidance.

    rng = np.random.default_rng(42)

    # Initialize
    uncovered = np.ones(space_size, dtype=np.int32)
    total_uncovered = space_size
    code_indices = []
    selected_mask = np.zeros(space_size, dtype=bool)

    # Dual variables
    lam = np.zeros(space_size, dtype=np.float64)

    # Precompute: for each candidate v, which vertices does it cover? That's neighbors[v].
    # score_v = sum(uncovered[neighbors[v]]) - sum(lam[neighbors[v]])
    # = uncovered @ neighbors_mask_v - lam @ neighbors_mask_v
    # We can precompute a matrix M where M[v, u] = 1 if u in N(v), else 0.
    # Then score = M @ (uncovered - lam)
    # But M is space_size x space_size = 2187 x 2187 ~ 4.8M entries. Fine.

    # Build adjacency matrix (incidence of covering relation)
    # A[v, u] = 1 if u is within distance radius of v
    A = np.zeros((space_size, space_size), dtype=np.int32)
    for v in range(space_size):
        A[v, neighbors[v]] = 1

    # Now score_v = A[v, :] @ (uncovered - lam)
    # We want max score_v (covers most uncovered, adjusted by duals)

    while total_uncovered > 0 and (time.time() - t_start) < deadline - 20.0:
        # Compute scores
        residual = uncovered.astype(np.float64) - lam
        scores = A @ residual  # shape (space_size,)

        # Mask out already selected
        scores[selected_mask] = -np.inf

        best_idx = int(np.argmax(scores))
        best_score = scores[best_idx]

        if best_score <= 0:
            # No positive gain even with dual adjustment; force greedy pick
            greedy_scores = A @ uncovered.astype(np.float64)
            greedy_scores[selected_mask] = -np.inf
            best_idx = int(np.argmax(greedy_scores))
            best_score = greedy_scores[best_idx]
            if best_score <= 0:
                break

        # Select this codeword
        code_indices.append(best_idx)
        selected_mask[best_idx] = True

        # Update uncovered
        nb = neighbors[best_idx]
        newly_covered = uncovered[nb]
        gain = int(np.sum(newly_covered))
        uncovered[nb] = 0
        total_uncovered -= gain

        # Subgradient update for dual variables
        # Subgradient of L w.r.t. lambda_u is (1 - sum_{v in N(u): x_v=1})
        # If u is uncovered, subgrad = 1; if covered, subgrad = 0 or negative
        # Standard subgradient: g_u = 1 - sum_{v selected, u in N(v)}
        # For uncovered u: g_u = 1 (since no selected v covers it... wait, that's not right)
        # Actually after selection, some previously uncovered may now be covered.
        # Let's compute: for each u, coverage_count = sum of selected v that cover u
        # g_u = 1 - coverage_count[u]
        # But we only update duals periodically to avoid oscillation.

        # Update duals: lam_u += step * (1 - cov_count[u]) for uncovered, or more carefully
        # Simpler: lam_u += step * (1 if u still uncovered else 0)
        # This increases dual for uncovered vertices, making them more "expensive" to leave uncovered,
        # which pushes the score of candidates covering them higher.

        # Adaptive step size
        elapsed = time.time() - t_start
        step = 0.5 / (1.0 + 0.1 * len(code_indices))

        # Only update for currently uncovered vertices
        unc_mask = uncovered.astype(bool)
        if np.any(unc_mask):
            lam[unc_mask] += step

        # Occasionally renormalize duals to prevent drift
        if len(code_indices) % 10 == 0:
            max_lam = np.max(lam)
            if max_lam > 5.0:
                lam *= 0.5

    # Fallback: if not fully covered, add greedy picks
    while total_uncovered > 0 and (time.time() - t_start) < deadline - 15.0:
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

    # --- Phase 2: Greedy Removal (Redundancy Check) ---
    cov = np.zeros(space_size, dtype=np.int32)
    for c_idx in code_indices:
        cov[neighbors[c_idx]] += 1

    active = [True] * len(code_indices)
    changed = True
    while changed and (time.time() - t_start) < deadline - 10.0:
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
    current_code = [idx_to_word(i) for i in current_indices]
    best_code = current_code[:]
    best_size = len(best_code)

    # --- Phase 3: Aggressive 1-Swap Local Search ---
    # Try to replace one codeword with another to allow removal of a second codeword.
    # Or: try removing a codeword; if invalid, find a replacement that fixes the holes.

    np_random = np.random.default_rng(42)

    for restart in range(50):
        if time.time() - t_start > deadline - 5.0:
            break

        cur_indices = best_code[:]
        cur_size = len(cur_indices)

        # Build coverage
        cov = np.zeros(space_size, dtype=np.int32)
        for c in cur_indices:
            c_idx = word_to_idx(c)
            cov[neighbors[c_idx]] += 1

        # Try to remove each codeword in random order
        order = list(range(len(cur_indices)))
        np_random.shuffle(order)

        improved = False
        for i in order:
            if time.time() - t_start > deadline - 5.0:
                break
            c = cur_indices[i]
            c_idx = word_to_idx(c)
            nbs = neighbors[c_idx]

            if np.min(cov[nbs]) >= 2:
                # Safe to remove
                cov[nbs] -= 1
                cur_indices[i] = None
                improved = True

        # Filter
        new_indices = [c for c in cur_indices if c is not None]
        if len(new_indices) < best_size:
            best_code = new_indices
            best_size = len(best_code)
            continue

        # 1-Swap: try replacing one codeword with a random other to enable further removal
        if len(best_code) > 1 and (time.time() - t_start) < deadline - 5.0:
            # Pick a random codeword to potentially replace
            swap_attempts = 0
            for _ in range(20):
                if time.time() - t_start > deadline - 5.0:
                    break
                swap_attempts += 1

                # Pick a random codeword in best_code
                pos = int(np_random.integers(0, len(best_code)))
                old_c = best_code[pos]
                old_idx = word_to_idx(old_c)

                # Pick a random replacement
                new_idx = int(np_random.integers(0, space_size))
                new_c = idx_to_word(new_idx)

                # Try: remove old, add new, then try to remove another
                test_indices = best_code[:]
                test_indices[pos] = new_c

                # Check validity
                cov_test = np.zeros(space_size, dtype=np.int32)
                for c in test_indices:
                    ci = word_to_idx(c)
                    cov_test[neighbors[ci]] += 1

                if np.any(cov_test == 0):
                    continue

                # Try to remove additional codewords
                changed2 = True
                while changed2:
                    changed2 = False
                    for j in range(len(test_indices)):
                        if time.time() - t_start > deadline - 5.0:
                            break
                        cj = test_indices[j]
                        cij = word_to_idx(cj)
                        if np.min(cov_test[neighbors[cij]]) >= 2:
                            cov_test[neighbors[cij]] -= 1
                            test_indices[j] = None
                            changed2 = True

                final_test = [c for c in test_indices if c is not None]
                if len(final_test) < best_size:
                    # Verify
                    cov_v = np.zeros(space_size, dtype=np.int32)
                    for c in final_test:
                        ci = word_to_idx(c)
                        cov_v[neighbors[ci]] += 1
                    if not np.any(cov_v == 0):
                        best_code = final_test
                        best_size = len(best_code)
                        break

    # Final Verification
    if best_size == 0:
        return [idx_to_word(i) for i in code_indices]

    cov_final = np.zeros(space_size, dtype=np.int32)
    for c in best_code:
        c_idx = word_to_idx(c)
        cov_final[neighbors[c_idx]] += 1

    if np.any(cov_final == 0):
        # Fallback to initial greedy code
        cov_init = np.zeros(space_size, dtype=np.int32)
        for i in code_indices:
            cov_init[neighbors[i]] += 1
        if not np.any(cov_init == 0):
            return [idx_to_word(i) for i in code_indices]
        return [idx_to_word(i) for i in range(space_size)]

    return best_code
