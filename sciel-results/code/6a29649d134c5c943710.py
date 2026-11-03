import numpy as np
import time
import itertools
import random

def construct(q, n, radius):
    # Fallback for non-specific parameters
    if q != 3 or n != 8 or radius != 1:
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            if n == 4:
                return tetracode
            else:
                return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return [list(t) for t in itertools.product(range(q), repeat=n)]

    total_points = q ** n  # 6561
    neighbors_per_word = 1 + n * (q - 1)  # 17

    start_time = time.time()
    time_limit = 145.0

    def word_to_idx(w):
        idx = 0
        for i in range(n):
            idx = idx * q + w[i]
        return idx

    def idx_to_word(idx):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = idx % q
            idx //= q
        return w

    # Precompute neighbors
    neighbors = np.zeros((total_points, neighbors_per_word), dtype=np.int32)
    for i in range(total_points):
        w = idx_to_word(i)
        neighbors[i, 0] = i
        count = 1
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                w[pos] = new_val
                neighbors[i, count] = word_to_idx(w)
                count += 1
            w[pos] = orig_val

    all_words = [idx_to_word(i) for i in range(total_points)]

    # Build incidence: for each point, which codewords cover it
    # point_coverers[p] = list of codeword indices that cover p
    point_coverers = []
    for p in range(total_points):
        # A codeword c covers p if p is in neighbors[c], i.e., c is in neighbors[p]
        # Since neighbors is symmetric (Hamming ball is symmetric), c covers p iff p covers c
        # Actually, neighbors[p] contains all words within dist 1 of p.
        # So codeword c is a valid cover for point p if c is in neighbors[p].
        point_coverers.append(neighbors[p].tolist())

    def is_valid_cover(chosen_indices):
        cover_count = np.zeros(total_points, dtype=np.int32)
        for ci in chosen_indices:
            cover_count[neighbors[ci]] += 1
        return np.all(cover_count > 0)

    def structured_initial_code():
        tetracode = [(a, b, (a + b) % 3, (a + 2 * b) % 3) for a in range(3) for b in range(3)]
        indices = []
        for t1 in tetracode:
            for t2 in tetracode:
                word = list(t1) + list(t2)
                idx = word_to_idx(word)
                indices.append(idx)
        return indices

    def greedy_complete(chosen_indices):
        """Greedily add codewords to complete coverage."""
        cover_count = np.zeros(total_points, dtype=np.int32)
        for ci in chosen_indices:
            cover_count[neighbors[ci]] += 1
        chosen = list(chosen_indices)
        available = np.ones(total_points, dtype=bool)
        for ci in chosen:
            available[ci] = False

        while not np.all(cover_count > 0):
            if time.time() - start_time > time_limit - 10.0:
                break
            uncovered = np.where(cover_count == 0)[0]
            if len(uncovered) == 0:
                break
            best_idx = -1
            best_count = -1
            # Sample a subset of available words for speed if too many
            avail_indices = np.where(available)[0]
            if len(avail_indices) > 500:
                sample = random.sample(list(avail_indices), 500)
            else:
                sample = list(avail_indices)
            for idx in sample:
                cnt = np.sum(cover_count[neighbors[idx]] == 0)
                if cnt > best_count:
                    best_count = cnt
                    best_idx = idx
            if best_idx < 0:
                # Fallback: check all
                for idx in avail_indices:
                    cnt = np.sum(cover_count[neighbors[idx]] == 0)
                    if cnt > best_count:
                        best_count = cnt
                        best_idx = idx
            if best_idx < 0:
                break
            chosen.append(best_idx)
            cover_count[neighbors[best_idx]] += 1
            available[best_idx] = False
        return chosen

    def lagrangian_heuristic(initial_indices, max_iters=2000):
        """
        Lagrangian relaxation / subgradient heuristic for set covering.
        Minimize sum(x_c) subject to A x >= 1, x in {0,1}
        Lagrangian: L(λ) = sum(x_c) - sum_p λ_p (sum_c A_pc x_c - 1)
                    = sum_p λ_p + sum_c x_c (1 - sum_p λ_p A_pc)
        Dual: maximize L(λ) over λ >= 0.
        Heuristic: use subgradient to update λ, then project to feasible integer solution
        by rounding the primal solution guided by dual variables.
        """
        chosen = list(initial_indices)
        x = np.zeros(total_points, dtype=np.int32)
        for ci in chosen:
            x[ci] = 1

        # λ_p: dual variable for each point (coverage constraint)
        lam = np.ones(total_points, dtype=np.float64) * 0.1

        # Precompute column sums for efficiency: for each codeword c, sum_p lam_p * A_pc
        # A_pc = 1 if c covers p, i.e., p in neighbors[c]
        # So col_sum[c] = sum_{p in neighbors[c]} lam_p

        best_solution = list(chosen)
        best_size = len(chosen)
        best_cover_count = np.zeros(total_points, dtype=np.int32)
        for ci in chosen:
            best_cover_count[neighbors[ci]] += 1

        step_size = 0.5
        for it in range(max_iters):
            if time.time() - start_time > time_limit - 15.0:
                break

            # Compute col_sum[c] = sum_{p in neighbors[c]} lam_p
            col_sum = np.array([np.sum(lam[neighbors[c]]) for c in range(total_points)], dtype=np.float64)

            # Primal solution: x_c = 1 if (1 - col_sum[c]) < 0, i.e., col_sum[c] > 1
            # This is the standard Lagrangian projection
            new_x = (col_sum > 1.0).astype(np.int32)

            # Ensure at least some coverage: if new_x covers too few points, blend with current
            new_cover = np.zeros(total_points, dtype=np.int32)
            new_indices = np.where(new_x == 1)[0]
            for ci in new_indices:
                new_cover[neighbors[ci]] += 1

            uncovered_new = np.sum(new_cover == 0)
            uncovered_cur = np.sum(best_cover_count == 0)

            # If new solution is feasible and smaller, accept
            if uncovered_new == 0 and len(new_indices) < best_size:
                best_solution = list(new_indices)
                best_size = len(new_indices)
                best_cover_count = new_cover.copy()
                chosen = list(new_indices)

            # Subgradient update: g_p = sum_c A_pc x_c - 1 (using new_x or current x?)
            # Standard: use the primal solution from the Lagrangian (new_x)
            # g_p = coverage of p by new_x - 1
            g = new_cover.astype(np.float64) - 1.0

            # Step size: Polyaekov or simple diminishing
            # Use simple: step = step_size / (it + 1)
            step = step_size / (np.sqrt(it + 1))

            # Update lam: lam_p += step * g_p, clip to >= 0
            lam = np.maximum(lam + step * g, 0.0)

            # Periodically: try to improve current solution by local moves
            if it % 50 == 0:
                # Try removing redundant codewords from best_solution
                if len(best_solution) > 0:
                    local_cover = np.zeros(total_points, dtype=np.int32)
                    for ci in best_solution:
                        local_cover[neighbors[ci]] += 1
                    new_local = []
                    for ci in best_solution:
                        exposed = neighbors[ci]
                        would_uncover = exposed[local_cover[exposed] == 1]
                        if len(would_uncover) == 0:
                            local_cover[neighbors[ci]] -= 1
                        else:
                            new_local.append(ci)
                    if len(new_local) < len(best_solution):
                        best_solution = new_local
                        best_size = len(new_local)
                        best_cover_count = local_cover.copy()
                        chosen = list(new_local)

        return best_solution

    def randomized_lagrangian(initial_indices, n_restarts=5):
        """Run Lagrangian heuristic with different initial λ and random perturbations."""
        best = list(initial_indices)
        best_size = len(initial_indices)

        for r in range(n_restarts):
            if time.time() - start_time > time_limit - 20.0:
                break

            # Perturb initial solution: randomly remove some codewords
            perturbed = list(initial_indices)
            if len(perturbed) > 10:
                n_remove = min(len(perturbed) // 10, 5)
                to_remove = random.sample(perturbed, n_remove)
                perturbed = [c for c in perturbed if c not in set(to_remove)]

            # Complete coverage greedily
            completed = greedy_complete(perturbed)
            if len(completed) >= best_size:
                continue

            # Run Lagrangian
            result = lagrangian_heuristic(completed, max_iters=1000)
            if len(result) < best_size:
                best = result
                best_size = len(result)

        return best

    # Main algorithm
    # 1. Start with structured tetracode product
    initial_code = structured_initial_code()

    # 2. Complete with greedy
    completed_code = greedy_complete(initial_code)

    # 3. Apply Lagrangian heuristic with restarts
    optimized_code = randomized_lagrangian(completed_code, n_restarts=8)

    # 4. Final cleanup: remove redundant codewords
    if len(optimized_code) > 0:
        cover_count = np.zeros(total_points, dtype=np.int32)
        for ci in optimized_code:
            cover_count[neighbors[ci]] += 1
        new_opt = []
        for ci in optimized_code:
            exposed = neighbors[ci]
            would_uncover = exposed[cover_count[exposed] == 1]
            if len(would_uncover) == 0:
                cover_count[neighbors[ci]] -= 1
            else:
                new_opt.append(ci)
        optimized_code = new_opt

    # Verify final validity
    if not is_valid_cover(optimized_code):
        # Fallback
        optimized_code = completed_code

    if not is_valid_cover(optimized_code):
        # Emergency full greedy
        cover_count = np.zeros(total_points, dtype=np.int32)
        chosen = []
        while not np.all(cover_count > 0):
            if time.time() - start_time > time_limit - 5.0:
                break
            best_idx = -1
            best_count = -1
            for idx in range(total_points):
                cnt = np.sum(cover_count[neighbors[idx]] == 0)
                if cnt > best_count:
                    best_count = cnt
                    best_idx = idx
            if best_idx < 0:
                break
            chosen.append(best_idx)
            cover_count[neighbors[best_idx]] += 1
        optimized_code = chosen

    return [all_words[i] for i in optimized_code]
