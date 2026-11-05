"""
Optimized Ternary Covering Code K_3(8,1) Solver.
Uses Randomized Greedy Set Cover with multiple restarts, aggressive pruning,
and a local-search swap phase for further reduction.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
import random
from collections import Counter

def construct(q, n, radius):
    # The task specifies q=3, n=8, radius=1.
    if q != 3 or n != 8 or radius != 1:
        # Fallback to simple construction if parameters differ
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            all_words = list(itertools.product(range(q), repeat=n))
            return all_words

    total_points = q ** n  # 6561
    ball_size = 1 + n * (q - 1)  # 17

    # Helper to convert index to word
    def idx_to_word(idx):
        w = [0] * n
        for i in range(n):
            w[i] = idx % q
            idx //= q
        return w

    # Pre-compute coverage sets
    coverage_sets = np.empty((total_points, ball_size), dtype=np.int32)
    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i-1] * q

    for idx in range(total_points):
        temp_idx = idx
        word = [0] * n
        for i in range(n):
            word[i] = temp_idx % q
            temp_idx //= q

        coverage_sets[idx, 0] = idx
        neighbor_idx = 1
        for pos in range(n):
            orig_digit = word[pos]
            power = powers[pos]

            # +1
            new_digit = (orig_digit + 1) % q
            if new_digit == 0:
                delta = -2 * power
            else:
                delta = power
            coverage_sets[idx, neighbor_idx] = idx + delta
            neighbor_idx += 1

            # -1
            new_digit = (orig_digit - 1) % q
            if new_digit == 2:
                delta = 2 * power
            else:
                delta = -power
            coverage_sets[idx, neighbor_idx] = idx + delta
            neighbor_idx += 1

    # Precompute ball membership: for each point, which codewords cover it
    # ball_membership[point] = list of codeword indices whose ball contains point
    # This is the transpose of coverage_sets
    ball_membership = [[] for _ in range(total_points)]
    for cw in range(total_points):
        for pt in coverage_sets[cw]:
            ball_membership[pt].append(cw)

    def run_greedy(seed, use_weighted=False):
        """Runs a single randomized greedy set cover pass."""
        rng = np.random.default_rng(seed)
        is_covered = np.zeros(total_points, dtype=bool)
        selected = []

        while not np.all(is_covered):
            uncovered_counts = np.sum(~is_covered[coverage_sets], axis=1)

            max_count = np.max(uncovered_counts)
            if max_count == 0:
                break

            candidates = np.where(uncovered_counts == max_count)[0]

            if len(candidates) > 1:
                if use_weighted:
                    # Weight by inverse of number of times this candidate has been "close" to being picked
                    # Simple heuristic: prefer candidates that cover more uncovered points
                    best_idx = rng.choice(candidates)
                else:
                    best_idx = rng.choice(candidates)
            else:
                best_idx = candidates[0]

            selected.append(best_idx)
            is_covered[coverage_sets[best_idx]] = True

        return selected

    def prune(selected_indices):
        """Removes redundant codewords from the selected set."""
        if len(selected_indices) == 0:
            return selected_indices

        selected_set = set(selected_indices)
        coverage_count = np.zeros(total_points, dtype=np.int32)
        for idx in selected_set:
            coverage_count[coverage_sets[idx]] += 1

        changed = True
        while changed:
            changed = False
            for idx in list(selected_set):
                ball = coverage_sets[idx]
                if np.all(coverage_count[ball] > 1):
                    selected_set.remove(idx)
                    coverage_count[ball] -= 1
                    changed = True

        return list(selected_set)

    def local_search_swap(selected_indices, time_limit=5.0):
        """
        Local search: try to remove one codeword and replace with one or two others
        to reduce total count or maintain count with better coverage.
        Actually, for covering codes, we want to MINIMIZE the number of codewords.
        So we try: remove codeword A, then see if we can add one codeword B that
        covers all points that A uniquely covered. If yes, we've reduced size by 0
        (removed 1, added 1) but that's not a reduction.

        Better approach: try to find a codeword that can be removed without
        breaking coverage (already done in prune). Then try to find a pair
        (remove A, remove B, add C) that reduces size by 1.
        """
        start_time = time.time()
        selected_set = set(selected_indices)
        coverage_count = np.zeros(total_points, dtype=np.int32)
        for idx in selected_set:
            coverage_count[coverage_sets[idx]] += 1

        # First, do another prune pass
        changed = True
        while changed:
            changed = False
            for idx in list(selected_set):
                ball = coverage_sets[idx]
                if np.all(coverage_count[ball] > 1):
                    selected_set.remove(idx)
                    coverage_count[ball] -= 1
                    changed = True

        # Now try 2-for-1 swaps: remove A and B, add C
        # For each pair (A, B) in selected_set, find points uniquely covered by A or B
        # Then find a C that covers all those points
        # This is expensive, so we do it selectively

        improved = True
        while improved and time.time() - start_time < time_limit:
            improved = False
            selected_list = list(selected_set)
            if len(selected_list) < 2:
                break

            # Sample a subset of pairs to check (for speed)
            num_pairs_to_check = min(500, len(selected_list) * (len(selected_list) - 1) // 2)
            pair_indices = []
            for i in range(len(selected_list)):
                for j in range(i + 1, len(selected_list)):
                    pair_indices.append((i, j))

            if len(pair_indices) > num_pairs_to_check:
                rng = np.random.default_rng(hash(tuple(sorted(selected_set))) % 2**32)
                pair_indices = [pair_indices[i] for i in rng.choice(len(pair_indices), num_pairs_to_check, replace=False)]

            for i, j in pair_indices:
                if time.time() - start_time > time_limit:
                    break

                A = selected_list[i]
                B = selected_list[j]

                # Find points uniquely covered by A or B (coverage_count == 1 and covered by A or B)
                ball_A = coverage_sets[A]
                ball_B = coverage_sets[B]

                # Points that are in ball_A or ball_B and have coverage_count == 1
                # These are points that would become uncovered if we remove both A and B
                # But we need to be careful: a point might be covered by both A and B,
                # so removing both might uncover it if no other codeword covers it.

                # Let's compute the set of points that would be uncovered
                # if we remove A and B
                temp_coverage = coverage_count.copy()
                temp_coverage[ball_A] -= 1
                temp_coverage[ball_B] -= 1

                # Points that become uncovered
                uncovered_points = np.where(temp_coverage == 0)[0]

                if len(uncovered_points) == 0:
                    # We can remove both A and B!
                    selected_set.remove(A)
                    selected_set.remove(B)
                    coverage_count[ball_A] -= 1
                    coverage_count[ball_B] -= 1
                    improved = True
                    break

                # Try to find a single codeword C that covers all uncovered_points
                # C must be such that all uncovered_points are in ball_C
                # This means C must be within distance 1 of each uncovered point
                # For multiple uncovered points, this is restrictive

                if len(uncovered_points) <= 3:
                    # For small sets, we can check all candidates
                    # A candidate C covers all uncovered points if
                    # for each p in uncovered_points, dist(C, p) <= 1
                    # Equivalently, C is in the intersection of balls around each uncovered point
                    # The intersection of balls around points at distance <= 1
                    # is the set of points within distance 1 of ALL of them

                    # For a single point, the ball has 17 points
                    # For two points at distance d, the intersection depends on d
                    # Let's just enumerate candidates from the ball of the first uncovered point
                    first_pt = uncovered_points[0]
                    candidate_Cs = coverage_sets[first_pt]  # 17 candidates

                    for C in candidate_Cs:
                        if C in selected_set:
                            continue  # Already selected, can't add it
                        # Check if C covers all uncovered points
                        ball_C = coverage_sets[C]
                        covers_all = True
                        for pt in uncovered_points:
                            if temp_coverage[pt] + (1 if pt in ball_C else 0) == 0:
                                # Wait, this logic is wrong. Let me redo.
                                # We need: for each pt in uncovered_points, pt is in ball_C
                                # i.e., coverage_sets[C] contains pt
                                if not np.any(ball_C == pt):
                                    covers_all = False
                                    break
                        if covers_all:
                            # We can remove A and B, add C
                            selected_set.remove(A)
                            selected_set.remove(B)
                            selected_set.add(C)
                            coverage_count[ball_A] -= 1
                            coverage_count[ball_B] -= 1
                            coverage_count[ball_C] += 1
                            improved = True
                            break

        return list(selected_set)

    # Main Loop: Multiple Randomized Greedy Runs
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s limit

    best_code = None
    best_len = float('inf')
    iteration = 0

    # Run at least one standard greedy (seed 0)
    while time.time() < deadline:
        # Use iteration as seed for diversity
        selected = run_greedy(iteration)

        # Prune the result
        selected = prune(selected)

        # Local search swap
        if time.time() < deadline - 10.0:
            selected = local_search_swap(selected, time_limit=3.0)

        if len(selected) < best_len:
            best_len = len(selected)
            best_code = selected
            # Print progress
            # print(f"Iteration {iteration}: size {best_len}")

        # Early exit if we hit the theoretical lower bound (403) or known best (486)
        if best_len <= 486:
            break

        iteration += 1

    # Final check: if best_code is None (shouldn't happen), fallback to first run
    if best_code is None:
        best_code = run_greedy(0)
        best_code = prune(best_code)

    # Convert indices to words
    result = []
    for idx in best_code:
        result.append(idx_to_word(idx))

    return result

# EVOLVE-BLOCK-END
