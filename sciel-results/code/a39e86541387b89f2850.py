import time
import random
import sys
from collections import defaultdict

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 112.0  # safety margin before 150s kill

    N = q ** n  # 2187 for q=3, n=7

    # Helper to convert index to list of digits
    def idx_to_word(idx):
        word = []
        for _ in range(n):
            word.append(idx % q)
            idx //= q
        return word

    # Precompute all words and word-to-index mapping
    words = [idx_to_word(i) for i in range(N)]
    word_to_idx = {tuple(w): i for i, w in enumerate(words)}

    # Compute neighbors as bitsets
    neighbors = [0] * N
    for i in range(N):
        w_i = words[i]
        mask = 1 << i
        for pos in range(n):
            for val in range(q):
                if val == w_i[pos]:
                    continue
                new_w = list(w_i)
                new_w[pos] = val
                j = word_to_idx[tuple(new_w)]
                mask |= (1 << j)
        neighbors[i] = mask

    # Precompute bit counts for quick popcount
    def popcount(x):
        return bin(x).count('1')

    # Full universe bitset
    full_mask = (1 << N) - 1

    def greedy_cover(uncovered_mask, code_list, time_limit, use_random=False):
        """Greedy set cover using bitsets. Returns updated code_list."""
        uncovered = uncovered_mask
        code = list(code_list)

        while uncovered and time.time() < time_limit:
            best_candidate = -1
            max_new_cov = 0
            best_cov_mask = 0
            candidates = []

            for c in range(N):
                new_cov = neighbors[c] & uncovered
                cov_count = popcount(new_cov)
                if cov_count > max_new_cov:
                    max_new_cov = cov_count
                    best_candidate = c
                    best_cov_mask = new_cov
                    candidates = [c]
                elif cov_count == max_new_cov and cov_count > 0:
                    candidates.append(c)

            if best_candidate == -1:
                break

            if use_random and len(candidates) > 1:
                best_candidate = random.choice(candidates)
                best_cov_mask = neighbors[best_candidate] & uncovered

            code.append(best_candidate)
            uncovered &= ~neighbors[best_candidate]

        return code

    def prune_redundant(code, time_limit):
        """Remove redundant codewords using cover counts."""
        if not code:
            return code

        # Count how many times each word is covered
        cover_count = [0] * N
        for c in code:
            mask = neighbors[c]
            m = mask
            while m:
                low = m & (-m)
                idx = low.bit_length() - 1
                cover_count[idx] += 1
                m ^= low

        changed = True
        while changed and time.time() < time_limit:
            changed = False
            to_remove = []
            for idx, c in enumerate(code):
                redundant = True
                mask = neighbors[c]
                m = mask
                while m:
                    low = m & (-m)
                    u = low.bit_length() - 1
                    if cover_count[u] <= 1:
                        redundant = False
                        break
                    m ^= low
                if redundant:
                    to_remove.append(idx)

            if to_remove:
                for idx in to_remove:
                    c = code[idx]
                    mask = neighbors[c]
                    m = mask
                    while m:
                        low = m & (-m)
                        u = low.bit_length() - 1
                        cover_count[u] -= 1
                        m ^= low
                remove_set = set(to_remove)
                code = [c for i, c in enumerate(code) if i not in remove_set]
                changed = True

        return code

    def compute_uncovered(code):
        """Compute the set of uncovered words given a code."""
        covered = 0
        for c in code:
            covered |= neighbors[c]
        return full_mask ^ covered

    def kick_and_repair(code, num_kicks, time_limit, use_random_greedy=False):
        """Remove random codewords and repair with greedy."""
        if len(code) <= num_kicks:
            return code

        code = list(code)
        removed = random.sample(range(len(code)), num_kicks)
        for idx in sorted(removed, reverse=True):
            code.pop(idx)

        uncovered = compute_uncovered(code)
        code = greedy_cover(uncovered, code, time_limit, use_random=use_random_greedy)
        code = prune_redundant(code, time_limit)

        return code

    def swap_improve(code, time_limit):
        """Try swapping one codeword for another to reduce total size."""
        if len(code) < 2:
            return code

        code = list(code)
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            for i in range(len(code)):
                for j in range(N):
                    if j in code:
                        continue
                    # Try replacing code[i] with j
                    new_code = code[:]
                    new_code[i] = j
                    # Check if new_code is still a valid cover and smaller
                    # Actually, we want to see if we can remove more after swap
                    # For now, just check if it's a valid cover
                    covered = 0
                    for c in new_code:
                        covered |= neighbors[c]
                    if covered == full_mask:
                        # It's valid, now try to prune
                        pruned = prune_redundant(new_code, time_limit)
                        if len(pruned) < len(code):
                            code = pruned
                            improved = True
                            break
                if improved:
                    break
        return code

    # Run multiple iterations with different strategies
    best_code = None
    best_size = float('inf')

    # First: fast greedy initialization
    initial_code = greedy_cover(full_mask, [], deadline - 30, use_random=False)
    initial_code = prune_redundant(initial_code, deadline - 30)
    best_size = len(initial_code)
    best_code = initial_code[:]

    # Multiple restarts with kicks of varying sizes
    random.seed(42)

    for iteration in range(300):
        if time.time() > deadline - 5:
            break

        # Start from current best or a fresh greedy
        if iteration % 7 == 0:
            # Fresh greedy with random tie-breaking
            code = greedy_cover(full_mask, [], deadline - 10, use_random=True)
            code = prune_redundant(code, deadline - 10)
        else:
            code = best_code[:]

        # Apply kicks of increasing size
        kick_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        for num_kicks in kick_sizes:
            if time.time() > deadline - 3:
                break
            if len(code) <= num_kicks:
                continue
            new_code = kick_and_repair(code, num_kicks, deadline - 3, use_random_greedy=(iteration % 3 == 0))
            if len(new_code) < len(code):
                code = new_code
                if len(code) < best_size:
                    best_size = len(code)
                    best_code = code[:]
                    if best_size <= 175:
                        return [list(words[c]) for c in best_code]
            # Occasionally try swap improvement
            if iteration % 5 == 0 and time.time() < deadline - 5:
                swapped = swap_improve(code, deadline - 5)
                if len(swapped) < len(code):
                    code = swapped
                    if len(code) < best_size:
                        best_size = len(code)
                        best_code = code[:]

    # Final: try more aggressive kicks on the best
    if time.time() < deadline - 5:
        for num_kicks in range(1, min(len(best_code), 15)):
            if time.time() > deadline - 2:
                break
            new_code = kick_and_repair(best_code, num_kicks, deadline - 2, use_random_greedy=True)
            if len(new_code) < best_size:
                best_size = len(new_code)
                best_code = new_code[:]

    if best_code is None:
        # Fallback: direct sum construction
        import itertools
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    return [list(words[c]) for c in best_code]
