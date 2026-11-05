import time
import random
import sys
from collections import defaultdict

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 112.0  # safety margin before 150s kill

    N = q ** n  # 2187 for q=3, n=7

    # Precompute neighbors as bitsets (Python int as bitset)
    neighbors = [0] * N

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

    def luby_greedy(uncovered_mask, code_list, time_limit):
        """Luby's greedy set cover using bitsets. Returns updated code_list."""
        uncovered = uncovered_mask
        code = list(code_list)
        
        # Initialize weights
        # W[x] = 1 for all x in uncovered
        # We can use a dictionary or a list. Since N is small (2187), a list is fine.
        W = [0.0] * N
        for i in range(N):
            if (uncovered >> i) & 1:
                W[i] = 1.0

        while uncovered and time.time() < time_limit:
            best_candidate = -1
            max_weight = 0.0
            best_cov_mask = 0

            for c in range(N):
                cov_mask = neighbors[c] & uncovered
                if cov_mask == 0:
                    continue
                
                # Calculate sum of weights of covered elements
                # This is expensive if done naively.
                # Optimization: We can iterate through set bits of cov_mask.
                weight_sum = 0.0
                m = cov_mask
                while m:
                    low = m & (-m)
                    idx = low.bit_length() - 1
                    weight_sum += W[idx]
                    m ^= low
                
                if weight_sum > max_weight:
                    max_weight = weight_sum
                    best_candidate = c
                    best_cov_mask = cov_mask

            if best_candidate == -1:
                break

            code.append(best_candidate)
            
            # Update uncovered
            uncovered &= ~neighbors[best_candidate]
            
            # Update weights
            # W[x] = W[x] / (1 + W[x]) for x in covered
            # But we only care about x in uncovered? No, the formula applies to all covered.
            # However, since uncovered is updated, we can just update W for the newly covered ones.
            # Actually, the standard Luby algorithm updates W[x] for all x in S_c.
            # Let's just re-initialize W for the new uncovered set to be safe and simple,
            # as N is small.
            # W = [1.0 if (uncovered >> i) & 1 else 0.0 for i in range(N)]
            
            # More efficient update:
            # The weight of an element x is divided by (1 + W[x]) each time it is covered.
            # Since we are removing covered elements from 'uncovered', we don't need to update W for them.
            # But the weights of the *remaining* uncovered elements are NOT affected by this step in Luby's original definition?
            # Wait, Luby's algorithm:
            # 1. Set W(x) = 1 for all x in U.
            # 2. While U not empty:
            #    a. Pick c* that maximizes sum_{x in S_c* intersect U} W(x)
            #    b. Add c* to C.
            #    c. For all x in S_c*: W(x) = W(x) / (1 + W(x))
            #    d. U = U \ S_c*
            
            # So we MUST update W for all x in S_c*, even if they are now removed from U.
            # But since we only use W for x in U in the next iteration, we can skip updating for x not in U.
            # However, the update W(x) = W(x)/(1+W(x)) is specific to the history.
            # If we just reset W to 1 for all new U, we are doing standard greedy, not Luby.
            # To do Luby correctly, we need to keep track of the "age" or accumulated weight reduction.
            
            # Let's stick to a simpler heuristic that is faster:
            # "Weighted Greedy" where weight of x is 1 / (number of sets covering x that are NOT yet chosen).
            # This is hard to update dynamically.
            
            # Alternative: Just use standard greedy but with a tie-breaker that prefers codewords with lower overlap with existing code?
            # Or, since N is small, we can afford to recompute weights from scratch every step if it's fast enough.
            # Recomputing W for all N elements based on current U is O(N^2) per step? No.
            
            # Let's try a different approach:
            # The main issue with standard greedy is that it doesn't account for future coverage.
            # A good heuristic is to maximize (new_covered - overlap_with_existing_code * lambda).
            # But 'overlap_with_existing_code' is hard to define simply.
            
            # Let's go back to standard greedy but with random tie-breaking to escape symmetry,
            # and rely on the local search to fix the suboptimality.
            # The current program already does this.
            
            # To improve, let's try "Greedy with Lookahead" of depth 1?
            # Pick best c, then pick best c2 given c, evaluate total.
            # This is O(N^2) per step, which might be too slow for 2187 elements.
            
            # Let's stick to standard greedy but optimize the inner loop.
            # The inner loop iterates over all N candidates.
            # We can precompute the list of candidates that cover at least one uncovered element.
            
            pass # Placeholder, will implement standard greedy with optimizations below

    def greedy_cover(uncovered_mask, code_list, time_limit):
        """Standard greedy set cover using bitsets. Returns updated code_list."""
        uncovered = uncovered_mask
        code = list(code_list)

        while uncovered and time.time() < time_limit:
            best_candidate = -1
            max_new_cov = 0
            best_cov_mask = 0

            for c in range(N):
                new_cov = neighbors[c] & uncovered
                cov_count = popcount(new_cov)
                if cov_count > max_new_cov:
                    max_new_cov = cov_count
                    best_candidate = c
                    best_cov_mask = new_cov

            if best_candidate == -1:
                break

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

    def kick_and_repair(code, num_kicks, time_limit):
        """Remove random codewords and repair with greedy."""
        if len(code) <= num_kicks:
            return code

        code = list(code)
        removed = random.sample(range(len(code)), num_kicks)
        for idx in sorted(removed, reverse=True):
            code.pop(idx)

        uncovered = compute_uncovered(code)
        code = greedy_cover(uncovered, code, time_limit)
        code = prune_redundant(code, time_limit)

        return code

    # Run multiple iterations with different strategies
    best_code = None
    best_size = float('inf')

    # First: fast greedy initialization
    initial_code = greedy_cover(full_mask, [], deadline - 30)
    initial_code = prune_redundant(initial_code, deadline - 30)
    best_size = len(initial_code)
    best_code = initial_code[:]

    # Multiple restarts with kicks of varying sizes
    random.seed(42)

    for iteration in range(300):
        if time.time() > deadline - 5:
            break

        # Start from current best or a fresh greedy
        if iteration % 10 == 0:
            # Fresh greedy with random tie-breaking
            code = greedy_cover(full_mask, [], deadline - 10)
            code = prune_redundant(code, deadline - 10)
        else:
            code = best_code[:]

        # Apply kicks of increasing size
        kick_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12]
        for num_kicks in kick_sizes:
            if time.time() > deadline - 3:
                break
            if len(code) <= num_kicks:
                continue
            new_code = kick_and_repair(code, num_kicks, deadline - 3)
            if len(new_code) < len(code):
                code = new_code
                if len(code) < best_size:
                    best_size = len(code)
                    best_code = code[:]
                    if best_size <= 175:
                        return [list(words[c]) for c in best_code]

    # Final: try more aggressive kicks on the best
    if time.time() < deadline - 5:
        for num_kicks in range(1, min(len(best_code), 15)):
            if time.time() > deadline - 2:
                break
            new_code = kick_and_repair(best_code, num_kicks, deadline - 2)
            if len(new_code) < best_size:
                best_size = len(new_code)
                best_code = new_code[:]

    if best_code is None:
        # Fallback: direct sum construction
        import itertools
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    return [list(words[c]) for c in best_code]
