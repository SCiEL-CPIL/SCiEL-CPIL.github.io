import time
import itertools
import numpy as np
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0

    # Fallback for non-standard or huge cases
    total = q ** n
    if total > 5_000_000:
        if q == 3 and n >= 4 and radius == 1:
            _TETRA = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in _TETRA
                    for tail in itertools.product(range(q), repeat=n - 4)]
        else:
            return [list(w) for w in itertools.product(range(q), repeat=n)]

    # Specialized for q=3, n=8, radius=1
    if q == 3 and n == 8 and radius == 1:
        return _solve_3_8_1(deadline)

    # Generic fallback for smaller cases
    return _generic_greedy(q, n, radius, deadline)

def _solve_3_8_1(deadline):
    q = 3
    n = 8
    radius = 1
    total_points = q ** n  # 6561
    ball_size = 1 + n * (q - 1)  # 17

    # Pre-compute neighbors
    # neighbors[idx] is a list of indices within Hamming distance <= 1
    neighbors = np.empty((total_points, ball_size), dtype=np.int32)
    for idx in range(total_points):
        # Convert idx to base-3 digits
        word = [0] * n
        temp = idx
        for i in range(n):
            word[i] = temp % q
            temp //= q

        neighbors[idx, 0] = idx
        k = 1
        for pos in range(n):
            orig = word[pos]
            # Change digit at pos to (orig+1)%3 and (orig-1)%3
            for delta in [1, 2]:  # +1 and -1 (which is +2 mod 3)
                new_digit = (orig + delta) % q
                # Calculate new index
                new_word = word[:]
                new_word[pos] = new_digit
                new_idx = 0
                multiplier = 1
                for d in new_word:
                    new_idx += d * multiplier
                    multiplier *= q
                neighbors[idx, k] = new_idx
                k += 1

    # Precompute reverse neighbors (for O(1) swap candidate search)
    # reverse_neighbors[idx] = list of codewords that cover idx
    # We'll build this dynamically, but for now, let's use a different approach.
    # Instead, for swap candidates, we look at neighbors of uncovered words.
    # When we remove a word c, uncovered words are those in neighbors[c] with coverage 0.
    # For each uncovered word u, any word in neighbors[u] can cover it.
    # So candidate replacements are union of neighbors[u] for u in uncovered.
    # This is the O(1) per candidate idea from the inspiration.

    def prune(selected):
        """Remove redundant codewords."""
        if not selected:
            return []
        selected = list(selected)
        # Count coverage
        coverage = np.zeros(total_points, dtype=np.int32)
        for idx in selected:
            coverage[neighbors[idx]] += 1

        changed = True
        while changed:
            changed = False
            # Iterate in reverse to handle removals easily
            for i in range(len(selected) - 1, -1, -1):
                idx = selected[i]
                # Check if any point in this ball is covered only by this codeword
                # If min(coverage[neighbors[idx]]) > 1, it is redundant
                if np.min(coverage[neighbors[idx]]) > 1:
                    selected.pop(i)
                    coverage[neighbors[idx]] -= 1
                    changed = True
        return selected

    def greedy_init(seed):
        """Vectorized greedy set cover."""
        rng = np.random.default_rng(seed)
        is_covered = np.zeros(total_points, dtype=bool)
        selected = []

        while not np.all(is_covered):
            if time.time() > deadline - 5:
                break

            uncovered_counts = np.sum(~is_covered[neighbors], axis=1)
            max_count = np.max(uncovered_counts)
            if max_count == 0:
                break

            # Find candidates with max count
            candidates = np.where(uncovered_counts == max_count)[0]
            if len(candidates) > 1:
                # Random tie-breaking
                best_idx = rng.choice(candidates)
            else:
                best_idx = candidates[0]

            selected.append(best_idx)
            is_covered[neighbors[best_idx]] = True

        return selected

    def local_search(initial_code, seed):
        """Kick and Prune local search with O(1) swap candidate search."""
        rng = np.random.default_rng(seed)
        code = prune(list(initial_code))
        best_code = code[:]
        best_len = len(code)

        # Coverage counts for current code
        coverage = np.zeros(total_points, dtype=np.int32)
        for idx in code:
            coverage[neighbors[idx]] += 1

        # Set of current codewords for O(1) lookup
        code_set = set(code)

        temperature = 1.0
        cool_rate = 0.999
        min_temp = 0.01

        while time.time() < deadline - 2:
            improved = False

            # 1. Try to prune
            pruned = prune(code)
            if len(pruned) < len(code):
                code = pruned
                code_set = set(code)
                # Recalculate coverage
                coverage = np.zeros(total_points, dtype=np.int32)
                for idx in code:
                    coverage[neighbors[idx]] += 1
                if len(code) < best_len:
                    best_len = len(code)
                    best_code = code[:]
                improved = True
                continue  # Try to prune again immediately

            # 2. Try a kick (swap)
            if not code:
                break

            # Pick a random codeword to move
            c_idx = rng.integers(len(code))
            c = code[c_idx]

            # Pick a random target.
            # Usually neighbors of c are good candidates.
            # Or a random word from the space.
            # Let's try neighbors of c first (high probability of preserving coverage)
            # and sometimes a random word from the space.
            if rng.random() < 0.8:
                # Pick a random neighbor of c (excluding self)
                n_idx = rng.integers(1, ball_size)
                c_prime = neighbors[c, n_idx]
            else:
                c_prime = rng.integers(total_points)

            if c_prime in code_set:
                continue

            # Tentative swap: remove c, add c_prime
            # Check if removing c breaks coverage
            # We need to check if any point in neighbors[c] has coverage count == 1
            # AND is not covered by c_prime.

            # Points that were uniquely covered by c:
            # unique_to_c = [p for p in neighbors[c] if coverage[p] == 1]
            # If any of these are not in neighbors[c_prime], the swap is invalid.

            # Vectorized check:
            # coverage[neighbors[c]] gives counts for points in c's ball.
            # We need to know if min(coverage[neighbors[c]]) > 1 OR
            # if min(coverage[neighbors[c]]) == 1, then the unique points must be in c_prime's ball.

            # Let's just apply the swap to a copy of coverage and check min.
            # But copying 6561 ints is cheap.

            new_coverage = coverage.copy()
            new_coverage[neighbors[c]] -= 1
            new_coverage[neighbors[c_prime]] += 1

            if np.min(new_coverage) == 0:
                # Invalid swap, revert
                continue
            else:
                # Valid swap
                code[c_idx] = c_prime
                code_set.remove(c)
                code_set.add(c_prime)
                coverage = new_coverage
                improved = True

                # Try to prune after swap
                pruned = prune(code)
                if len(pruned) < len(code):
                    code = pruned
                    code_set = set(code)
                    coverage = np.zeros(total_points, dtype=np.int32)
                    for idx in code:
                        coverage[neighbors[idx]] += 1
                    if len(code) < best_len:
                        best_len = len(code)
                        best_code = code[:]
                else:
                    # Swap didn't allow immediate pruning.
                    # Accept it if we are "hot" enough to explore?
                    # Or just keep it as it might lead to better pruning later.
                    # For now, keep it.
                    pass

            if not improved:
                # No prune, no valid swap found in this iteration?
                # Actually, if swap was valid, improved is True.
                # If swap was invalid, we continue.
                # If we get stuck (no valid swaps for a while), we might need to add a word?
                # Or just keep trying random kicks.
                pass

            temperature *= cool_rate
            if temperature < min_temp:
                temperature = min_temp

        return best_code

    def local_search_v2(initial_code, seed):
        """Improved local search with O(1) swap candidate search via uncovered neighbors."""
        rng = np.random.default_rng(seed)
        code = prune(list(initial_code))
        best_code = code[:]
        best_len = len(code)

        # Coverage counts for current code
        coverage = np.zeros(total_points, dtype=np.int32)
        for idx in code:
            coverage[neighbors[idx]] += 1

        # Set of current codewords for O(1) lookup
        code_set = set(code)

        while time.time() < deadline - 2:
            improved = False

            # 1. Try to prune
            pruned = prune(code)
            if len(pruned) < len(code):
                code = pruned
                code_set = set(code)
                # Recalculate coverage
                coverage = np.zeros(total_points, dtype=np.int32)
                for idx in code:
                    coverage[neighbors[idx]] += 1
                if len(code) < best_len:
                    best_len = len(code)
                    best_code = code[:]
                improved = True
                continue  # Try to prune again immediately

            # 2. Try a kick (swap) with O(1) candidate search
            if not code:
                break

            # Pick a random codeword to move
            c_idx = rng.integers(len(code))
            c = code[c_idx]

            # Tentatively remove c
            new_coverage = coverage.copy()
            new_coverage[neighbors[c]] -= 1

            # Find uncovered words after removing c
            uncovered_words = np.where(new_coverage[neighbors[c]] == 0)[0]

            if len(uncovered_words) == 0:
                # c is redundant, just remove it
                code.pop(c_idx)
                code_set.remove(c)
                coverage = new_coverage
                if len(code) < best_len:
                    best_len = len(code)
                    best_code = code[:]
                improved = True
                continue

            # Collect candidate replacements: neighbors of uncovered words
            # For each uncovered word u, any word in neighbors[u] can cover it
            candidate_set = set()
            for u in uncovered_words:
                candidate_set.update(neighbors[u].tolist())

            # Remove c from candidates (we're replacing it, not keeping it)
            candidate_set.discard(c)
            # Remove existing codewords (we don't want duplicates)
            candidate_set -= code_set

            if not candidate_set:
                continue

            # Try a random candidate
            candidates = list(candidate_set)
            c_prime = candidates[rng.integers(len(candidates))]

            # Check if c_prime covers all uncovered words
            new_coverage[neighbors[c_prime]] += 1
            if np.min(new_coverage) > 0:
                # Valid swap
                code[c_idx] = c_prime
                code_set.remove(c)
                code_set.add(c_prime)
                coverage = new_coverage
                improved = True

                # Try to prune after swap
                pruned = prune(code)
                if len(pruned) < len(code):
                    code = pruned
                    code_set = set(code)
                    coverage = np.zeros(total_points, dtype=np.int32)
                    for idx in code:
                        coverage[neighbors[idx]] += 1
                    if len(code) < best_len:
                        best_len = len(code)
                        best_code = code[:]
            else:
                # Invalid swap, revert
                pass

            if not improved:
                # No prune, no valid swap found in this iteration
                pass

        return best_code

    # Main logic
    best_code = None
    best_len = float('inf')

    # Run multiple greedy instances with different seeds
    # Then local search on the best ones

    # Estimate time for greedy: ~1s each?
    # Let's run 5-10 greedy instances.

    num_greedy = 5
    greedy_codes = []

    for i in range(num_greedy):
        if time.time() > deadline - 20:
            break
        code = greedy_init(i)
        pruned = prune(code)
        greedy_codes.append(pruned)
        if len(pruned) < best_len:
            best_len = len(pruned)
            best_code = pruned[:]

    # Local search on the best greedy results
    # Sort by length
    greedy_codes.sort(key=len)

    # Take top 3
    for code in greedy_codes[:3]:
        if time.time() > deadline - 5:
            break
        ls_code = local_search_v2(code, seed=42)
        if len(ls_code) < best_len:
            best_len = len(ls_code)
            best_code = ls_code[:]

    # Convert indices back to words
    if best_code is None:
        # Fallback
        _TETRA = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in _TETRA
                for tail in itertools.product(range(q), repeat=n - 4)]

    result = []
    for idx in best_code:
        word = [0] * n
        temp = idx
        for i in range(n):
            word[i] = temp % q
            temp //= q
        result.append(word)

    return result

def _generic_greedy(q, n, radius, deadline):
    total = q ** n
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}

    balls = []
    for w in all_words:
        w_idx = word_to_idx[w]
        neighbors = [w_idx]
        for i in range(n):
            for d in range(q):
                if d != w[i]:
                    new_w = list(w)
                    new_w[i] = d
                    neighbors.append(word_to_idx[tuple(new_w)])
        balls.append(neighbors)

    uncovered = np.ones(total, dtype=bool)
    covered_count = total
    chosen = []

    while covered_count > 0 and time.time() < deadline:
        best_idx = -1
        best_gain = 0
        for i in range(total):
            gain = 0
            for idx in balls[i]:
                if uncovered[idx]:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_idx = i
        if best_gain == 0:
            break
        chosen.append(best_idx)
        for idx in balls[best_idx]:
            if uncovered[idx]:
                uncovered[idx] = False
                covered_count -= 1

    return [list(all_words[i]) for i in chosen]
