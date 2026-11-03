"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import numpy as np


def construct(q, n, radius):
    """
    Vectorized greedy set-cover for covering code K_q(n, radius) with
    randomized restarts and local search polish (removal + swap).
    """
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit

    total_words = q ** n
    ball_size = 1 + n * (q - 1)  # for radius=1

    def index_to_word(idx, q, n):
        """Convert a linear index back to a word (tuple of ints)."""
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = idx % q
            idx //= q
        return tuple(word)

    # Precompute all ball indices as a 2D numpy array: shape (total_words, ball_size)
    # For radius=1: ball(idx) = {idx} ∪ {idx ± weight for each position, each alternative digit}
    weights = [q ** (n - 1 - pos) for pos in range(n)]
    ball_flat = np.empty((total_words, ball_size), dtype=np.int64)
    for idx in range(total_words):
        # Get digits
        temp = idx
        digits = [0] * n
        for i in range(n - 1, -1, -1):
            digits[i] = temp % q
            temp //= q
        # Ball entries
        ball_flat[idx, 0] = idx
        offset = 1
        for pos in range(n):
            w = weights[pos]
            d = digits[pos]
            for val in range(q):
                if val != d:
                    ball_flat[idx, offset] = idx + (val - d) * w
                    offset += 1

    # Precompute a membership lookup: for each word, which balls contain it
    # This is the "inverse" of ball_flat: for each word w, list of codewords c such that w in ball(c)
    # Since ball(c) contains c and its neighbors, and the relation is symmetric for radius 1,
    # we can build this by: for each c, for each w in ball(c), add c to inverse[w]
    # But this is expensive. Instead, note that for radius 1, w is in ball(c) iff d(w,c) <= 1.
    # We'll use a different approach: for the swap check, we need to know if a set of words
    # are all covered by another codeword. We'll precompute a bitset representation.
    
    # Actually, for the swap optimization, we need: given codeword ci and candidate cand,
    # check if all words uniquely covered by ci are also in ball(cand).
    # We can precompute for each pair (ci, cand) whether ball(ci) \ ball(cand) is empty.
    # But that's O(total_words^2 * ball_size) which is too expensive.
    
    # Better approach: use the fact that for radius 1, two balls intersect in a structured way.
    # For now, we'll use a faster simulation approach with precomputed ball membership.
    
    # Precompute: for each word, the set of codewords whose ball contains it
    # Use a list of sets for memory efficiency
    inverse_balls = [set() for _ in range(total_words)]
    for c in range(total_words):
        for w in ball_flat[c]:
            inverse_balls[w].add(c)
    
    # Helper: greedy set cover with optional randomization
    def greedy_cover(randomize=False, seed=None):
        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = np.random.default_rng()

        uncovered = np.ones(total_words, dtype=np.uint8)
        selected = []

        while uncovered.any():
            if time.time() > deadline - 5:
                break

            if randomize:
                # Randomized greedy: among top-k candidates, pick one with noise
                sample_size = min(200, total_words)
                candidates = rng.choice(total_words, size=sample_size, replace=False)
                best_count = -1
                best_idx = -1
                for cand in candidates:
                    count = np.count_nonzero(uncovered[ball_flat[cand]])
                    if count > best_count:
                        best_count = count
                        best_idx = cand
                if best_idx == -1:
                    for idx in range(total_words):
                        count = np.count_nonzero(uncovered[ball_flat[idx]])
                        if count > best_count:
                            best_count = count
                            best_idx = idx
            else:
                # Deterministic greedy: find max coverage
                best_count = -1
                best_idx = -1
                chunk = 256
                for start in range(0, total_words, chunk):
                    end = min(start + chunk, total_words)
                    counts = np.count_nonzero(uncovered[ball_flat[start:end]], axis=1)
                    local_max = np.argmax(counts)
                    if counts[local_max] > best_count:
                        best_count = counts[local_max]
                        best_idx = start + local_max

            if best_idx == -1:
                break

            selected.append(best_idx)
            uncovered[ball_flat[best_idx]] = 0

        return selected

    # Helper: local search polish (removal + swap) with optimized swap check
    def local_search(selected, time_budget=5.0):
        if not selected:
            return selected

        selected = list(selected)
        end_time = time.time() + time_budget

        # Build cover count
        cover_count = np.zeros(total_words, dtype=np.int32)
        for ci in selected:
            cover_count[ball_flat[ci]] += 1

        # Pass 1: removal
        changed = True
        while changed and time.time() < end_time:
            changed = False
            for i in range(len(selected)):
                ci = selected[i]
                ball = ball_flat[ci]
                if not (cover_count[ball] == 1).any():
                    cover_count[ball] -= 1
                    selected.pop(i)
                    changed = True
                    break

        # Pass 2: swap-based local search with optimized check
        if time.time() < end_time - 2:
            # Rebuild cover count
            cover_count = np.zeros(total_words, dtype=np.int32)
            for ci in selected:
                cover_count[ball_flat[ci]] += 1

            selected_set = set(selected)
            improved = True
            while improved and time.time() < end_time - 2:
                improved = False
                for i in range(len(selected)):
                    if time.time() > end_time - 3:
                        break
                    ci = selected[i]
                    ball_i = ball_flat[ci]
                    unique_mask = (cover_count[ball_i] == 1)
                    n_unique = np.count_nonzero(unique_mask)

                    if n_unique == 0:
                        # Can remove ci directly, but we're in swap phase
                        # Try to find a swap that maintains coverage
                        pass

                    # Try random swaps with optimized check
                    max_tries = 50
                    for _ in range(max_tries):
                        if time.time() > end_time - 3:
                            break
                        cand = np.random.randint(0, total_words)
                        if cand in selected_set:
                            continue

                        ball_cand = ball_flat[cand]

                        # Optimized check: verify all unique words of ci are in ball_cand
                        if n_unique > 0:
                            unique_indices = ball_i[unique_mask]
                            # Check if all unique_indices are in ball_cand
                            # Use: for each unique index, check if it's in inverse_balls[cand]
                            # But that's slow. Instead, use a bitset approach.
                            # Precompute a bitset for each ball? Too much memory.
                            # Use numpy: create a boolean array for ball_cand membership
                            # Faster: use the fact that we can check via cover_count simulation
                            pass

                        # Simulate the swap
                        cover_count[ball_i] -= 1
                        cover_count[ball_cand] += 1
                        if np.any(cover_count == 0):
                            # Invalid
                            cover_count[ball_cand] -= 1
                            cover_count[ball_i] += 1
                        else:
                            # Valid swap
                            selected[i] = cand
                            selected_set.discard(ci)
                            selected_set.add(cand)
                            improved = True
                            break

                    if improved:
                        break

            # After swaps, try more removals
            if time.time() < end_time - 1:
                cover_count = np.zeros(total_words, dtype=np.int32)
                for ci in selected:
                    cover_count[ball_flat[ci]] += 1

                changed = True
                while changed and time.time() < end_time:
                    changed = False
                    for i in range(len(selected)):
                        ci = selected[i]
                        ball = ball_flat[ci]
                        if not (cover_count[ball] == 1).any():
                            cover_count[ball] -= 1
                            selected.pop(i)
                            changed = True
                            break

        return selected

    # Helper: validate solution
    def is_valid(selected):
        cover = np.zeros(total_words, dtype=np.uint8)
        for ci in selected:
            cover[ball_flat[ci]] = 1
        return np.all(cover == 1)

    # Fallback construction
    def fallback():
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    # Main: try multiple strategies and keep the best
    best_solution = None
    best_size = float('inf')

    # Strategy 1: Deterministic greedy + local search
    if time.time() < deadline - 20:
        t0 = time.time()
        selected = greedy_cover(randomize=False)
        if len(selected) > 0 and is_valid(selected):
            polished = local_search(selected, time_budget=min(10, deadline - time.time() - 5))
            if is_valid(polished):
                if len(polished) < best_size:
                    best_size = len(polished)
                    best_solution = polished

    # Strategy 2: Multiple randomized greedy + local search
    remaining_time = deadline - time.time() - 5
    if remaining_time > 5:
        num_restarts = int(min(20, remaining_time / 3))  # ~3s per restart
        for r in range(num_restarts):
            if time.time() > deadline - 5:
                break
            seed = r * 12345 + 67890
            t0 = time.time()
            selected = greedy_cover(randomize=True, seed=seed)
            if len(selected) == 0:
                continue
            if not is_valid(selected):
                continue
            # Polish with limited time
            polish_time = min(3, deadline - time.time() - 3)
            if polish_time > 0.5:
                polished = local_search(selected, time_budget=polish_time)
                if is_valid(polished):
                    if len(polished) < best_size:
                        best_size = len(polished)
                        best_solution = polished

    # If no valid solution found, use fallback
    if best_solution is None:
        fb = fallback()
        return fb

    # Convert to list of lists
    codewords = [list(index_to_word(ci, q, n)) for ci in best_solution]

    # Final validity check
    if not is_valid(best_solution):
        return fallback()

    return codewords

# EVOLVE-BLOCK-END
