import time
import numpy as np

def construct(q, n, radius):
    """
    Constructs a covering code using MWIS-inspired greedy with targeted
    local search and multi-start restarts.
    """
    start_time = time.time()
    deadline = start_time + 145.0

    total_words = q ** n
    ball_size = 1 + n * (q - 1)  # for radius=1

    def int_to_digits(num, length, base):
        digits = [0] * length
        for i in range(length):
            digits[i] = num % base
            num //= base
        return digits

    def digits_to_int(digits, base):
        num = 0
        for d in reversed(digits):
            num = num * base + d
        return num

    # Precompute balls
    balls_list = [None] * total_words
    for i in range(total_words):
        d = int_to_digits(i, n, q)
        neighbors = [i]
        for pos in range(n):
            orig = d[pos]
            for new_val in range(q):
                if new_val == orig:
                    continue
                d[pos] = new_val
                neighbors.append(digits_to_int(d, q))
            d[pos] = orig
        balls_list[i] = neighbors

    balls = np.array(balls_list, dtype=np.int32)

    # Precompute: for each center, which other centers share at least one neighbor
    # This helps with fast swap validation
    # ball_sets[i] = frozenset of balls[i] for O(1) subset checks
    ball_sets = [frozenset(balls[i].tolist()) for i in range(total_words)]

    def greedy_cover(seed_offset=0):
        """Greedy set cover with optional randomization."""
        uncovered = np.ones(total_words, dtype=bool)
        code = []
        cover_count = np.zeros(total_words, dtype=np.int32)

        # Randomize the order of centers for diversity
        center_order = np.arange(total_words)
        if seed_offset > 0:
            rng = np.random.RandomState(seed_offset)
            rng.shuffle(center_order)

        # Process centers in order, but use gain-based selection
        # Actually, let's do standard greedy but with random tie-breaking
        while np.any(uncovered):
            if time.time() > deadline:
                break

            gains = uncovered[balls].sum(axis=1)
            max_gain = gains.max()
            if max_gain == 0:
                break

            # Get all centers with max gain
            best_centers = np.where(gains == max_gain)[0]
            if len(best_centers) == 1:
                best_center = int(best_centers[0])
            else:
                # Random tie-break for diversity
                best_center = int(best_centers[np.random.randint(len(best_centers))])

            code.append(best_center)
            uncovered[balls[best_center]] = False
            cover_count[balls[best_center]] += 1

        return code, cover_count

    def trim_code(code, cover_count):
        """Remove redundant codewords."""
        changed = True
        while changed and time.time() < deadline:
            changed = False
            to_remove = []
            for i, c in enumerate(code):
                if np.any(cover_count[balls[c]] == 1):
                    continue
                to_remove.append(i)

            if to_remove:
                for idx in sorted(to_remove, reverse=True):
                    c = code[idx]
                    code.pop(idx)
                    cover_count[balls[c]] -= 1
                changed = True
        return code, cover_count

    def targeted_local_search(code, cover_count, max_iter=200000):
        """
        Aggressive targeted local search:
        For each codeword, find candidates that cover its uniquely-covered words.
        Use precomputed ball_sets for fast subset checks.
        """
        code_set = set(code)
        candidates = list(set(range(total_words)) - code_set)

        improved = True
        iter_count = 0
        while improved and time.time() < deadline and iter_count < max_iter:
            improved = False
            iter_count += 1

            # Shuffle code order
            code_indices = list(range(len(code)))
            np.random.shuffle(code_indices)

            for ci in code_indices:
                if time.time() > deadline:
                    break
                i = code[ci]
                # Words only covered by i
                i_ball = balls[i]
                only_mask = cover_count[i_ball] == 1
                only_by_i = i_ball[only_mask]

                if len(only_by_i) == 0:
                    # Can remove i
                    code.pop(ci)
                    cover_count[i_ball] -= 1
                    code_set.discard(i)
                    improved = True
                    break

                # Targeted search: find candidates that cover all of only_by_i
                # First, pre-filter: candidate j must have balls[j] ⊇ only_by_i
                # Use ball_sets for fast check
                found = False
                n_candidates = len(candidates)
                if n_candidates == 0:
                    break

                # Sample candidates, prioritizing those with high overlap
                # Compute a quick score: number of only_by_i words in balls[j]
                # For efficiency, sample a subset
                sample_size = min(200, n_candidates)
                sample_idx = np.random.choice(n_candidates, size=sample_size, replace=False)

                for si in sample_idx:
                    j = candidates[si]
                    if j in code_set:
                        continue
                    # Check if balls[j] covers all of only_by_i
                    # only_by_i is small (typically < 17), so set check is fast
                    j_set = ball_sets[j]
                    # Check subset: all(w in j_set for w in only_by_i)
                    valid = True
                    for w in only_by_i:
                        if w not in j_set:
                            valid = False
                            break
                    if valid:
                        # Valid swap!
                        code[ci] = j
                        code_set.discard(i)
                        code_set.add(j)
                        cover_count[i_ball] -= 1
                        cover_count[balls[j]] += 1
                        improved = True
                        found = True
                        break

                if found:
                    break

        # Final trim
        code, cover_count = trim_code(code, cover_count)
        return code, cover_count

    # Multi-start with different random seeds
    best_code = None
    best_count = float('inf')
    np.random.seed(42)

    n_starts = 3
    for start_i in range(n_starts):
        if time.time() > deadline - 30:  # Leave 30s for final local search
            break

        # Greedy initialization with different randomization
        code, cover_count = greedy_cover(seed_offset=start_i * 1000)

        # Trim
        code, cover_count = trim_code(code, cover_count)

        # Local search
        code, cover_count = targeted_local_search(code, cover_count, max_iter=100000)

        if len(code) < best_count:
            best_count = len(code)
            best_code = code[:]

    # Final aggressive local search on best solution
    if best_code is not None:
        cover_count = np.zeros(total_words, dtype=np.int32)
        for c in best_code:
            cover_count[balls[c]] += 1
        best_code, cover_count = targeted_local_search(best_code, cover_count, max_iter=300000)

    # Convert to list of lists
    if best_code is None:
        # Fallback: should not happen
        best_code = []

    result = []
    for c in best_code:
        result.append(int_to_digits(c, n, q))

    return result
