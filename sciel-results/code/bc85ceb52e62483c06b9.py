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

    # Final phase: targeted local search + simulated annealing with pair swaps
    if best_code is not None:
        cover_count = np.zeros(total_words, dtype=np.int32)
        for c in best_code:
            cover_count[balls[c]] += 1
        best_code, cover_count = targeted_local_search(best_code, cover_count, max_iter=100000)

        # Simulated annealing with pair-swap moves
        if time.time() < deadline - 5:
            code = best_code[:]
            cc = cover_count.copy()
            code_set = set(code)
            candidates = list(set(range(total_words)) - code_set)

            T_start = 5.0
            T_end = 0.01
            n_sa_iters = 0
            max_sa_iters = 500000
            sa_improved = True

            # Build a list of candidate indices for fast random access
            cand_arr = np.array(candidates, dtype=np.int32)
            n_cand = len(cand_arr)

            while sa_improved and n_sa_iters < max_sa_iters and time.time() < deadline - 5:
                sa_improved = False
                T = T_start * (T_end / T_start) ** (n_sa_iters / max(1, max_sa_iters))

                # Try a pair swap: remove code[i], code[j], add cand[a], cand[b]
                # Or single swap: remove code[i], add cand[a]
                # Or single add: add cand[a] (only if it helps cover uncovered)
                # Or single remove: remove code[i] (if all its words still covered)

                move_type = np.random.random()

                if len(code) < 2 or n_cand < 2:
                    break

                if move_type < 0.3:
                    # Single swap: replace code[i] with cand[a]
                    i = np.random.randint(len(code))
                    a = np.random.randint(n_cand)
                    j = int(cand_arr[a])
                    if j in code_set:
                        continue

                    old_i = code[i]
                    # Evaluate: remove old_i, add j
                    # Words that lose coverage: those in balls[old_i] not in balls[j] and not covered by others
                    # Words that gain coverage: those in balls[j] not in balls[old_i] and not covered by others

                    # Quick check: does j cover all words uniquely covered by old_i?
                    i_ball = balls[old_i]
                    only_mask = cc[i_ball] == 1
                    only_by_i = i_ball[only_mask]

                    if len(only_by_i) == 0:
                        # Can just remove old_i
                        code.pop(i)
                        cc[i_ball] -= 1
                        code_set.discard(old_i)
                        sa_improved = True
                        continue

                    # Check if j covers all of only_by_i
                    j_ball = balls[j]
                    j_set = ball_sets[j]
                    valid = True
                    for w in only_by_i:
                        if w not in j_set:
                            valid = False
                            break

                    if valid:
                        # Good swap
                        code[i] = j
                        code_set.discard(old_i)
                        code_set.add(j)
                        cc[i_ball] -= 1
                        cc[j_ball] += 1
                        sa_improved = True
                    else:
                        # Bad swap: count how many only_by_i words j covers
                        covered_by_j = sum(1 for w in only_by_i if w in j_set)
                        deficit = len(only_by_i) - covered_by_j
                        if deficit <= 1 and np.random.random() < np.exp(-deficit / T):
                            # Accept with probability
                            code[i] = j
                            code_set.discard(old_i)
                            code_set.add(j)
                            cc[i_ball] -= 1
                            cc[j_ball] += 1
                            sa_improved = True

                elif move_type < 0.5:
                    # Pair swap: replace code[i], code[j] with cand[a], cand[b]
                    if len(code) < 2:
                        continue
                    i = np.random.randint(len(code))
                    j_idx = np.random.randint(len(code))
                    if j_idx == i:
                        continue
                    a = np.random.randint(n_cand)
                    b = np.random.randint(n_cand)
                    if a == b:
                        continue
                    new_a = int(cand_arr[a])
                    new_b = int(cand_arr[b])
                    if new_a in code_set or new_b in code_set:
                        continue
                    if new_a == new_b:
                        continue

                    old_i = code[i]
                    old_j = code[j_idx]

                    # Check if {new_a, new_b} cover all words uniquely covered by {old_i, old_j}
                    i_ball = balls[old_i]
                    j_ball = balls[old_j]

                    # Words uniquely covered by old_i (not by old_j or others)
                    only_i_mask = (cc[i_ball] == 1)
                    only_i = i_ball[only_i_mask]
                    # But some of these might be covered by old_j
                    # Actually cc counts total coverage, so cc[i_ball]==1 means only old_i covers them
                    # Similarly for old_j

                    only_j_mask = (cc[j_ball] == 1)
                    only_j = j_ball[only_j_mask]

                    # Combine: words that need to be covered by {new_a, new_b}
                    # These are words in only_i ∪ only_j (words covered by exactly one of old_i, old_j and no one else)
                    # Actually: words in i_ball where cc==1 (only old_i) OR words in j_ball where cc==1 (only old_j)
                    # But if a word is in both i_ball and j_ball, cc would be >= 2, so it's not in either only set

                    # Build set of required words
                    required = set(only_i.tolist()) | set(only_j.tolist())

                    if len(required) == 0:
                        # Can remove both
                        # Remove j_idx first (higher index) to keep indices valid
                        if j_idx > i:
                            code.pop(j_idx)
                            cc[balls[old_j]] -= 1
                            code_set.discard(old_j)
                            code.pop(i)
                            cc[balls[old_i]] -= 1
                            code_set.discard(old_i)
                        else:
                            code.pop(i)
                            cc[balls[old_i]] -= 1
                            code_set.discard(old_i)
                            code.pop(j_idx)
                            cc[balls[old_j]] -= 1
                            code_set.discard(old_j)
                        sa_improved = True
                        continue

                    # Check if new_a and new_b together cover all required words
                    na_set = ball_sets[new_a]
                    nb_set = ball_sets[new_b]
                    deficit = 0
                    for w in required:
                        if w not in na_set and w not in nb_set:
                            deficit += 1

                    if deficit == 0:
                        # Good pair swap
                        code[i] = new_a
                        code[j_idx] = new_b
                        code_set.discard(old_i)
                        code_set.discard(old_j)
                        code_set.add(new_a)
                        code_set.add(new_b)
                        cc[balls[old_i]] -= 1
                        cc[balls[old_j]] -= 1
                        cc[balls[new_a]] += 1
                        cc[balls[new_b]] += 1
                        sa_improved = True
                    elif deficit <= 2 and np.random.random() < np.exp(-deficit / T):
                        # Accept with probability
                        code[i] = new_a
                        code[j_idx] = new_b
                        code_set.discard(old_i)
                        code_set.discard(old_j)
                        code_set.add(new_a)
                        code_set.add(new_b)
                        cc[balls[old_i]] -= 1
                        cc[balls[old_j]] -= 1
                        cc[balls[new_a]] += 1
                        cc[balls[new_b]] += 1
                        sa_improved = True

                elif move_type < 0.7:
                    # Try removing a codeword (if all its words are covered by others)
                    i = np.random.randint(len(code))
                    old_i = code[i]
                    i_ball = balls[old_i]
                    if np.any(cc[i_ball] == 1):
                        continue
                    # All words covered by others, remove it
                    code.pop(i)
                    cc[i_ball] -= 1
                    code_set.discard(old_i)
                    sa_improved = True

                else:
                    # Try adding a codeword that covers some uncovered words
                    # This increases size, so only accept if it enables future reductions
                    # For now, skip this move type as it's hard to evaluate
                    pass

                n_sa_iters += 1

            # Final trim after SA
            code, cc = trim_code(code, cc)

            if len(code) < len(best_code):
                best_code = code
                best_code_set = set(best_code)

    # Convert to list of lists
    if best_code is None:
        # Fallback: should not happen
        best_code = []

    result = []
    for c in best_code:
        result.append(int_to_digits(c, n, q))

    return result
