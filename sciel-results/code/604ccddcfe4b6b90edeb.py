import time
import random
import numpy as np


def construct(q, n, radius):
    """
    Covering code K_q(n, radius) for the football-pool problem.

    Strategy:
      - Full tetracode-product initialization (9-word tetracode -> 81 words),
      - Vectorized greedy set-cover extension,
      - Redundant-word trimming,
      - Local search: removals + 1-swaps (waste-minimizing) + 2-remove-1-add,
        using inverse-index candidate enumeration for fast swap selection.
      - Multi-start with diverse seeds.

    Returns: list of distinct codewords (each a list of n ints in 0..q-1)
    such that every word of {0..q-1}^n is within Hamming distance `radius`.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    total = q ** n
    ball_size = 1 + n * (q - 1)  # radius = 1

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

    # ---------- Precompute balls ----------
    balls = np.empty((total, ball_size), dtype=np.int32)
    for w in range(total):
        digits = int_to_digits(w, n, q)
        ball = [w]
        for pos in range(n):
            orig = digits[pos]
            for nv in range(q):
                if nv == orig:
                    continue
                digits[pos] = nv
                ball.append(digits_to_int(digits, q))
            digits[pos] = orig
        balls[w] = ball

    # ---------- Inverse index: which centers cover each word ----------
    centers_covering = [None] * total
    for i in range(total):
        for w in balls[i]:
            if centers_covering[w] is None:
                centers_covering[w] = []
            centers_covering[w].append(i)
    for w in range(total):
        if centers_covering[w] is not None:
            centers_covering[w] = frozenset(centers_covering[w])

    # ---------- Structured tetracode-product initialization ----------
    def structured_init():
        # Full tetracode: 9 words of length 4 over {0,1,2}
        # The tetracode is a (9,4,2) code with covering radius 1
        tetracode = []
        for a in range(3):
            for b in range(3):
                tetracode.append([a, b, (a + b) % 3, (a + 2 * b) % 3])
        
        product_code = []
        for i in range(9):
            for j in range(9):
                product_code.append(tetracode[i] + tetracode[j])
        
        code_ints = list({digits_to_int(c, q) for c in product_code})
        cover_count = np.zeros(total, dtype=np.int32)
        for c in code_ints:
            cover_count[balls[c]] += 1
        
        # Greedy extension of the structured seed
        uncovered = (cover_count == 0)
        while np.any(uncovered) and time.time() < deadline - 5:
            gains = uncovered[balls].sum(axis=1)
            for c in code_ints:
                gains[c] = -1
            max_gain = int(gains.max())
            if max_gain <= 0:
                break
            best_center = int(np.argmax(gains))
            code_ints.append(best_center)
            cover_count[balls[best_center]] += 1
            uncovered = (cover_count == 0)
        return code_ints, cover_count

    # ---------- Pure greedy set-cover fallback ----------
    def greedy_cover():
        covered = np.zeros(total, dtype=np.bool_)
        covered_count = 0
        code = []
        cover_count = np.zeros(total, dtype=np.int32)
        while covered_count < total and time.time() < deadline - 5:
            gains = ball_size - covered[balls].sum(axis=1)
            best_word = int(np.argmax(gains))
            if gains[best_word] == 0:
                break
            code.append(best_word)
            bp = balls[best_word]
            newly = ~covered[bp]
            if newly.any():
                covered[bp] = True
                covered_count += int(newly.sum())
            cover_count[bp] += 1
        return code, cover_count

    # ---------- Choose the better starting point ----------
    code, cover_count = structured_init()
    if not code:
        code, cover_count = greedy_cover()
    code_set = set(code)

    # ---------- Aggressive trimming of redundant codewords ----------
    def trim():
        nonlocal code, cover_count, code_set
        changed = True
        while changed and time.time() < deadline - 10:
            changed = False
            to_remove = []
            for i, c in enumerate(code):
                if (cover_count[balls[c]] > 1).all():
                    to_remove.append(i)
            if to_remove:
                for idx in sorted(to_remove, reverse=True):
                    c = code[idx]
                    code.pop(idx)
                    cover_count[balls[c]] -= 1
                    code_set.discard(c)
                changed = True

    trim()

    # ---------- Helper: candidate pool via inverse-index intersection ----------
    def candidate_pool_for(words, exclude_set):
        """Return set of centers covering every word in `words`, minus exclude_set."""
        if not words:
            return set()
        pool = set(centers_covering[words[0]])
        for w in words[1:]:
            pool &= centers_covering[w]
            if not pool:
                break
        if exclude_set:
            pool -= exclude_set
        return pool

    # ---------- Helper: best candidate by minimum waste ----------
    def best_candidate_min_waste(pool, bp_old):
        """Pick candidate minimizing waste (coverage of already-covered words)."""
        best_w = None
        best_waste = float('inf')
        in_old = (cover_count[bp_old] > 0)
        for w in pool:
            bp_new = balls[w]
            waste = int(np.sum((cover_count[bp_new] >= 2) & in_old))
            if waste < best_waste:
                best_waste = waste
                best_w = w
        return best_w

    # ---------- Local search: removals + 1-swaps + 2-remove-1-add ----------
    def local_search(time_budget=100.0, seed=12345):
        nonlocal code, cover_count, code_set
        end_time = time.time() + time_budget
        rng = random.Random(seed)
        max_rounds = 500
        for rnd in range(max_rounds):
            if time.time() > end_time:
                break
            improved = False
            code_list = list(code)
            rng.shuffle(code_list)

            # Phase 1: Try removals
            for c in code_list:
                if (cover_count[balls[c]] > 1).all():
                    cover_count[balls[c]] -= 1
                    code.remove(c)
                    code_set.discard(c)
                    improved = True
                    break
            if improved:
                continue

            # Phase 2: Try 1-swaps (remove c, add w) using inverse-index
            for c in code_list:
                if time.time() > end_time:
                    break
                bp_c = balls[c]
                unique_words = bp_c[cover_count[bp_c] == 1]
                if len(unique_words) == 0:
                    # Redundant - should have been caught in phase 1
                    cover_count[bp_c] -= 1
                    code.remove(c)
                    code_set.discard(c)
                    improved = True
                    break

                pool = candidate_pool_for(unique_words.tolist(), code_set)
                if not pool:
                    continue

                best_w = best_candidate_min_waste(pool, bp_c)
                if best_w is not None:
                    cover_count[bp_c] -= 1
                    cover_count[balls[best_w]] += 1
                    code[code.index(c)] = best_w
                    code_set.discard(c)
                    code_set.add(best_w)
                    improved = True
                    break
            if improved:
                continue

            # Phase 3: Try 2-remove 1-add (remove c1, c2, add w)
            if time.time() > end_time - 15:
                break
            n_code = len(code)
            if n_code < 3:
                continue

            for _ in range(min(20, n_code * (n_code - 1) // 2)):
                if time.time() > end_time:
                    break
                i1, i2 = rng.sample(range(n_code), 2)
                c1, c2 = code[i1], code[i2]

                # Compute risk words: words that would become uncovered
                risk_words = []
                bp1 = balls[c1]
                bp2 = balls[c2]
                for w in bp1:
                    if cover_count[w] == 1:
                        risk_words.append(int(w))
                for w in bp2:
                    if cover_count[w] == 1:
                        risk_words.append(int(w))
                    elif cover_count[w] == 2 and (w in bp1):
                        risk_words.append(int(w))

                if not risk_words:
                    # Both c1 and c2 are redundant together
                    cover_count[bp1] -= 1
                    cover_count[bp2] -= 1
                    code.remove(c1)
                    code.remove(c2)
                    code_set.discard(c1)
                    code_set.discard(c2)
                    improved = True
                    break

                pool = candidate_pool_for(risk_words, code_set)
                if not pool:
                    continue

                # For 2-remove-1-add, minimize waste relative to both removed
                bp1 = balls[c1]
                bp2 = balls[c2]
                in_both = (cover_count[bp1] > 0) | (cover_count[bp2] > 0)
                best_w = None
                best_waste = float('inf')
                for w in pool:
                    bp_new = balls[w]
                    waste = int(np.sum((cover_count[bp_new] >= 2) & in_both))
                    if waste < best_waste:
                        best_waste = waste
                        best_w = w

                if best_w is not None:
                    cover_count[bp1] -= 1
                    cover_count[bp2] -= 1
                    cover_count[balls[best_w]] += 1
                    code.remove(c1)
                    code.remove(c2)
                    code.append(best_w)
                    code_set.discard(c1)
                    code_set.discard(c2)
                    code_set.add(best_w)
                    improved = True
                    break

            if not improved:
                # Random perturbation to escape local minima
                if rnd % 10 == 0 and len(code) > 2:
                    if time.time() > end_time:
                        break
                    c = rng.choice(code)
                    bp_c = balls[c]
                    unique_words = bp_c[cover_count[bp_c] == 1]
                    if len(unique_words) > 0:
                        pool = candidate_pool_for(unique_words.tolist(), code_set)
                        if pool:
                            best_w = rng.choice(list(pool))
                            cover_count[bp_c] -= 1
                            cover_count[balls[best_w]] += 1
                            code[code.index(c)] = best_w
                            code_set.discard(c)
                            code_set.add(best_w)
                            improved = True

        # Final trim
        trim()

    # Run local search with a few different random seeds
    seeds = [42, 123, 456, 789, 1011, 2022, 3033, 4044]
    for s in seeds:
        if time.time() > deadline - 5:
            break
        local_search(time_budget=min(25.0, max(5.0, deadline - time.time() - 5)), seed=s)
        trim()

    # Final validation: ensure no duplicates
    seen = set()
    final_code = []
    for c in code:
        if c not in seen:
            seen.add(c)
            final_code.append(c)
    code = final_code

    # Convert back to list of lists
    result = [int_to_digits(c, n, q) for c in code]
    return result
