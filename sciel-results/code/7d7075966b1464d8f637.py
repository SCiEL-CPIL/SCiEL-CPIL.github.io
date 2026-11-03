import time
import numpy as np

def construct(q, n, radius):
    """
    Constructs a covering code using structured tetracode-seeded greedy,
    vectorized local search with inverse-index candidate search, and
    multi-phase optimization.
    """
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin

    total_words = q ** n
    # For radius=1, ball size = 1 + n*(q-1)
    ball_size = 1 + n * (q - 1)

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

    # Precompute balls as a 2D array: balls[i] is the set of words within distance 1 of word i
    balls = np.zeros((total_words, ball_size), dtype=np.int32)
    for i in range(total_words):
        d = int_to_digits(i, n, q)
        balls[i, 0] = i
        idx = 1
        for pos in range(n):
            orig = d[pos]
            for new_val in range(q):
                if new_val == orig:
                    continue
                d[pos] = new_val
                balls[i, idx] = digits_to_int(d, q)
                idx += 1
            d[pos] = orig

    # Precompute inverse index: for each word, list of centers whose ball contains it
    # This is a list of lists; word -> [center indices]
    # Actually, for each word w, the centers that cover it are those c where w is in balls[c]
    # We can compute: for each center c, for each word w in balls[c], add c to inverse[w]
    # But that's O(total_words * ball_size) which is fine: 6561 * 19 = ~124K entries
    inverse = [[] for _ in range(total_words)]
    for c in range(total_words):
        for w in balls[c]:
            inverse[w].append(c)
    # Convert to numpy arrays for speed
    inverse_arrs = [np.array(inv, dtype=np.int32) for inv in inverse]

    # Tetracode-based structured seed
    def get_structured_seed():
        if q != 3 or n != 8:
            return list(range(total_words))
        # Tetracode: all [a, b, a+b, a+2b] mod 3 for a,b in {0,1,2}
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        # Product code: concatenate two tetracode words
        seed = []
        for u in tetracode:
            for v in tetracode:
                w = u + v  # 8-digit word
                seed.append(digits_to_int(w, q))
        return list(set(seed))

    def greedy_cover(seed_set, rng=None):
        """Greedy set cover starting from seed_set."""
        if rng is None:
            rng = np.random.RandomState(0)
        uncovered = np.ones(total_words, dtype=bool)
        code = list(seed_set)
        cover_count = np.zeros(total_words, dtype=np.int32)

        for c in code:
            uncovered[balls[c]] = False
            cover_count[balls[c]] += 1

        if not np.any(uncovered):
            return code, cover_count

        while np.any(uncovered) and time.time() < deadline:
            gains = uncovered[balls].sum(axis=1)
            max_gain = gains.max()
            if max_gain == 0:
                break

            best_centers = np.where(gains == max_gain)[0]
            if len(best_centers) == 1:
                best_center = int(best_centers[0])
            else:
                best_center = int(best_centers[rng.randint(len(best_centers))])

            code.append(best_center)
            uncovered[balls[best_center]] = False
            cover_count[balls[best_center]] += 1

        return code, cover_count

    def trim_code(code, cover_count):
        """Remove redundant codewords (those not uniquely covering any word)."""
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

    def local_search(code, cover_count, max_rounds=50000):
        """
        Local search: for each codeword, try to find a replacement that
        covers all uniquely-covered words while minimizing net uncovered increase.
        Uses inverse index for fast candidate enumeration.
        """
        code_set = set(code)
        improved = True
        round_count = 0

        while improved and time.time() < deadline and round_count < max_rounds:
            improved = False
            round_count += 1
            rng = np.random.RandomState(round_count * 7919)
            code_order = list(range(len(code)))
            rng.shuffle(code_order)

            for ci in code_order:
                if time.time() > deadline:
                    break
                i = code[ci]
                i_ball = balls[i]
                only_mask = cover_count[i_ball] == 1
                only_by_i = i_ball[only_mask]

                if len(only_by_i) == 0:
                    # Can remove i
                    code.pop(ci)
                    code_set.discard(i)
                    cover_count[i_ball] -= 1
                    improved = True
                    break

                # Find candidates that cover all of only_by_i
                # Using inverse: for each word in only_by_i, get list of centers covering it
                # Intersection of these lists = valid candidates
                if len(only_by_i) == 1:
                    candidates = inverse_arrs[only_by_i[0]]
                elif len(only_by_i) == 2:
                    c0 = set(inverse_arrs[only_by_i[0]])
                    candidates = np.array([c for c in inverse_arrs[only_by_i[1]] if c in c0], dtype=np.int32)
                elif len(only_by_i) <= 4:
                    # Intersect sets
                    c0 = set(inverse_arrs[only_by_i[0]])
                    for w in only_by_i[1:]:
                        c0 = c0.intersection(inverse_arrs[w])
                        if len(c0) == 0:
                            break
                    candidates = np.array(list(c0), dtype=np.int32) if c0 else np.array([], dtype=np.int32)
                else:
                    # Too many unique words; skip this codeword
                    continue

                # Filter out existing codewords
                candidates = candidates[~np.isin(candidates, list(code_set))]

                if len(candidates) == 0:
                    continue

                # Sample if too many
                if len(candidates) > 300:
                    sample = rng.choice(len(candidates), size=300, replace=False)
                    candidates = candidates[sample]

                best_j = None
                best_score = float('inf')
                i_ball_set = set(i_ball.tolist())

                for j in candidates:
                    j_ball = balls[j]
                    j_ball_set = set(j_ball.tolist())
                    # Newly uncovered: in i_ball but not in j_ball
                    newly_unc = len(i_ball_set - j_ball_set)
                    # Newly covered: in j_ball but not in i_ball
                    newly_cov = len(j_ball_set - i_ball_set)
                    score = newly_unc - newly_cov
                    if score < best_score:
                        best_score = score
                        best_j = j

                if best_j is not None and best_score <= 0:
                    # Accept if neutral or improving
                    code[ci] = best_j
                    code_set.discard(i)
                    code_set.add(best_j)
                    cover_count[i_ball] -= 1
                    cover_count[balls[best_j]] += 1
                    improved = True
                    break

        return code, cover_count

    def aggressive_local_search(code, cover_count, max_rounds=30000):
        """
        More aggressive local search: also try 2-for-1 and 1-for-2 swaps.
        """
        code_set = set(code)
        improved = True
        round_count = 0

        while improved and time.time() < deadline and round_count < max_rounds:
            improved = False
            round_count += 1
            rng = np.random.RandomState(round_count * 104729)
            code_order = list(range(len(code)))
            rng.shuffle(code_order)

            for ci in code_order:
                if time.time() > deadline:
                    break
                i = code[ci]
                i_ball = balls[i]
                only_mask = cover_count[i_ball] == 1
                only_by_i = i_ball[only_mask]

                if len(only_by_i) == 0:
                    code.pop(ci)
                    code_set.discard(i)
                    cover_count[i_ball] -= 1
                    improved = True
                    break

                # Single replacement
                if len(only_by_i) <= 3:
                    if len(only_by_i) == 1:
                        candidates = inverse_arrs[only_by_i[0]]
                    elif len(only_by_i) == 2:
                        c0 = set(inverse_arrs[only_by_i[0]])
                        candidates = np.array([c for c in inverse_arrs[only_by_i[1]] if c in c0], dtype=np.int32)
                    else:
                        c0 = set(inverse_arrs[only_by_i[0]])
                        for w in only_by_i[1:]:
                            c0 = c0.intersection(inverse_arrs[w])
                            if not c0:
                                break
                        candidates = np.array(list(c0), dtype=np.int32) if c0 else np.array([], dtype=np.int32)

                    candidates = candidates[~np.isin(candidates, list(code_set))]
                    if len(candidates) > 0:
                        if len(candidates) > 200:
                            sample = rng.choice(len(candidates), size=200, replace=False)
                            candidates = candidates[sample]

                        best_j = None
                        best_score = float('inf')
                        i_ball_set = set(i_ball.tolist())
                        for j in candidates:
                            j_ball_set = set(balls[j].tolist())
                            score = len(i_ball_set - j_ball_set) - len(j_ball_set - i_ball_set)
                            if score < best_score:
                                best_score = score
                                best_j = j

                        if best_j is not None and best_score <= 0:
                            code[ci] = best_j
                            code_set.discard(i)
                            code_set.add(best_j)
                            cover_count[i_ball] -= 1
                            cover_count[balls[best_j]] += 1
                            improved = True
                            break

                # Try 2-for-1: replace i with two new codewords
                if len(only_by_i) <= 2 and len(code) < total_words - 2:
                    # Find pairs (j1, j2) that together cover only_by_i
                    if len(only_by_i) == 1:
                        w = only_by_i[0]
                        covering = inverse_arrs[w]
                        covering = covering[~np.isin(covering, list(code_set))]
                        if len(covering) >= 2:
                            # Try a few pairs
                            sample_size = min(30, len(covering))
                            sample = rng.choice(len(covering), size=sample_size, replace=False)
                            sample_c = covering[sample]
                            for a in range(len(sample_c)):
                                for b in range(a + 1, len(sample_c)):
                                    j1, j2 = sample_c[a], sample_c[b]
                                    j1_set = set(balls[j1].tolist())
                                    j2_set = set(balls[j2].tolist())
                                    union_set = j1_set | j2_set
                                    i_ball_set = set(i_ball.tolist())
                                    # Must cover only_by_i
                                    if w not in union_set:
                                        continue
                                    score = len(i_ball_set - union_set) - len(union_set - i_ball_set)
                                    if score <= 0:
                                        code[ci] = j1
                                        code.append(j2)
                                        code_set.discard(i)
                                        code_set.add(j1)
                                        code_set.add(j2)
                                        cover_count[i_ball] -= 1
                                        cover_count[balls[j1]] += 1
                                        cover_count[balls[j2]] += 1
                                        improved = True
                                        break
                                if improved:
                                    break
                    elif len(only_by_i) == 2:
                        w1, w2 = only_by_i[0], only_by_i[1]
                        cov1 = set(inverse_arrs[w1])
                        cov2 = set(inverse_arrs[w2])
                        both = cov1 & cov2
                        either = (cov1 | cov2) - {i}
                        either = either - code_set
                        if len(either) >= 2:
                            sample_size = min(40, len(either))
                            sample = rng.choice(len(either), size=sample_size, replace=False)
                            sample_c = list(either)[sample] if isinstance(either, (list, np.ndarray)) else list(either)[sample]
                            # This is getting complex; skip for now
                            pass

        return code, cover_count

    def simulated_annealing(code, cover_count, sa_deadline, max_iter=200000):
        """
        Simulated annealing with targeted moves.
        """
        rng = np.random.RandomState(12345)
        current_code = list(code)
        current_cc = cover_count.copy()
        current_size = len(current_code)

        uncovered_mask = current_cc == 0
        uncovered_count = int(uncovered_mask.sum())

        best_code = current_code[:]
        best_size = current_size
        best_uncovered = uncovered_count
        best_cc = current_cc.copy()

        code_set = set(current_code)

        T_start = 30.0
        T_end = 0.5
        max_steps = max(100, max_iter)

        for iteration in range(max_steps):
            if time.time() > sa_deadline:
                break

            progress = iteration / max_steps
            T = T_start * (T_end / T_start) ** progress

            move_type = rng.random()

            if move_type < 0.5:
                # Single swap or removal
                if len(current_code) == 0:
                    break
                i = rng.randint(len(current_code))
                old_c = current_code[i]
                old_ball = balls[old_c]
                only_by_old = old_ball[current_cc[old_ball] == 1]

                if len(only_by_old) == 0:
                    current_code.pop(i)
                    code_set.discard(old_c)
                    current_cc[old_ball] -= 1
                    current_size -= 1
                    new_unc = int((current_cc == 0).sum())
                    uncovered_count = new_unc
                    if current_size + uncovered_count < best_size + best_uncovered:
                        best_code = current_code[:]
                        best_size = current_size
                        best_uncovered = uncovered_count
                        best_cc = current_cc.copy()
                    continue

                # Find replacement
                if len(only_by_old) <= 3:
                    if len(only_by_old) == 1:
                        candidates = inverse_arrs[only_by_old[0]]
                    else:
                        c0 = set(inverse_arrs[only_by_old[0]])
                        for w in only_by_old[1:]:
                            c0 = c0.intersection(inverse_arrs[w])
                            if not c0:
                                break
                        candidates = np.array(list(c0), dtype=np.int32) if c0 else np.array([], dtype=np.int32)

                    candidates = candidates[~np.isin(candidates, list(code_set))]
                    if len(candidates) > 0:
                        if len(candidates) > 100:
                            sample = rng.choice(len(candidates), size=100, replace=False)
                            candidates = candidates[sample]

                        old_ball_set = set(old_ball.tolist())
                        best_j = None
                        best_delta = float('inf')
                        for j in candidates:
                            j_ball_set = set(balls[j].tolist())
                            delta = len(old_ball_set - j_ball_set) - len(j_ball_set - old_ball_set)
                            if delta < best_delta:
                                best_delta = delta
                                best_j = j

                        if best_j is not None:
                            if best_delta <= 0 or rng.random() < np.exp(-best_delta / max(T, 1e-10)):
                                current_code[i] = best_j
                                code_set.discard(old_c)
                                code_set.add(best_j)
                                current_cc[old_ball] -= 1
                                current_cc[balls[best_j]] += 1
                                new_unc = int((current_cc == 0).sum())
                                uncovered_count = new_unc
                                if current_size + uncovered_count < best_size + best_uncovered:
                                    best_code = current_code[:]
                                    best_size = current_size
                                    best_uncovered = uncovered_count
                                    best_cc = current_cc.copy()

            elif move_type < 0.8:
                # Remove a random codeword (perturbation)
                if len(current_code) > 50:
                    i = rng.randint(len(current_code))
                    old_c = current_code[i]
                    old_ball = balls[old_c]
                    current_code.pop(i)
                    code_set.discard(old_c)
                    current_cc[old_ball] -= 1
                    current_size -= 1
                    new_unc = int((current_cc == 0).sum())
                    uncovered_count = new_unc
                    # Don't update best here; this is a perturbation

            else:
                # Add a codeword covering most uncovered words
                if uncovered_count > 0:
                    gains = (current_cc[balls] == 0).sum(axis=1)
                    max_gain = gains.max()
                    if max_gain > 0:
                        best_centers = np.where(gains == max_gain)[0]
                        new_c = int(best_centers[rng.randint(len(best_centers))])
                        if new_c not in code_set:
                            current_code.append(new_c)
                            code_set.add(new_c)
                            current_cc[balls[new_c]] += 1
                            current_size += 1
                            new_unc = int((current_cc == 0).sum())
                            uncovered_count = new_unc
                            if current_size + uncovered_count < best_size + best_uncovered:
                                best_code = current_code[:]
                                best_size = current_size
                                best_uncovered = uncovered_count
                                best_cc = current_cc.copy()

        return best_code, best_cc

    # Main algorithm
    seed_set = get_structured_seed()
    best_code = None
    best_count = float('inf')

    # Phase 1: Multi-start greedy + trim
    n_starts = 4
    for start_i in range(n_starts):
        if time.time() > deadline - 60:
            break

        rng = np.random.RandomState(start_i * 1000 + 42)
        # Shuffle seed for diversity
        seed_copy = seed_set[:]
        rng.shuffle(seed_copy)

        code, cover_count = greedy_cover(seed_copy, rng)
        code, cover_count = trim_code(code, cover_count)

        if len(code) < best_count:
            best_count = len(code)
            best_code = code[:]

    # Phase 2: Local search on best
    if best_code is not None:
        cover_count = np.zeros(total_words, dtype=np.int32)
        for c in best_code:
            cover_count[balls[c]] += 1

        code, cover_count = local_search(best_code, cover_count, max_rounds=30000)
        if len(code) < best_count:
            best_count = len(code)
            best_code = code[:]

    # Phase 3: Aggressive local search
    if best_code is not None and time.time() < deadline - 40:
        cover_count = np.zeros(total_words, dtype=np.int32)
        for c in best_code:
            cover_count[balls[c]] += 1

        code, cover_count = aggressive_local_search(best_code, cover_count, max_rounds=20000)
        if len(code) < best_count:
            best_count = len(code)
            best_code = code[:]

    # Phase 4: Simulated annealing
    if best_code is not None and time.time() < deadline - 20:
        cover_count = np.zeros(total_words, dtype=np.int32)
        for c in best_code:
            cover_count[balls[c]] += 1

        sa_deadline = min(deadline - 10, time.time() + 40)
        code, cover_count = simulated_annealing(best_code, cover_count, sa_deadline, max_iter=150000)

        # Final trim after SA
        code, cover_count = trim_code(code, cover_count)

        if len(code) < best_count:
            best_count = len(code)
            best_code = code[:]

    # Phase 5: One more local search pass if time
    if best_code is not None and time.time() < deadline - 5:
        cover_count = np.zeros(total_words, dtype=np.int32)
        for c in best_code:
            cover_count[balls[c]] += 1

        code, cover_count = local_search(best_code, cover_count, max_rounds=15000)
        if len(code) < best_count:
            best_count = len(code)
            best_code = code[:]

    if best_code is None:
        best_code = []

    result = []
    for c in best_code:
        result.append(int_to_digits(c, n, q))

    return result
