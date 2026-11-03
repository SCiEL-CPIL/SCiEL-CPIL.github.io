"""
Covering code K_3(8,1): structured tetracode init + vectorized greedy +
redundancy trimming + inverse-index swap local search.
"""
import time
import numpy as np
import random
import itertools

_TETRA = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
_TETRA_SET = frozenset(tuple(t) for t in _TETRA)
_TAIL_IDS = list(range(3 ** 4))


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 118.0  # safety margin under the 150 s wall-clock kill

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

    # ---------- Trivial exact cases ----------
    if total <= ball_size:
        return [[0] * n]
    if q == 2 and n == 1:
        return [[0], [1]]
    if q == 3 and n == 1:
        return [[0], [1], [2]]

    if q != 3 or n != 8 or radius != 1:
        # Generic fallback for other parameters
        if total <= 200000:
            words = [list(w) for w in itertools.product(range(q), repeat=n)]
            chosen = []
            covered = set()
            for w in words:
                if tuple(w) in covered:
                    continue
                chosen.append(w)
                for i in range(n):
                    for v in range(q):
                        if v == w[i]:
                            covered.add(tuple(w))
                        else:
                            nw = w[:i] + [v] + w[i + 1:]
                            covered.add(tuple(nw))
                if len(covered) >= total:
                    break
            return chosen
        # Direct-sum seed fallback
        base = n - 4
        if base >= 0:
            return [h + list(t) for h in _TETRA
                    for t in itertools.product(range(q), repeat=base)]
        return [list(t) for t in itertools.product(range(q), repeat=n)]

    # ---- Specialized K_3(8,1) ----

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
        tetracode = [
            [0, 0, 0, 0],
            [0, 1, 2, 1],
            [0, 2, 1, 2],
            [1, 0, 1, 2],
            [1, 1, 2, 0],
            [1, 2, 0, 2],
        ]
        product_code = []
        for a in tetracode:
            for b in tetracode:
                product_code.append(a + b)
        code_ints = list({digits_to_int(c, q) for c in product_code})
        cover_count = np.zeros(total, dtype=np.int32)
        for c in code_ints:
            cover_count[balls[c]] += 1
        # Greedy extension
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

    # ---------- Local search: targeted swap via inverse index ----------
    def local_search(time_budget=100.0):
        nonlocal code, cover_count
        end_time = time.time() + time_budget
        improved = True
        rng = random.Random(12345)
        while improved and time.time() < end_time:
            improved = False
            indices = list(range(len(code)))
            rng.shuffle(indices)
            for idx in indices:
                if time.time() > end_time:
                    break
                i = code[idx]
                bp_i = balls[i]
                unique_pts = bp_i[cover_count[bp_i] == 1]
                if len(unique_pts) == 0:
                    cover_count[bp_i] -= 1
                    code.pop(idx)
                    code_set.discard(i)
                    improved = True
                    break
                # Intersection of centers covering every uniquely-covered word
                candidate_set = None
                for w in unique_pts:
                    cc = centers_covering[w]
                    if candidate_set is None:
                        candidate_set = set(cc)
                    else:
                        candidate_set &= cc
                    if not candidate_set:
                        break
                if not candidate_set:
                    continue
                candidates = [c for c in candidate_set if c not in code_set]
                if not candidates:
                    continue
                # Score: minimize wasted coverage after the swap
                best_swap = None
                best_waste = None
                for j in candidates:
                    if time.time() > end_time:
                        return
                    bp_j = balls[j]
                    in_i = (cover_count[bp_i] > 0)
                    waste = int(np.sum((cover_count[bp_j] >= 2) & in_i))
                    if best_waste is None or waste < best_waste:
                        best_waste = waste
                        best_swap = j
                if best_swap is not None:
                    j = best_swap
                    cover_count[balls[i]] -= 1
                    cover_count[balls[j]] += 1
                    code[idx] = j
                    code_set.discard(i)
                    code_set.add(j)
                    improved = True
                    break

    local_search(time_budget=100.0)
    trim()

    # ---------- Convert back to list of lists of integers ----------
    result = [int_to_digits(c, n, q) for c in code]
    return result
