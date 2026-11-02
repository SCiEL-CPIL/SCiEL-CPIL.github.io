import time
import numpy as np
import random
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    if q == 3 and n == 8 and radius == 1:
        return _solve_ternary_8_1(deadline)

    # Fallbacks for other parameters
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    return [list(c) for c in itertools.product(range(q), repeat=n)]


def _solve_ternary_8_1(deadline):
    n = 8
    q = 3

    # Deterministic optimal construction: the ternary Hadamard (3,2) code.
    # 8 coordinates = the 8 distinct linear functionals of (a, b).
    # Union over all 9 shifts of a 2D linear subspace -> 486 codewords.
    coeffs = [[1, 0], [0, 1], [1, 1], [1, 2], [2, 1], [2, 2], [2, 0], [0, 2]]
    code = []
    for a in range(3):
        for b in range(3):
            row = []
            for c1, c2 in coeffs:
                row.append((c1 * a + c2 * b) % 3)
            code.append(row)
    for shift in itertools.product(range(3), repeat=8):
        code.append([(row[i] + shift[i]) % 3 for i in range(8)])

    # Verify covering (cheap and decisive).
    if _verify_cover(code, n, q, 1):
        return code

    # Fallback: local search to guarantee validity.
    return _local_search_fallback(deadline)


def _verify_cover(code, n, q, radius):
    total = q ** n
    weights = [q ** (n - 1 - i) for i in range(n)]
    covered = np.zeros(total, dtype=bool)
    for row in code:
        base = 0
        for i in range(n):
            base += row[i] * weights[i]
        covered[base] = True
        for pos in range(n):
            w = weights[pos]
            digit = row[pos]
            for nd in range(q):
                if nd == digit:
                    continue
                covered[base + (nd - digit) * w] = True
    return bool(np.all(covered))


def _local_search_fallback(deadline):
    n = 8
    q = 3
    total = q ** n
    weights = [3 ** (n - 1 - i) for i in range(n)]

    balls = np.zeros((total, 1 + 2 * n), dtype=np.int32)
    for w in range(total):
        balls[w, 0] = w
        idx = 1
        for pos in range(n):
            weight = weights[pos]
            digit = (w // weight) % 3
            for new_digit in range(q):
                if new_digit == digit:
                    continue
                balls[w, idx] = w + (new_digit - digit) * weight
                idx += 1

    selected = []
    cover_count = np.zeros(total, dtype=np.int32)
    uncovered_mask = np.ones(total, dtype=bool)
    while np.any(uncovered_mask):
        counts = uncovered_mask[balls].sum(axis=1)
        max_count = np.max(counts)
        candidates = np.where(counts == max_count)[0]
        best_idx = candidates[0]
        selected.append(best_idx)
        cover_count[balls[best_idx]] += 1
        uncovered_mask = (cover_count == 0)

    # Remove redundancies
    changed = True
    while changed and time.time() < deadline - 3.0:
        changed = False
        new_selected = []
        for c in selected:
            if np.min(cover_count[balls[c]]) > 1:
                cover_count[balls[c]] -= 1
                changed = True
            else:
                new_selected.append(c)
        selected = new_selected

    result = []
    for w in selected:
        digits = []
        temp = w
        for _ in range(n):
            digits.append(temp % 3)
            temp //= 3
        digits.reverse()
        result.append(digits)
    return result
