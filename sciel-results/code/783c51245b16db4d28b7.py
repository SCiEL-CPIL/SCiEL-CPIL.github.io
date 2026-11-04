import time
import itertools
import random
import numpy as np


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0

    if q != 3 or n != 8 or radius != 1:
        return _general_construct(q, n, radius, start_time, deadline)

    # Known 486-word code: direct sum of two Tetracodes (radius-1, length 4, q=3)
    # Tetracode: codewords (a, b, a+b, a+2b) for a,b in {0,1,2}
    tetracode = []
    for a in range(3):
        for b in range(3):
            tetracode.append((a, b, (a + b) % 3, (a + 2 * b) % 3))

    code = []
    for t1 in tetracode:
        for t2 in tetracode:
            code.append(tuple(t1[i] + t2[i] for i in range(4)))
    # 81 * 81 = 6561? No: 9 * 9 = 81, 81 * 81 = 6561. Wait, that's the whole space!
    # Direct sum: codeword = (c1, c2) where c1, c2 are tetracode words.
    # That gives 9*9 = 81 codewords of length 8. But that only covers radius 1 in each half,
    # giving radius 2 total. We need radius 1 overall.
    #
    # Correct direct sum for radius 1: use two radius-1 codes of length 4.
    # K_3(4,1) = 9 (the tetracode covers radius 1 for length 4).
    # Direct sum: length 8, radius 1, size = 9 * 9 = 81? But 81 < 403 (lower bound)!
    #
    # The direct sum construction: a word (x1..x4, y1..y4) is covered if
    # (x1..x4) is within dist 1 of some c1 AND (y1..y4) is within dist 1 of some c2.
    # That's radius 2, not radius 1.
    #
    # For radius 1 overall: the direct sum gives a radius-1 code only if each
    # component is a "perfect" covering (covers exactly radius 0 in a structured way).
    #
    # The actual 486 code is different. Let me use the greedy + local search
    # but with much faster coverage counting via numpy.

    total_words = 3 ** 8
    deadline = start_time + 110.0

    # Precompute digits
    digits_of = np.zeros((total_words, 8), dtype=np.int8)
    for i in range(total_words):
        x = i
        for pos in range(7, -1, -1):
            digits_of[i, pos] = x % 3
            x //= 3

    # Precompute neighbor indices for each word (radius 1: self + 8*2 = 17 neighbors)
    # For word i, neighbors: change position pos to value v (v != current)
    neighbor_list = []
    for i in range(total_words):
        d = digits_of[i].tolist()
        nb = [i]
        for pos in range(8):
            orig = d[pos]
            for v in range(3):
                if v != orig:
                    d[pos] = v
                    idx = 0
                    for j in range(8):
                        idx = idx * 3 + d[j]
                    nb.append(idx)
                    d[pos] = orig
        neighbor_list.append(nb)

    # Coverage-based local search
    # Start with a valid covering, try to remove codewords

    def greedy_cover():
        uncovered = np.ones(total_words, dtype=bool)
        code_idx = []
        remaining = np.arange(total_words)
        while uncovered.any():
            best_i = -1
            best_cnt = 0
            # Sample if too slow, but 6561 words is manageable
            for i in remaining:
                cnt = np.sum(uncovered[neighbor_list[i]])
                if cnt > best_cnt:
                    best_cnt = cnt
                    best_i = i
            if best_i < 0 or best_cnt == 0:
                break
            code_idx.append(best_i)
            uncovered[neighbor_list[best_i]] = False
        return code_idx

    def coverage_counts(code_idx):
        cov = np.zeros(total_words, dtype=np.int32)
        for c in code_idx:
            cov[neighbor_list[c]] += 1
        return cov

    def try_remove(code_idx, cov):
        # Try to remove codewords whose removal keeps coverage >= 1
        changed = True
        while changed and time.time() < deadline - 10:
            changed = False
            for idx in range(len(code_idx) - 1, -1, -1):
                c = code_idx[idx]
                if np.all(cov[neighbor_list[c]] >= 2):
                    cov[neighbor_list[c]] -= 1
                    code_idx.pop(idx)
                    changed = True
        return code_idx, cov

    def swap_improve(code_idx, cov):
        # Try removing c and adding c' to reduce or maintain size while improving structure
        # For now, just do removals
        return code_idx, cov

    best_code = None
    best_size = float('inf')

    # Multiple greedy restarts with random tie-breaking
    for restart in range(200):
        if time.time() > deadline - 20:
            break
        code_idx = greedy_cover()
        cov = coverage_counts(code_idx)
        code_idx, cov = try_remove(code_idx, cov)
        if len(code_idx) < best_size:
            best_size = len(code_idx)
            best_code = code_idx[:]

    if best_code is None:
        best_code = greedy_cover()

    # Final cleanup
    cov = coverage_counts(best_code)
    best_code, cov = try_remove(best_code, cov)

    return [digits_of[i].tolist() for i in best_code]


def _general_construct(q, n, radius, start_time, deadline):
    total_words = q ** n

    if total_words > 100000:
        # Too large for exhaustive, use random heuristic
        random.seed(42)
        code = []
        covered = np.zeros(total_words, dtype=bool)
        all_words = np.arange(total_words)
        while not covered.all() and time.time() < deadline - 5:
            cand = random.choice(all_words.tolist())
            # Compute neighbors within radius
            d = []
            x = cand
            for _ in range(n):
                d.append(x % q)
                x //= q
            d.reverse()
            neigh = _neighbors(d, q, n, radius)
            if np.any(covered[neigh]):
                continue
            code.append(cand)
            covered[neigh] = True
        return [list(_to_digits(c, q, n)) for c in code]

    digits_of = np.zeros((total_words, n), dtype=np.int8)
    for i in range(total_words):
        x = i
        for pos in range(n - 1, -1, -1):
            digits_of[i, pos] = x % q
            x //= q

    neighbor_list = []
    for i in range(total_words):
        neighbor_list.append(_neighbors(digits_of[i].tolist(), q, n, radius))

    uncovered = np.ones(total_words, dtype=bool)
    code = []
    while uncovered.any() and time.time() < deadline - 5:
        best_i = -1
        best_cnt = 0
        for i in range(total_words):
            cnt = np.sum(uncovered[neighbor_list[i]])
            if cnt > best_cnt:
                best_cnt = cnt
                best_i = i
        if best_i < 0:
            break
        code.append(best_i)
        uncovered[neighbor_list[best_i]] = False

    return [digits_of[i].tolist() for i in code]


def _to_digits(idx, q, n):
    d = []
    x = idx
    for _ in range(n):
        d.append(x % q)
        x //= q
    d.reverse()
    return d


def _neighbors(d, q, n, radius):
    nb = [0]
    idx = 0
    for pos in range(n):
        idx = idx * q + d[pos]
    nb = [idx]
    for pos in range(n):
        orig = d[pos]
        for v in range(q):
            if v != orig:
                d[pos] = v
                idx2 = 0
                for j in range(n):
                    idx2 = idx2 * q + d[j]
                nb.append(idx2)
                d[pos] = orig
    return nb
