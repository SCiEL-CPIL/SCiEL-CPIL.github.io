"""Sumset/difference-set exponent lower bound optimizer.

Keeps the seed-program interface: construct() -> list of nonnegative ints
containing 0.  Score is
    1 + log(|U-U| / |U+U|) / log(2*max(U)+1).
"""
import time
import math
import random
import numpy as np

_T0 = time.time()
DEADLINE = _T0 + 138.0  # hard kill 150s; safety margin


def _time_left():
    return DEADLINE - time.time()


def _score_set(U):
    """Score a set U (nonnegative ints, containing 0)."""
    U = sorted(set(U))
    n = len(U)
    if n == 0:
        return 0.0
    if n < 2 or U[-1] == 0:
        return 1.0
    if n <= 60:
        sumset = set()
        for a in U:
            for b in U:
                sumset.add(a + b)
        diffset = set()
        for a in U:
            for b in U:
                diffset.add(a - b)
        ratio = len(diffset) / len(sumset)
        return 1.0 + math.log(ratio) / math.log(2 * U[-1] + 1)

    M = 2 * U[-1]
    L = 1
    while L < 2 * M + 1:
        L <<= 1
    arr = np.zeros(L, dtype=np.float64)
    for v in U:
        arr[v] = 1.0
    conv = np.fft.irfft(np.fft.rfft(arr) ** 2)
    sumset_size = int(np.sum(conv[:2 * U[-1] + 1] > 0.5))

    arr2 = np.zeros(L, dtype=np.float64)
    for v in U:
        arr2[v + U[-1]] = 1.0
    conv2 = np.fft.irfft(np.fft.rfft(arr2) * np.fft.rfft(arr))
    diffset_size = int(np.sum(conv2 > 0.5))

    ratio = diffset_size / sumset_size
    if ratio <= 0:
        return 0.0
    return 1.0 + math.log(ratio) / math.log(2 * U[-1] + 1)


def _mian_chowla(n, seed=None):
    """Greedy Sidon (Mian-Chowla) sequence starting with 0."""
    if n <= 1:
        return [0]
    if n <= 2:
        return [0, 1]
    seq = [0, 1]
    sums = {0, 1, 2}
    for _ in range(2, n):
        cand = seq[-1] + 1
        while True:
            ok = True
            for a in seq:
                if a + cand in sums:
                    ok = False
                    break
            if ok:
                break
            cand += 1
        for a in seq:
            sums.add(a + cand)
        seq.append(cand)
    return seq


def _greedy_golomb(n, max_val=None, seed=None):
    """Greedy Golomb ruler of order n starting at 0."""
    if n <= 1:
        return [0]
    if max_val is None:
        max_val = n * n * 4
    rng = random.Random(seed)
    ruler = [0]
    diffs = set()
    for _ in range(1, n):
        best = None
        for pos in range(ruler[-1] + 1, max_val + 1):
            ok = True
            for a in ruler:
                if pos - a in diffs:
                    ok = False
                    break
            if ok:
                best = pos
                break
        if best is None:
            break
        for a in ruler:
            diffs.add(best - a)
        ruler.append(best)
    return ruler


def _known_golomb_small():
    return {
        1: [0],
        2: [0, 1],
        3: [0, 1, 3],
        4: [0, 1, 4, 6],
        5: [0, 1, 4, 10, 11],
        6: [0, 1, 4, 10, 12, 17],
        7: [0, 1, 4, 10, 12, 17, 22],
        8: [0, 1, 4, 10, 12, 17, 22, 29],
        9: [0, 1, 4, 10, 12, 17, 22, 29, 36],
        10: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44],
    }


def _welch_golomb(p):
    """Welch Golomb ruler of order p for prime p."""
    if p < 3:
        return [0]
    S = sorted(x * p + (x * x) % p for x in range(p))
    diffs = set()
    ok = True
    for i in range(len(S)):
        for j in range(i + 1, len(S)):
            d = S[j] - S[i]
            if d in diffs:
                ok = False
                break
            diffs.add(d)
        if not ok:
            break
    if not ok:
        return _greedy_golomb(p, max_val=p * p * 3, seed=17)
    return S


def _structured_candidates():
    """Generate several structured families of sets.

    For each scale s, generate variants:
      - pure s * (i*i) for i=0..m-1
      - s * (i*i + floor(i/2)) for i=0..m-1
      - s * (i*i + floor(i/3)) for i=0..m-1
      - s * (i*i + i//2 + (i//3)) for i=0..m-1
      - s * (i*i + (i * i * i) % s) for i=0..m-1 (mod perturbation)
      - s * (i*i + floor(sqrt(i))) for i=0..m-1
    """
    fams = []
    scales = [1, 2, 3, 4, 5, 7, 8, 11, 13, 16, 17, 19, 23, 29, 32, 37, 41, 47, 53, 61]
    for s in scales:
        m = 30
        base = [i * i for i in range(m)]
        v1 = [s * (i * i) for i in range(m)]
        v2 = [s * (i * i + i // 2) for i in range(m)]
        v3 = [s * (i * i + i // 3) for i in range(m)]
        v4 = [s * (i * i + i // 2 + i // 3) for i in range(m)]
        v5 = [s * (i * i + (i * i * i) % s) for i in range(m)]
        v6 = [s * (i * i + int(math.isqrt(i))) for i in range(m)]
        for v in (v1, v2, v3, v4, v5, v6):
            if 0 in v:
                fams.append(sorted(set(v)))
    return fams


def _local_search(U, max_iters=500, deadline=None, seed=None):
    """Local search with add/remove/swap moves."""
    if deadline is None:
        deadline = DEADLINE
    rng = random.Random(seed)
    U = sorted(set(U))
    best_score = _score_set(U)
    best_U = U[:]
    max_u = U[-1] if U else 0
    upper = max_u + max(200, max_u // 2)

    for it in range(max_iters):
        if time.time() > deadline:
            break
        improved = False
        n = len(U)
        span = max_u if max_u > 0 else 1
        delta_range = max(20, span // 15)
        deltas = [rng.randint(-delta_range, delta_range) for _ in range(10)]
        deltas = [d for d in deltas if abs(d) >= 1]

        if n > 3 and rng.random() < 0.35:
            idx = rng.randint(1, n - 1)
            new_U = U[:idx] + U[idx + 1:]
            s = _score_set(new_U)
            if s > best_score + 1e-12:
                best_score, best_U = s, new_U
                improved = True

        if not improved and rng.random() < 0.35:
            nv = rng.randint(1, upper)
            if nv not in U:
                new_U = sorted(U + [nv])
                s = _score_set(new_U)
                if s > best_score + 1e-12:
                    best_score, best_U = s, new_U
                    improved = True

        if not improved and n > 2:
            idx = rng.randint(1, n - 1)
            old = U[idx]
            for delta in deltas:
                nv = old + delta
                if nv < 1 or nv > upper or nv in U:
                    continue
                new_U = sorted(U[:idx] + [nv] + U[idx + 1:])
                s = _score_set(new_U)
                if s > best_score + 1e-12:
                    best_score, best_U = s, new_U
                    improved = True
                    break

        U = best_U[:]
        max_u = U[-1] if U else 0
        if not improved and it % 40 == 0:
            upper += max(100, max_u // 8)

    return best_U, best_score


def _scale(U, p):
    return sorted(p * v for v in U)


def construct():
    deadline = DEADLINE
    best_U = [0, 1, 3]
    best_score = _score_set(best_U)

    def consider(U):
        nonlocal best_U, best_score
        if not U or 0 not in U:
            return
        U = sorted(set(U))
        s = _score_set(U)
        if s > best_score:
            best_score = s
            best_U = U
        return s

    candidates = []

    # Known small Golomb rulers
    for gr in _known_golomb_small().values():
        candidates.append(gr)

    # Mian-Chowla at several sizes
    for n in [15, 25, 40, 60, 90, 130, 180, 250, 350]:
        if time.time() > deadline - 35:
            break
        mc = _mian_chowla(n)
        if len(mc) >= n:
            candidates.append(mc)

    # Greedy Golomb rulers
    for n in [12, 20, 35, 60, 100, 160, 240]:
        if time.time() > deadline - 30:
            break
        gr = _greedy_golomb(n, max_val=n * n * 4, seed=3)
        if len(gr) >= n:
            candidates.append(gr)

    # Welch Golomb rulers for primes
    for p in [7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83]:
        if time.time() > deadline - 25:
            break
        w = _welch_golomb(p)
        if len(w) >= p:
            candidates.append(w)

    # Structured families (quadratic + perturbations)
    if time.time() < deadline - 20:
        for v in _structured_candidates():
            candidates.append(v)

    # Score all candidates, keep top
    scored = []
    for U in candidates:
        if time.time() > deadline - 25:
            break
        s = _score_set(U)
        scored.append((s, U))
    scored.sort(key=lambda t: -t[0])

    for s, U in scored[:15]:
        if time.time() > deadline - 15:
            break
        consider(U)

    # Local search on top candidates
    for s, U in scored[:8]:
        if time.time() > deadline - 10:
            break
        improved_U, improved_s = _local_search(U, max_iters=450, deadline=deadline, seed=13)
        if improved_s > best_score:
            best_score = improved_s
            best_U = improved_U

    # Prime scaling of best
    if time.time() < deadline - 8 and best_U:
        for p in [2, 3, 5, 7]:
            if time.time() > deadline - 4:
                break
            consider(_scale(best_U, p))

    # Final local search polish
    if time.time() < deadline - 3:
        improved_U, improved_s = _local_search(best_U, max_iters=700, deadline=deadline, seed=23)
        if improved_s > best_score:
            best_score = improved_s
            best_U = improved_U

    # Try extending best set
    if time.time() < deadline - 1 and best_U:
        max_u = max(best_U) if best_U else 0
        for offset in range(1, 80):
            if time.time() > deadline - 0.5:
                break
            nv = max_u + offset
            new_U = sorted(best_U + [nv])
            s = _score_set(new_U)
            if s > best_score:
                best_score = s
                best_U = new_U

    return sorted(set(best_U))


if __name__ == "__main__":
    U = construct()
    print("size:", len(U), "max:", max(U) if U else 0, "score:", _score_set(U))
