"""Optimize finite set U to maximize the sumset/difference-set exponent lower bound.

score = 1 + log(|U-U| / |U+U|) / log(2*max(U)+1)

Strategy:
  * FFT-based fast scoring so large sets are cheap to evaluate.
  * Strong structured seeds: greedy Golomb rulers (maximize |U-U| with small
    span), greedy Sidon sets, and geometric sets.
  * Smooth (simulated-annealing) local-search polish over a continuous
    surrogate: move each element by a small continuous amount, round to the
    nearest integer, keep the best.  This gives a soft, stochastic escape from
    local minima rather than a hard accept-only hill climb.
"""
import time
import math
import random
import numpy as np

_START_TIME = time.time()
_DEADLINE = 145.0  # wall-clock limit; keep a 5s safety margin before 150s


def _elapsed():
    return time.time() - _START_TIME


def _remaining():
    return _DEADLINE - _elapsed()


def _score(U):
    """Score a set U (list of nonnegative ints, containing 0)."""
    U = sorted(set(int(x) for x in U))
    n = len(U)
    if n == 0:
        return 0.0
    if 0 not in U:
        U.append(0)
        U = sorted(U)
    n = len(U)
    max_u = U[-1]
    if max_u == 0:
        return 1.0
    if n < 2:
        return 1.0

    # Fast path for tiny sets (avoid FFT overhead)
    if n <= 70:
        sumset = set()
        diffset = set()
        for a in U:
            for b in U:
                sumset.add(a + b)
                diffset.add(a - b)
        ratio = len(diffset) / len(sumset)
        if ratio <= 0:
            return 0.0
        return 1.0 + math.log(ratio) / math.log(2 * max_u + 1)

    # FFT path for large sets
    M = 2 * max_u
    L = 1
    while L < M + 1:
        L <<= 1
    arr = np.zeros(L, dtype=np.float64)
    for v in U:
        arr[v] = 1.0
    conv = np.fft.irfft(np.fft.rfft(arr) ** 2)
    sumset_size = int(np.sum(conv[:max_u + 1] > 0.5))

    # difference: shift by max_u so a-b -> (a+max_u)-b in [0, 2*max_u]
    arr2 = np.zeros(L, dtype=np.float64)
    for v in U:
        arr2[v + max_u] = 1.0
    conv2 = np.fft.irfft(np.fft.rfft(arr2) * np.fft.rfft(arr))
    diffset_size = int(np.sum(conv2 > 0.5))

    ratio = diffset_size / sumset_size
    if ratio <= 0:
        return 0.0
    return 1.0 + math.log(ratio) / math.log(2 * max_u + 1)


# ---------------------------------------------------------------------------
# Structured seed constructions
# ---------------------------------------------------------------------------
def _greedy_golomb(n, max_val=None, seed=None):
    """Greedy Golomb ruler (distinct pairwise differences) of order n."""
    if n <= 1:
        return [0]
    if max_val is None:
        max_val = n * n * 4
    rng = random.Random(seed)
    ruler = [0]
    diffs = set()
    for _ in range(1, n):
        best_pos = None
        lo = ruler[-1] + 1
        for pos in range(lo, max_val + 1):
            ok = True
            for a in ruler:
                d = pos - a
                if d in diffs:
                    ok = False
                    break
            if ok:
                best_pos = pos
                break
        if best_pos is None:
            for _ in range(3000):
                pos = rng.randint(lo, max_val)
                ok = True
                for a in ruler:
                    if pos - a in diffs:
                        ok = False
                        break
                if ok:
                    best_pos = pos
                    break
        if best_pos is None:
            break
        for a in ruler:
            diffs.add(best_pos - a)
        ruler.append(best_pos)
    return ruler


def _greedy_sidon(n, max_val=None, seed=None):
    """Greedy Sidon (B2) set: all pairwise sums distinct (a<=b)."""
    if n <= 1:
        return [0]
    if max_val is None:
        max_val = n * n * 5
    rng = random.Random(seed)
    seq = [0]
    sums = {0}
    for _ in range(1, n):
        lo = seq[-1] + 1
        best_pos = None
        for pos in range(lo, max_val + 1):
            ok = True
            for a in seq:
                if a + pos in sums:
                    ok = False
                    break
            if ok:
                best_pos = pos
                break
        if best_pos is None:
            for _ in range(3000):
                pos = rng.randint(lo, max_val)
                ok = True
                for a in seq:
                    if a + pos in sums:
                        ok = False
                        break
                if ok:
                    best_pos = pos
                    break
        if best_pos is None:
            break
        for a in seq:
            sums.add(a + best_pos)
        seq.append(best_pos)
    return seq


def _geometric_set(k, base):
    """Geometric progression {0, base, base^2, ...} -- superincreasing."""
    U = [0]
    v = base
    for _ in range(1, k):
        U.append(v)
        v *= base
    return U


def _prime_gap_set(n):
    """Cumulative primes as gaps."""
    def gen_primes(count):
        primes = []
        cand = 2
        while len(primes) < count:
            ok = True
            for p in primes:
                if p * p > cand:
                    break
                if cand % p == 0:
                    ok = False
                    break
            if ok:
                primes.append(cand)
            cand += 1
        return primes

    primes = gen_primes(max(2, n))
    U = [0]
    for p in primes[1:]:
        U.append(U[-1] + p)
    return U


# ---------------------------------------------------------------------------
# Smooth (simulated-annealing) local search over a continuous surrogate
# ---------------------------------------------------------------------------
def _anneal(U, max_iter=4000, seed=None):
    """Stochastic smooth local search.

    Treats elements as continuous variables; each step perturbs one element by
    a Gaussian amount, rounds to the nearest integer, and accepts with the
    Metropolis rule (soft max over the objective) so we can escape local
    minima.  Returns (best_U, best_score).
    """
    rng = random.Random(seed)
    U = sorted(set(int(x) for x in U))
    if 0 not in U:
        U.append(0)
        U = sorted(U)
    cur = list(U)
    cur_score = _score(cur)
    best = list(cur)
    best_score = cur_score

    span = max(1, max(cur))
    # temperature starts around the scale of a single score step
    T0 = 0.05
    Tend = 1e-4
    n_iter = max(1, max_iter)

    for it in range(n_iter):
        if _remaining() < 5.0:
            break
        T = T0 * (Tend / T0) ** (it / n_iter)
        i = rng.randrange(1, len(cur))  # never move 0
        delta = rng.gauss(0.0, max(1.0, span * 0.02))
        new_val = int(round(cur[i] + delta))
        if new_val < 0 or new_val in cur:
            continue
        cand = list(cur)
        cand[i] = new_val
        cand = sorted(set(cand))
        if 0 not in cand:
            cand.append(0)
            cand = sorted(cand)
        s = _score(cand)
        if s >= cur_score or rng.random() < math.exp((s - cur_score) / max(T, 1e-9)):
            cur = cand
            cur_score = s
            if s > best_score:
                best_score = s
                best = list(cur)
    return best, best_score


def _hill_climb(U, max_iter=2000, seed=None):
    """Hard accept-only local search: add / remove / shift one element."""
    rng = random.Random(seed)
    U = sorted(set(int(x) for x in U))
    if 0 not in U:
        U.append(0)
        U = sorted(U)
    best_U = list(U)
    best_score = _score(best_U)
    for _ in range(max_iter):
        if _remaining() < 5.0:
            break
        r = rng.random()
        if r < 0.34:
            max_u = max(U)
            new_val = rng.randint(1, max_u + 2)
            if new_val in U:
                continue
            cand = sorted(U + [new_val])
        elif r < 0.67:
            removable = [x for x in U if x != 0]
            if not removable:
                continue
            rv = rng.choice(removable)
            cand = sorted(x for x in U if x != rv)
        else:
            shiftable = [x for x in U if x != 0]
            if not shiftable:
                continue
            sv = rng.choice(shiftable)
            dv = rng.choice([-3, -2, -1, 1, 2, 3, 5, 8, 13])
            nv = sv + dv
            if nv < 0 or nv in U:
                continue
            cand = sorted((nv if x == sv else x) for x in U)
        s = _score(cand)
        if s > best_score:
            best_score = s
            best_U = cand
            U = cand
    return best_U, best_score


def construct():
    """Main entry point. Returns a list of nonnegative integers containing 0."""
    best_U = [0, 1, 3]
    best_score = _score(best_U)

    def consider(U):
        nonlocal best_U, best_score
        U = sorted(set(int(x) for x in U))
        if 0 not in U:
            U.append(0)
            U = sorted(U)
        s = _score(U)
        if s > best_score:
            best_score = s
            best_U = list(U)
        return U, s

    # Strategy 1: Golomb rulers (strongest for |U-U|/span) of many sizes
    for k in range(3, 60):
        if _remaining() < 8.0:
            break
        U, s = consider(_greedy_golomb(k, seed=k))
        if _remaining() > 15.0:
            Ua, sa = _anneal(U, max_iter=2500, seed=1000 + k)
            consider(Ua)
            Uh, sh = _hill_climb(Ua, max_iter=1500, seed=2000 + k)
            consider(Uh)

    # Strategy 2: Sidon sets
    for k in range(3, 40):
        if _remaining() < 8.0:
            break
        U, s = consider(_greedy_sidon(k, seed=k))
        if _remaining() > 15.0:
            Ua, sa = _anneal(U, max_iter=2000, seed=3000 + k)
            consider(Ua)

    # Strategy 3: geometric / superincreasing sets
    for base in (2, 3, 4, 5):
        for k in range(3, 16):
            if _remaining() < 6.0:
                break
            U, s = consider(_geometric_set(k, base))
            if _remaining() > 12.0:
                Uh, sh = _hill_climb(U, max_iter=1500, seed=4000 + base * 100 + k)
                consider(Uh)

    # Strategy 4: prime-gap sets
    for k in range(3, 30):
        if _remaining() < 6.0:
            break
        U, s = consider(_prime_gap_set(k))
        if _remaining() > 12.0:
            Ua, sa = _anneal(U, max_iter=1500, seed=5000 + k)
            consider(Ua)

    # Strategy 5: random structured sets + polish
    for trial in range(60):
        if _remaining() < 12.0:
            break
        n = random.randint(4, 30)
        max_val = n * n * 3
        U = [0]
        for _ in range(n - 1):
            U.append(random.randint(1, max_val))
        U, s = consider(U)
        if _remaining() > 15.0:
            Ua, sa = _anneal(U, max_iter=2500, seed=6000 + trial)
            consider(Ua)
            Uh, sh = _hill_climb(Ua, max_iter=1500, seed=7000 + trial)
            consider(Uh)

    # Strategy 6: final high-precision polish on the best found so far
    if _remaining() > 20.0:
        Ua, sa = _anneal(best_U, max_iter=8000, seed=9000)
        consider(Ua)
    if _remaining() > 15.0:
        Uh, sh = _hill_climb(best_U, max_iter=6000, seed=9500)
        consider(Uh)
    if _remaining() > 10.0:
        Ua, sa = _anneal(best_U, max_iter=4000, seed=9800)
        consider(Ua)

    if 0 not in best_U:
        best_U.append(0)
        best_U = sorted(best_U)
    return best_U
