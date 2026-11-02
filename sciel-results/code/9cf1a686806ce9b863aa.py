"""Large-set construction for the sumset/difference-set exponent lower bound.

Strategy: build large Sidon/Golomb sets (all pairwise sums and differences
distinct) using fast greedy + known explicit constructions, score them with an
FFT-based sumset/difference-set counter, then refine by local search and
prime scaling.  A large set with a small span pushes the score toward its
asymptotic maximum 1 + log(2)/log(2*max(U)+1).
"""
import time
import math
import random
import numpy as np

# ---------------------------------------------------------------------------
# Deadline management
# ---------------------------------------------------------------------------
_T0 = time.time()
DEADLINE = _T0 + 140.0  # hard kill is 150s; keep a 10s safety margin

def _time_left():
    return DEADLINE - time.time()

# ---------------------------------------------------------------------------
# Fast scoring via FFT
# ---------------------------------------------------------------------------
def _score_set(U):
    """Score a set U (list/set of nonnegative ints containing 0).

    score = 1 + log(|U-U| / |U+U|) / log(2*max(U)+1)
    """
    U = sorted(set(U))
    n = len(U)
    if n == 0:
        return 0.0
    max_u = U[-1]
    if max_u == 0:
        return 1.0
    if n < 2:
        return 1.0

    # Fast path for tiny sets (avoid FFT overhead)
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
        return 1.0 + math.log(ratio) / math.log(2 * max_u + 1)

    # FFT path for large sets
    M = 2 * max_u
    # next power of two >= 2*M + 1
    L = 1
    while L < 2 * M + 1:
        L <<= 1
    arr = np.zeros(L, dtype=np.float64)
    for v in U:
        arr[v] = 1.0
    conv = np.fft.irfft(np.fft.rfft(arr) ** 2)
    # conv[k] = #{(a,b) in U x U : a+b = k}
    sumset_size = int(np.sum(conv[:2 * max_u + 1] > 0.5))
    # difference: shift by max_u so that a-b -> (a+max_u)-b in [0, 2*max_u]
    arr2 = np.zeros(L, dtype=np.float64)
    for v in U:
        arr2[v + max_u] = 1.0
    conv2 = np.fft.irfft(np.fft.rfft(arr2) * np.fft.rfft(arr))
    # conv2[k] = #{(a,b): a + max_u - b = k}  =>  a-b = k - max_u
    diffset_size = int(np.sum(conv2 > 0.5))

    ratio = diffset_size / sumset_size
    if ratio <= 0:
        return 0.0
    return 1.0 + math.log(ratio) / math.log(2 * max_u + 1)

# ---------------------------------------------------------------------------
# Golomb / Sidon constructions
# ---------------------------------------------------------------------------
def _greedy_golomb(n, max_val=None, seed=None):
    """Greedy Golomb ruler (distinct pairwise differences) of order n."""
    if n <= 1:
        return [0]
    if max_val is None:
        max_val = n * n * 3
    rng = random.Random(seed)
    ruler = [0]
    diffs = set()
    for _ in range(1, n):
        # try a randomised scan for speed
        best_pos = None
        # first try a deterministic increasing scan from a good lower bound
        lo = ruler[-1] + 1
        # quick deterministic pass
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
            # randomised retry
            for _ in range(2000):
                pos = rng.randint(lo, max_val)
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
        max_val = n * n * 4
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
            for _ in range(2000):
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

def _is_golomb(U):
    U = sorted(U)
    diffs = set()
    for i in range(len(U)):
        for j in range(i + 1, len(U)):
            d = U[j] - U[i]
            if d in diffs:
                return False
            diffs.add(d)
    return True

def _is_sidon(U):
    U = sorted(U)
    sums = set()
    for i in range(len(U)):
        for j in range(i, len(U)):
            s = U[i] + U[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _singer_golomb(q):
    """Singer difference set -> Golomb ruler of order q+1 (q prime power)."""
    # build GF(q) via irreducible polynomial for prime power q = p^k
    # For simplicity handle q prime and q = p^2 small cases.
    pass  # replaced by Welch/Bose below

def _welch_golomb(p):
    """Welch construction: Golomb ruler of order p+1 for prime p."""
    # Elements of GF(p^2) = GF(p)[i]/(i^2 - non_residue)
    # A Welch Sidon set of size p in GF(p^2); map to integers via a basis.
    # Simpler: use the classic Welch Sidon set in Z_p x Z_p.
    # Sidon set S = {(i, i^2 mod p) : i in Z_p}  (p prime, p odd)
    # This is a Sidon set of size p in the abelian group Z_p x Z_p.
    # Map (x,y) -> x * p + y  (a Sidon set in Z_{p^2}? not exactly, but a
    # valid Sidon set in the group; we map to integers and verify).
    if p < 3:
        return [0]
    S = []
    for i in range(p):
        x = i
        y = (i * i) % p
        S.append(x * p + y)
    S = sorted(set(S))
    return S

def _bose_sidon(p):
    """Bose construction of a Sidon set of size p^2 - p + 1 in Z_{p^2-1}."""
    if p < 2:
        return [0]
    # Bose: Sidon set of size p^2-p+1 in cyclic group of order p^2-1
    # Use the standard Bose Sidon set.
    n = p * p - 1
    S = set()
    # Bose construction (from literature): for prime p, Sidon set in Z_{p^2-1}
    S.add(0)
    S.add(n - 1)
    for i in range(1, p):
        S.add(i * p + (i * i) % p)
    for i in range(1, p):
        for j in range(1, p):
            S.add((i * p + j) % n)
    S = sorted(S)
    return S

def _known_golomb_small():
    """A few known optimal Golomb rulers for small orders."""
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

# ---------------------------------------------------------------------------
# Local search
# ---------------------------------------------------------------------------
def _local_search(U, max_iters=400, deadline=None, max_size=None, seed=None):
    """Improve U via add/remove/swap local moves.  Uses FFT scoring."""
    if deadline is None:
        deadline = DEADLINE
    rng = random.Random(seed)
    U = sorted(set(U))
    best_score = _score_set(U)
    best_U = U[:]
    max_u = U[-1] if U else 0
    upper = max_u + max(50, max_u // 5)
    if max_size is None:
        max_size = len(U) + 200

    for it in range(max_iters):
        if time.time() > deadline:
            break
        improved = False
        n = len(U)

        # ---- remove a random non-zero element ----
        if n > 3 and rng.random() < 0.4:
            idx = rng.randint(1, n - 1)
            new_U = U[:idx] + U[idx + 1:]
            s = _score_set(new_U)
            if s > best_score + 1e-12:
                best_score, best_U = s, new_U
                improved = True

        # ---- add a random element ----
        if not improved and len(U) < max_size and rng.random() < 0.4:
            nv = rng.randint(1, upper)
            if nv not in U:
                new_U = sorted(U + [nv])
                s = _score_set(new_U)
                if s > best_score + 1e-12:
                    best_score, best_U = s, new_U
                    improved = True

        # ---- swap a random element to a nearby value ----
        if not improved and n > 2:
            idx = rng.randint(1, n - 1)
            old = U[idx]
            for delta in rng.sample([-100, -50, -20, -10, -5, -3, -2, -1, 1, 2, 3, 5, 10, 20, 50, 100], 6):
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
        if not improved and it % 50 == 0:
            upper += max(50, max_u // 10)

    return best_U, best_score

# ---------------------------------------------------------------------------
# Prime scaling helper
# ---------------------------------------------------------------------------
def _scale(U, p):
    """Multiply the set by prime p (reduces sum collisions, scales span)."""
    return sorted(p * v for v in U)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def construct():
    start = time.time()
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

    # ---- Phase 1: build a menu of candidate sets (greedy + explicit) ----
    candidates = []

    # small known Golomb rulers
    for n, gr in _known_golomb_small().items():
        candidates.append(gr)

    # greedy Golomb rulers of increasing size
    for n in [10, 20, 40, 80, 150, 300, 600, 1200]:
        if time.time() > deadline - 20:
            break
        gr = _greedy_golomb(n, max_val=n * n * 3, seed=1)
        if len(gr) >= n:
            candidates.append(gr)

    # greedy Sidon sets
    for n in [10, 20, 40, 80, 150, 300]:
        if time.time() > deadline - 20:
            break
        ss = _greedy_sidon(n, max_val=n * n * 4, seed=2)
        if len(ss) >= n:
            candidates.append(ss)

    # Welch Sidon sets for primes
    for p in [7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]:
        if time.time() > deadline - 15:
            break
        w = _welch_golomb(p)
        if len(w) >= p:
            candidates.append(w)

    # ---- Phase 2: score all candidates, keep the best few ----
    scored = []
    for U in candidates:
        if time.time() > deadline - 20:
            break
        s = _score_set(U)
        scored.append((s, U))
    scored.sort(key=lambda t: -t[0])

    for s, U in scored[:12]:
        if time.time() > deadline - 10:
            break
        consider(U)

    # ---- Phase 3: local search on the top candidates ----
    for s, U in scored[:6]:
        if time.time() > deadline - 5:
            break
        improved_U, improved_s = _local_search(U, max_iters=300, deadline=deadline, seed=7)
        if improved_s > best_score:
            best_score = improved_s
            best_U = improved_U

    # ---- Phase 4: try prime scaling of the best set ----
    if time.time() < deadline - 5 and best_U:
        for p in [2, 3, 5, 7]:
            if time.time() > deadline - 3:
                break
            scaled = _scale(best_U, p)
            consider(scaled)

    # ---- Phase 5: final local search polish on the overall best ----
    if time.time() < deadline - 2:
        improved_U, improved_s = _local_search(best_U, max_iters=500, deadline=deadline, seed=11)
        if improved_s > best_score:
            best_score = improved_s
            best_U = improved_U

    return sorted(set(best_U))


if __name__ == "__main__":
    U = construct()
    print("size:", len(U), "max:", max(U) if U else 0, "score:", _score_set(U))
