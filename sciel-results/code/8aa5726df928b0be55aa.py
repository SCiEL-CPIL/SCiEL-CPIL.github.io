import math
import time

MAX_N = 10**18
DEADLINE = time.time() + 130.0  # safety margin under the 150s kill

# primes in order; we use the first ~15 (product of first 15 ~ 6e17)
PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61]
MAXK = len(PRIMES)

# max exponent for prime i such that p_i^e <= MAX_N
MAXE = [int(math.log(MAX_N, p)) for p in PRIMES]


def _score(exps):
    """Number of coprime adjacent divisor pairs for n = prod p_i^exps[i]."""
    k = len(exps)
    divs = [1]
    for i in range(k):
        e = exps[i]
        if e <= 0:
            continue
        p = PRIMES[i]
        base = len(divs)
        pw = 1
        for _ in range(e):
            pw *= p
            for j in range(base):
                divs.append(divs[j] * pw)
    divs.sort()
    cnt = 0
    m = len(divs)
    for i in range(m - 1):
        if math.gcd(divs[i], divs[i + 1]) == 1:
            cnt += 1
    return cnt


def _fits(exps):
    n = 1
    for i, e in enumerate(exps):
        if e > 0:
            n *= PRIMES[i] ** e
            if n > MAX_N:
                return False
    return True


def _polish(exps):
    """Local hill-climbing: adjust individual exponents (within bounds & monotone)."""
    exps = list(exps)
    improved = True
    while improved:
        improved = False
        cur = _score(exps)
        for i in range(len(exps)):
            lo = exps[i + 1] if i + 1 < len(exps) else 0
            hi = exps[i - 1] if i > 0 else MAXE[i]
            e0 = exps[i]
            for ne in range(lo, hi + 1):
                if ne == e0:
                    continue
                cand = exps[:i] + [ne] + exps[i + 1:]
                if not _fits(cand):
                    continue
                s = _score(cand)
                if s > cur:
                    exps = cand
                    cur = s
                    improved = True
                    break
    return exps, cur


def construct():
    best_n = 2 * 3 * 5 * 7 * 11 * 13 * 17 * 19
    best_s = _score([1] * 8)
    global DEADLINE

    # DFS over non-increasing exponent sequences, longest-first (more primes).
    # For each k (number of distinct primes), recurse on exps with e0>=e1>=...
    def dfs(k, max_prev, exps, depth):
        nonlocal best_n, best_s
        if time.time() > DEADLINE:
            return
        if depth > 0:
            # evaluate this candidate
            if _fits(exps):
                s = _score(exps)
                if s > best_s:
                    best_s = s
                    best_n = 1
                    for i in range(k):
                        best_n *= PRIMES[i] ** exps[i]
                    # polish
                    pe, ps = _polish(exps)
                    if ps > best_s:
                        best_s = ps
                        best_n = 1
                        for i in range(k):
                            best_n *= PRIMES[i] ** pe[i]
                    if time.time() > DEADLINE:
                        return
        if k == 0:
            return
        # choose exponent for prime k-1, between max_prev and 1 (we keep all primes used)
        # to keep product small, prefer small exponents; try 1 first then grow
        for e in range(1, max_prev + 1):
            if PRIMES[k - 1] ** e > MAX_N:
                break
            dfs(k - 1, e, exps + [e], depth + 1)
            if time.time() > DEADLINE:
                return

    # Start with k primes, each exponent at least 1, non-increasing.
    # We bound depth by time. Try a range of k values.
    for k in range(2, MAXK + 1):
        if time.time() > DEADLINE:
            break
        # initial: all ones is the minimal non-increasing; grow via DFS
        dfs(k, MAXE[k - 1], [0] * k, 0)

    if best_n < 2:
        best_n = 2 * 3 * 5 * 7 * 11 * 13 * 17 * 19
    return best_n
