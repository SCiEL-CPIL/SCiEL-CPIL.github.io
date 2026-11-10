"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random

def _is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0:
        return False
    if n < 9:
        return True
    if n % 3 == 0:
        return False
    r = 5
    while r * r <= n:
        if n % r == 0 or n % (r + 2) == 0:
            return False
        r += 6
    return True

def _prime_exponent_vector(n):
    """Return dict {prime: exponent} for n."""
    vec = {}
    d = 2
    while d * d <= n:
        if n % d == 0:
            c = 0
            while n % d == 0:
                n //= d
                c += 1
            vec[d] = c
        d += 1
    if n > 1:
        vec[n] = vec.get(n, 0) + 1
    return vec

def _max_subset_product(vecs, selected):
    """Upper bound on subset product size: product of largest |selected| |log| values."""
    logs = []
    for v in vecs:
        if v in selected:
            logs.append(math.log(v))
    if not logs:
        return 1
    logs.sort(reverse=True)
    s = 0.0
    for x in logs:
        s += x
    return s

def _rank_of_matrix(rows):
    """Rank of a list of integer vectors over the rationals."""
    if not rows:
        return 0
    ncols = len(rows[0])
    mat = [list(map(float, r)) for r in rows]
    r = 0
    for c in range(ncols):
        # find pivot
        piv = -1
        best = 1e-30
        for i in range(r, len(mat)):
            a = abs(mat[i][c])
            if a > best:
                best = a
                piv = i
        if piv < 0:
            continue
        mat[r], mat[piv] = mat[piv], mat[r]
        pv = mat[r][c]
        for j in range(c, ncols):
            mat[r][j] /= pv
        for i in range(len(mat)):
            if i == r:
                continue
            f = mat[i][c]
            if f == 0.0:
                continue
            for j in range(c, ncols):
                mat[i][j] -= f * mat[r][j]
        r += 1
    return r

def _independent(vecs, selected):
    """True if prime-exponent vectors of selected numbers are linearly independent (over Q)."""
    rows = [vecs[i] for i in selected]
    return _rank_of_matrix(rows) == len(rows)

def _verify(selected, primes, logv):
    """Brute-force check all subset products distinct."""
    prods = {1}
    for i in selected:
        cur = set()
        for p in prods:
            cur.add(p * i)
        if cur & prods:
            return False
        prods |= cur
    return True

def construct():
    deadline = time.time() + 52  # safety margin before 60s
    t0 = time.time()
    random.seed(12345)

    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    pidx = {p: i for i, p in enumerate(primes)}
    candidates = list(range(2, 41))
    n = len(candidates)

    # exponent vectors (dense over primes) and logs
    vecs = []
    for num in candidates:
        d = _prime_exponent_vector(num)
        v = [0] * len(primes)
        for p, e in d.items():
            v[pidx[p]] = e
        vecs.append(v)
    logv = [math.log(num) for num in candidates]

    # weight for maximizing count: prefer "easy" (low log) numbers
    weight = [1000.0 - logv[i] for i in range(n)]

    def upper_bound(selected, start):
        """Count of selected + max additional (by log-size bound)."""
        cap = _max_subset_product(primes, selected)
        cap = cap * 1.0001
        extra = 0
        for j in range(start, n):
            if j not in selected:
                # can we fit? use product of top (len+extra+1) logs
                pass
        # greedy: sort remaining by log desc, accumulate
        rem = [logv[j] for j in range(start, n) if j not in selected]
        rem.sort(reverse=True)
        # current product size
        cur = 1.0
        for i in selected:
            cur *= candidates[i]
        k = len(selected)
        extra = 0
        for x in rem:
            # if adding this (and all larger already added) keeps product <= cap?
            # We estimate: new product if we add top (k+1+extra) logs
            # Simpler: product of all selected plus this candidate
            if cur * math.exp(x) <= cap:
                cur *= math.exp(x)
                extra += 1
            else:
                # adding a smaller one might still fit, but we sorted desc so stop
                break
        return k + extra

    best = []
    best_w = -1.0

    def dfs(selected, start, selset, curw):
        nonlocal best, best_w
        if time.time() > deadline:
            return
        if curw > best_w:
            best = list(selected)
            best_w = curw
        if start >= n:
            return
        # bound
        if upper_bound(selected, start) <= len(best):
            return
        for j in range(start, n):
            if time.time() > deadline:
                return
            newsel = selected + [j]
            if _independent(vecs, newsel):
                dfs(newsel, j + 1, selset | {j}, curw + weight[j])

    # Branch-and-bound (weight-ordered to find large sets early)
    dfs([], 0, frozenset(), 0.0)

    # Local search: try to add one more element to best
    def try_grow(sel):
        sel = list(sel)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for j in range(n):
                if j in sel:
                    continue
                newsel = sel + [j]
                if _independent(vecs, newsel):
                    sel = newsel
                    improved = True
                    break
        return sel

    best = try_grow(best)

    # Randomized restarts: pick a random greedy seed then grow
    for it in range(2000):
        if time.time() > deadline:
            break
        # random ordering greedy
        order = list(range(n))
        random.shuffle(order)
        seed = []
        for j in order:
            newsel = seed + [j]
            if _independent(vecs, newsel):
                seed = newsel
        grown = try_grow(seed)
        if len(grown) > len(best):
            best = grown

    # Final verification of best (brute force)
    if not _verify(best, primes, logv):
        # fallback: recompute from scratch with a safe greedy
        sel = []
        for j in range(n):
            newsel = sel + [j]
            if _independent(vecs, newsel):
                sel = newsel
        best = sel

    out = [candidates[i] for i in best]
    return sorted(out)
# EVOLVE-BLOCK-END
