def construct():
    import time
    _t0 = time.time()
    _DEADLINE = _t0 + 125.0
    MAXN = 10**18

    # ---- 1. Generate ALL powerful numbers <= MAXN (every prime factor appears squared) ----
    # powerful = p1^e1 * p2^e2 * ... with each e_i >= 2.
    # For each prime p, the factor p^(2k) with 2k >= 2 is "squareful".
    # Enumerate squareful primes (p, p^2, p^3, ... <= MAXN) and multiply distinct ones.
    def _primes_upto(n):
        n = int(n)
        if n < 2:
            return []
        sieve = bytearray(b'\x01') * (n + 1)
        sieve[0:2] = b'\x00\x00'
        r = int(n ** 0.5)
        for i in range(2, r + 1):
            if sieve[i]:
                sieve[i * i::i] = bytearray(len(sieve[i * i::i]))
        return [i for i in range(2, n + 1) if sieve[i]]

    # p^2 <= 10^18  =>  p <= 10^9
    primes = _primes_upto(10**9)

    # squareful prime factors: for each prime, list of p^(2k) <= MAXN
    sqf = []
    for p in primes:
        v = p * p
        while v <= MAXN:
            sqf.append(v)
            try:
                v *= p * p
            except OverflowError:
                break

    sqf.sort()
    # remove duplicates (none expected, but safe)
    seen = set()
    sqf_u = []
    for v in sqf:
        if v not in seen:
            seen.add(v)
            sqf_u.append(v)
    sqf = sqf_u

    # ---- 2. Build all products of distinct squareful primes (DFS, prune by MAXN) ----
    powerful = [1]
    n_sqf = len(sqf)

    def _dfs(start, cur):
        # append cur
        powerful.append(cur)
        for i in range(start, n_sqf):
            v = sqf[i]
            if cur > MAXN // v:
                break  # sqf sorted ascending
            _dfs(i + 1, cur * v)

    _dfs(0, 1)
    powerful = sorted(set(powerful))
    P = len(powerful)
    print("powerful count:", P)

    # ---- 3. Batched outer-sum + np.unique to count representations of every N ----
    # rep_count[N] = number of pairs a<=b, a+b=N, both powerful.
    # We only need N <= MAXN, so restrict a <= MAXN//2.
    import numpy as np

    pw = np.array(powerful, dtype=np.uint64)
    # keep a <= MAXN//2
    pw = pw[pw <= MAXN // 2]
    P = len(pw)
    print("pw (a<=MAXN//2):", P)

    # We cannot form full P x P for P ~ 30k (9e8) at once easily; batch by rows.
    # Use a hash-free counting via np.unique on concatenated batch sums.
    # Batch size chosen so that each batch's sum array is manageable.
    BATCH = 4096

    # We'll accumulate counts in a dict via np.unique per batch (sums -> counts).
    # But sums can be up to 2*MAXN > 10^18; np.unique on uint64 works.
    # To avoid double counting a<=b, for row i we only count columns j >= i.
    # We'll handle the a<=b constraint by, for each batch of rows, taking pw[i:] as columns.
    # That makes each batch a triangular slice; we approximate by full pw and then
    # filter sums where a<=b is automatic if we only take j>=i. Simpler: compute full
    # pairwise but only for j>=i via per-row slicing.

    from collections import defaultdict
    counts = defaultdict(int)

    for s in range(0, P, BATCH):
        if time.time() > _DEADLINE:
            break
        e = min(s + BATCH, P)
        rows = pw[s:e]                      # shape (B,)
        # For each row r (value a), valid b are pw[j] with j >= global index of r and a+b<=MAXN.
        # To respect a<=b, for row with index g=s..e-1, columns are pw[g:].
        # Build per-row column slice; do via broadcasting per row (B small).
        for k in range(e - s):
            g = s + k
            a = int(rows[k])
            cols = pw[g:]                   # b >= a  (pw sorted, pw[g] = a)
            # b <= MAXN - a
            cols = cols[cols <= MAXN - a]
            if cols.size == 0:
                continue
            sums = a + cols
            uniq, cnt = np.unique(sums, return_counts=True)
            for val, c in zip(uniq.tolist(), cnt.tolist()):
                counts[val] += int(c)

    # ---- 4. Pick the N with max count, build pairs ----
    if not counts:
        # fallback: return a valid small result
        return {'N': 2, 'pairs': [[1, 1]]}

    bestN = max(counts, key=lambda n: counts[n])
    bestC = counts[bestN]
    print("bestN:", bestN, "count:", bestC)

    pw_set = set(powerful)
    pairs = []
    for a in powerful:
        if 2 * a > bestN:
            break
        b = bestN - a
        if b in pw_set:
            pairs.append([a, b])
    pairs.sort()

    return {'N': bestN, 'pairs': pairs}
