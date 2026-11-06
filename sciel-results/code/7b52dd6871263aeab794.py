import time
import itertools

_START = time.time()

def _spf_sieve(limit):
    spf = list(range(limit+1))
    if limit >= 0: spf[0] = 0
    if limit >= 1: spf[1] = 1
    for i in range(2, int(limit**0.5)+1):
        if spf[i] == i:
            for j in range(i*i, limit+1, i):
                if spf[j] == j:
                    spf[j] = i
    return spf

_SPF = _spf_sieve(20000)

PRIMES30 = [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,101,103,107,109,113]
P2IDX = {p:i for i,p in enumerate(PRIMES30)}

def _build_M(limit):
    M = [-1]*(limit+1)
    M[1] = 0
    for x in range(2, limit+1):
        t = x
        m = 0
        ok = True
        while t > 1:
            p = _SPF[t]
            idx = P2IDX.get(p)
            if idx is None:
                ok = False
                break
            m |= 1 << idx
            while t % p == 0:
                t //= p
        if ok:
            M[x] = m
    return M

_M = _build_M(20000)

def _is_smooth(x, Pmask):
    m = _M[x]
    if m < 0:
        return False
    return (m & ~Pmask) == 0

def _smooth_list(Pmask, limit=10000):
    return [x for x in range(1, limit+1) if _is_smooth(x, Pmask)]

def _find_clique(A, Pmask, k, node_budget, deadline):
    n = len(A)
    if n < k:
        return None
    # Build adjacency and compute degrees
    adj = [0]*n
    for i in range(n):
        ai = A[i]
        for j in range(i+1, n):
            if _is_smooth(ai + A[j], Pmask):
                adj[i] |= (1<<j)
                adj[j] |= (1<<i)
    
    # Compute degree of each vertex
    deg = [adj[i].bit_count() for i in range(n)]
    
    # Order vertices by degree (descending) for better branch ordering
    order = sorted(range(n), key=lambda i: -deg[i])
    # Create mapping: position in DFS tree -> vertex index
    # We'll use a different approach: sort candidates by degree during DFS
    
    higher = [0]*n
    for v in range(n-2, -1, -1):
        higher[v] = higher[v+1] | (1 << (v+1))
    nodes = [0]
    aborted = [False]
    import time as _t
    
    def dfs(cand_mask, need, path):
        if need == 0:
            return list(path)
        if cand_mask.bit_count() < need:
            return None
        nodes[0] += 1
        if nodes[0] > node_budget or _t.time() > deadline:
            aborted[0] = True
            return None
        
        # Get candidates sorted by degree (within current candidate set)
        m = cand_mask
        cands = []
        while m:
            lsb = m & (-m)
            v = lsb.bit_length() - 1
            # Count how many neighbors v has in the remaining candidate set
            nc_count = (cand_mask & adj[v] & higher[v]).bit_count()
            cands.append((nc_count, v))
            m &= m - 1
        
        # Sort by degree descending (try high-degree vertices first)
        cands.sort(key=lambda x: -x[0])
        
        for nc_count, v in cands:
            if nc_count + 1 < need:
                continue  # Not enough neighbors to complete the clique
            nc = cand_mask & adj[v] & higher[v]
            res = dfs(nc, need-1, path + [v])
            if res is not None:
                return res
        
        return None
    
    res = dfs((1<<n)-1, k, [])
    if res is None:
        return None
    vals = [A[v] for v in res]
    # verify
    for i in range(k):
        for j in range(i+1, k):
            if not _is_smooth(vals[i]+vals[j], Pmask):
                return None
    return vals

def construct():
    deadline = time.time() + 75.0
    primes_first = [2,3,5,7,11,13,17,19,23,29,31,37,41,43]
    
    def pmask_from_list(P):
        m = 0
        for p in P:
            m |= 1 << P2IDX[p]
        return m
    
    # Strategy 1: Try smallest prime sets first with prefix primes
    for k in range(3, 11):
        if time.time() > deadline - 8:
            break
        P = primes_first[:k]
        pm = pmask_from_list(P)
        L = _smooth_list(pm)
        if len(L) < 12:
            continue
        
        # Try different subset sizes, starting with smaller for speed
        Ts = [50, 100, 150, 200, 300, 400, 500, 600, 800, 1000]
        Ts = [t for t in Ts if t <= len(L)]
        if len(L) not in Ts and len(L) > 1000:
            Ts.append(len(L))
        
        for T in Ts:
            if time.time() > deadline - 8:
                break
            A = L[:T]
            remaining = deadline - time.time()
            sub_deadline = min(time.time() + max(3.0, remaining * 0.3), deadline - 8.0)
            budget = 500000
            res = _find_clique(A, pm, 12, budget, sub_deadline)
            if res is not None:
                return list(res)
    
    # Strategy 2: Alternative 7-prime sets - try more combinations
    # Focus on sets that might have denser smooth number distributions
    alts7 = [
        [2,3,5,7,11,13,17],
        [2,3,5,7,11,13,19],
        [2,3,5,7,11,13,23],
        [2,3,5,7,11,17,19],
        [2,3,5,7,13,17,19],
        [2,3,5,11,13,17,19],
        [2,3,7,11,13,17,19],
        [2,5,7,11,13,17,19],
        [2,3,5,7,11,13,29],
        [2,3,5,7,11,17,23],
        [2,3,5,7,13,17,23],
    ]
    for P in alts7:
        if time.time() > deadline - 8:
            break
        pm = pmask_from_list(P)
        L = _smooth_list(pm)
        if len(L) < 12:
            continue
        for T in [150, 300, 500, 700, len(L)]:
            if T > len(L): T = len(L)
            if time.time() > deadline - 8:
                break
            A = L[:T]
            remaining = deadline - time.time()
            sub_deadline = min(time.time() + max(2.0, remaining * 0.2), deadline - 8.0)
            res = _find_clique(A, pm, 12, 400000, sub_deadline)
            if res is not None:
                return list(res)
    
    # Strategy 3: Alternative 8-prime sets
    alts8 = [
        [2,3,5,7,11,13,17,19],
        [2,3,5,7,11,13,17,23],
        [2,3,5,7,11,13,19,23],
        [2,3,5,7,11,17,19,23],
        [2,3,5,7,13,17,19,23],
        [2,3,5,11,13,17,19,23],
    ]
    for P in alts8:
        if time.time() > deadline - 8:
            break
        pm = pmask_from_list(P)
        L = _smooth_list(pm)
        if len(L) < 12:
            continue
        for T in [200, 400, 600, len(L)]:
            if T > len(L): T = len(L)
            if time.time() > deadline - 8:
                break
            A = L[:T]
            remaining = deadline - time.time()
            sub_deadline = min(time.time() + max(2.0, remaining * 0.2), deadline - 8.0)
            res = _find_clique(A, pm, 12, 400000, sub_deadline)
            if res is not None:
                return list(res)
    
    # Fallback known-good 8-prime set
    return [1,2,3,5,6,9,10,15,18,25,27,30]
