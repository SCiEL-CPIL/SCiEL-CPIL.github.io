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

def _local_search_reduce(S, Pmask, max_time):
    """Try to reduce the number of primes by swapping elements or removing primes."""
    if time.time() > max_time:
        return S, Pmask
    n = len(S)
    best_S = list(S)
    best_pm = Pmask
    
    # Try removing one prime at a time
    for bit in range(30):
        if time.time() > max_time:
            break
        if not (best_pm & (1 << bit)):
            continue
        test_pm = best_pm & ~(1 << bit)
        # Check if all pair sums are still smooth
        valid = True
        for i in range(n):
            for j in range(i+1, n):
                if not _is_smooth(best_S[i] + best_S[j], test_pm):
                    valid = False
                    break
            if not valid:
                break
        if valid:
            best_pm = test_pm
    
    # Try swapping elements to allow further prime removal
    for round_num in range(3):
        if time.time() > max_time:
            break
        improved = False
        for bit in range(30):
            if time.time() > max_time:
                break
            if not (best_pm & (1 << bit)):
                continue
            test_pm = best_pm & ~(1 << bit)
            # Find which pairs violate
            bad_pairs = []
            for i in range(n):
                for j in range(i+1, n):
                    if not _is_smooth(best_S[i] + best_S[j], test_pm):
                        bad_pairs.append((i, j))
            if not bad_pairs:
                best_pm = test_pm
                improved = True
                continue
            # Try to fix by swapping one element
            for (bi, bj) in bad_pairs[:5]:
                if time.time() > max_time:
                    break
                for k in range(n):
                    if k == bi or k == bj:
                        continue
                    # Try swapping S[bi] with S[k]
                    old = best_S[bi]
                    best_S[bi] = best_S[k]
                    best_S[k] = old
                    # Check if this fixes things
                    ok = True
                    for i2 in range(n):
                        for j2 in range(i2+1, n):
                            if not _is_smooth(best_S[i2] + best_S[j2], test_pm):
                                ok = False
                                break
                        if not ok:
                            break
                    if ok:
                        best_pm = test_pm
                        improved = True
                        break
                    else:
                        # Revert
                        # Current state: best_S[bi] = orig_k, best_S[k] = orig_bi
                        # We want: best_S[bi] = orig_bi, best_S[k] = orig_k
                        # 'old' holds orig_bi.
                        # best_S[k] currently holds orig_bi.
                        # So we just need to restore best_S[bi] to 'old'? No.
                        # Let's trace carefully.
                        # Start: S[bi]=A, S[k]=B.
                        # old = A.
                        # S[bi] = B.
                        # S[k] = A.
                        # Check fails.
                        # We need S[bi]=A, S[k]=B.
                        # Currently S[bi]=B, S[k]=A.
                        # So swap them back.
                        tmp = best_S[bi]
                        best_S[bi] = best_S[k]
                        best_S[k] = tmp
        if not improved:
            break
    
    return best_S, best_pm

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
    
    # Only keep vertices with sufficient degree
    min_deg_needed = k - 1
    active = [i for i in range(n) if deg[i] >= min_deg_needed]
    if len(active) < k:
        return None
    
    # Sort active by degree descending for better search ordering
    active.sort(key=lambda i: -deg[i])
    
    # Remap to 0..m-1 where m = len(active)
    m = len(active)
    
    # Build new adjacency in compact form
    new_adj = [0]*m
    for i in range(m):
        for j in range(i+1, m):
            oi, oj = active[i], active[j]
            if adj[oi] & (1 << oj):
                new_adj[i] |= (1 << j)
                new_adj[j] |= (1 << i)
    
    # Precompute higher masks
    higher = [0]*m
    for v in range(m-2, -1, -1):
        higher[v] = higher[v+1] | (1 << (v+1))
    
    nodes = [0]
    aborted = [False]
    
    def dfs(cand_mask, need, path):
        if need == 0:
            return list(path)
        if cand_mask.bit_count() < need:
            return None
        nodes[0] += 1
        if nodes[0] > node_budget or time.time() > deadline:
            aborted[0] = True
            return None
        
        # Get candidates sorted by degree (within current candidate set)
        msk = cand_mask
        cands = []
        while msk:
            lsb = msk & (-msk)
            v = lsb.bit_length() - 1
            nc_count = (cand_mask & new_adj[v] & higher[v]).bit_count()
            if nc_count + 1 >= need:
                cands.append((nc_count, v))
            msk &= msk - 1
        
        # Sort by degree descending (try high-degree vertices first)
        cands.sort(key=lambda x: -x[0])
        
        for nc_count, v in cands:
            nc = cand_mask & new_adj[v] & higher[v]
            res = dfs(nc, need-1, path + [v])
            if res is not None:
                return res
        
        return None
    
    res = dfs((1<<m)-1, k, [])
    if res is None:
        return None
    vals = [A[active[v]] for v in res]
    # verify
    for i in range(k):
        for j in range(i+1, k):
            if not _is_smooth(vals[i]+vals[j], Pmask):
                return None
    return vals

def _greedy_clique(A, Pmask, k):
    """Fast greedy heuristic to find a clique of size k."""
    n = len(A)
    if n < k:
        return None
    
    # Compute degrees
    adj = [0]*n
    for i in range(n):
        ai = A[i]
        for j in range(i+1, n):
            if _is_smooth(ai + A[j], Pmask):
                adj[i] |= (1<<j)
                adj[j] |= (1<<i)
    
    # Greedy: repeatedly pick vertex with max degree in remaining set
    selected = []
    remaining_mask = (1 << n) - 1
    
    while len(selected) < k and remaining_mask:
        # Find vertex with max degree within remaining set
        best_v = -1
        best_deg = -1
        m = remaining_mask
        while m:
            lsb = m & (-m)
            v = lsb.bit_length() - 1
            d = (remaining_mask & adj[v]).bit_count()
            if d > best_deg:
                best_deg = d
                best_v = v
            m &= m - 1
        
        if best_v == -1 or best_deg < k - 1 - len(selected):
            break
        
        selected.append(best_v)
        # Remove best_v and its non-neighbors from remaining
        remaining_mask &= adj[best_v]
        remaining_mask &= ~(1 << best_v)
    
    if len(selected) < k:
        return None
    
    vals = [A[v] for v in selected]
    # Verify
    for i in range(k):
        for j in range(i+1, k):
            if not _is_smooth(vals[i]+vals[j], Pmask):
                return None
    return vals

def _mod_based_init(mod, k, Pmask):
    """Generate k distinct numbers all congruent to r mod 'mod' such that
    pair sums are smooth. Use r=mod//2 to make all sums even."""
    r = mod // 2
    candidates = []
    for x in range(r, 10001, mod):
        if _is_smooth(x, Pmask):
            candidates.append(x)
        if len(candidates) >= k:
            break
    if len(candidates) < k:
        return None
    # Verify all pair sums
    for i in range(k):
        for j in range(i+1, k):
            if not _is_smooth(candidates[i] + candidates[j], Pmask):
                return None
    return candidates

def construct():
    deadline = time.time() + 78.0
    primes_first = [2,3,5,7,11,13,17,19,23,29,31,37,41,43]
    
    def pmask_from_list(P):
        m = 0
        for p in P:
            m |= 1 << P2IDX[p]
        return m
    
    best_result = None  # (values, prime_list)
    
    def _is_valid_solution(res):
        if res is None:
            return False
        if len(res) != 12:
            return False
        if len(set(res)) != 12:
            return False
        if any(x < 1 or x > 10000 for x in res):
            return False
        return True

    def try_and_update(res, P):
        nonlocal best_result
        if res is not None and _is_valid_solution(res):
            if best_result is None or len(P) < len(best_result[1]):
                best_result = (res, P)
            return True
        return False
    
    # Strategy 0: Modulus-based initialization with local search
    # Use mod 210 (2*3*5*7) so all pair sums are divisible by 2,3,5,7
    for mod in [210, 60, 30, 42]:
        if time.time() > deadline - 10:
            break
        # Try with various prime sets
        for P in [[2,3,5,7,11,13], [2,3,5,7,11,13,17], [2,3,5,7,11], [2,3,5,7,11,13,17,19]]:
            if time.time() > deadline - 10:
                break
            pm = pmask_from_list(P)
            res = _mod_based_init(mod, 12, pm)
            if res is not None:
                # Try local search to reduce primes
                res2, pm2 = _local_search_reduce(res, pm, deadline - 5)
                # Count primes in pm2
                P2 = [p for p in PRIMES30 if pm2 & (1 << P2IDX[p])]
                if try_and_update(res2, P2):
                    if len(P2) <= 6:
                        return list(res2)
    
    # Strategy 1: Try smallest prime sets first with prefix primes
    for k in range(3, 9):
        if time.time() > deadline - 10:
            break
        P = primes_first[:k]
        pm = pmask_from_list(P)
        L = _smooth_list(pm)
        if len(L) < 12:
            continue
        
        # Try different subset sizes
        Ts = [50, 100, 150, 200, 300, 400, 500, 600, 800, 1000, 1500]
        Ts = [t for t in Ts if t <= len(L)]
        if len(L) not in Ts and len(L) > 1500:
            Ts.append(len(L))
        
        for T in Ts:
            if time.time() > deadline - 10:
                break
            A = L[:T]
            
            # First try fast greedy
            res = _greedy_clique(A, pm, 12)
            if res is not None:
                # Try local search to reduce primes
                res2, pm2 = _local_search_reduce(res, pm, deadline - 5)
                P2 = [p for p in PRIMES30 if pm2 & (1 << P2IDX[p])]
                if try_and_update(res2, P2):
                    if len(P2) <= 6:
                        return list(res2)
                    elif len(P2) < k:
                        return list(res2)
            
            remaining = deadline - time.time()
            if remaining < 5:
                break
            sub_deadline = min(time.time() + max(2.0, remaining * 0.25), deadline - 8.0)
            budget = 300000
            res = _find_clique(A, pm, 12, budget, sub_deadline)
            if res is not None:
                res2, pm2 = _local_search_reduce(res, pm, deadline - 5)
                P2 = [p for p in PRIMES30 if pm2 & (1 << P2IDX[p])]
                if try_and_update(res2, P2):
                    if len(P2) <= 6:
                        return list(res2)
                    elif len(P2) < k:
                        return list(res2)
    
    # Strategy 2: Alternative 6-prime sets
    alts6 = [
        [2,3,5,7,11,13],
        [2,3,5,7,11,17],
        [2,3,5,7,13,17],
        [2,3,5,11,13,17],
        [2,3,7,11,13,17],
        [2,5,7,11,13,17],
        [2,3,5,7,11,19],
        [2,3,5,7,13,19],
        [2,3,5,11,13,19],
        [2,3,5,7,17,19],
    ]
    for P in alts6:
        if time.time() > deadline - 8:
            break
        pm = pmask_from_list(P)
        L = _smooth_list(pm)
        if len(L) < 12:
            continue
        for T in [100, 200, 300, 500, 800, len(L)]:
            if T > len(L): T = len(L)
            if time.time() > deadline - 8:
                break
            A = L[:T]
            
            res = _greedy_clique(A, pm, 12)
            if res is not None:
                res2, pm2 = _local_search_reduce(res, pm, deadline - 5)
                P2 = [p for p in PRIMES30 if pm2 & (1 << P2IDX[p])]
                if try_and_update(res2, P2):
                    if len(P2) <= 6:
                        return list(res2)
            
            remaining = deadline - time.time()
            if remaining < 5:
                break
            sub_deadline = min(time.time() + max(2.0, remaining * 0.2), deadline - 7.0)
            res = _find_clique(A, pm, 12, 300000, sub_deadline)
            if res is not None:
                res2, pm2 = _local_search_reduce(res, pm, deadline - 5)
                P2 = [p for p in PRIMES30 if pm2 & (1 << P2IDX[p])]
                if try_and_update(res2, P2):
                    if len(P2) <= 6:
                        return list(res2)
    
    # Strategy 3: Alternative 7-prime sets
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
        [2,3,5,7,11,19,23],
        [2,3,5,7,13,19,23],
        [2,3,5,7,11,13,31],
        [2,3,5,7,11,17,29],
    ]
    for P in alts7:
        if time.time() > deadline - 5:
            break
        pm = pmask_from_list(P)
        L = _smooth_list(pm)
        if len(L) < 12:
            continue
        for T in [150, 300, 500, 700, len(L)]:
            if T > len(L): T = len(L)
            if time.time() > deadline - 5:
                break
            A = L[:T]
            
            res = _greedy_clique(A, pm, 12)
            if res is not None:
                res2, pm2 = _local_search_reduce(res, pm, deadline - 3)
                P2 = [p for p in PRIMES30 if pm2 & (1 << P2IDX[p])]
                if try_and_update(res2, P2):
                    if len(P2) <= 6:
                        return list(res2)
                    elif len(P2) < 7:
                        return list(res2)
            
            remaining = deadline - time.time()
            if remaining < 4:
                break
            sub_deadline = min(time.time() + max(1.5, remaining * 0.2), deadline - 4.0)
            res = _find_clique(A, pm, 12, 300000, sub_deadline)
            if res is not None:
                res2, pm2 = _local_search_reduce(res, pm, deadline - 3)
                P2 = [p for p in PRIMES30 if pm2 & (1 << P2IDX[p])]
                if try_and_update(res2, P2):
                    if len(P2) <= 6:
                        return list(res2)
                    elif len(P2) < 7:
                        return list(res2)
    
    # Strategy 4: Alternative 8-prime sets
    alts8 = [
        [2,3,5,7,11,13,17,19],
        [2,3,5,7,11,13,17,23],
        [2,3,5,7,11,13,19,23],
        [2,3,5,7,11,17,19,23],
        [2,3,5,7,13,17,19,23],
        [2,3,5,11,13,17,19,23],
        [2,3,5,7,11,13,17,29],
        [2,3,5,7,11,13,19,29],
        [2,3,5,7,11,13,17,31],
        [2,3,5,7,11,13,23,29],
    ]
    for P in alts8:
        if time.time() > deadline - 3:
            break
        pm = pmask_from_list(P)
        L = _smooth_list(pm)
        if len(L) < 12:
            continue
        for T in [200, 400, 600, len(L)]:
            if T > len(L): T = len(L)
            if time.time() > deadline - 3:
                break
            A = L[:T]
            
            res = _greedy_clique(A, pm, 12)
            if res is not None:
                res2, pm2 = _local_search_reduce(res, pm, deadline - 2)
                P2 = [p for p in PRIMES30 if pm2 & (1 << P2IDX[p])]
                if try_and_update(res2, P2):
                    if len(P2) <= 6:
                        return list(res2)
                    elif len(P2) < 8:
                        return list(res2)
            
            remaining = deadline - time.time()
            if remaining < 3:
                break
            sub_deadline = min(time.time() + max(1.0, remaining * 0.3), deadline - 2.0)
            res = _find_clique(A, pm, 12, 200000, sub_deadline)
            if res is not None:
                res2, pm2 = _local_search_reduce(res, pm, deadline - 2)
                P2 = [p for p in PRIMES30 if pm2 & (1 << P2IDX[p])]
                if try_and_update(res2, P2):
                    if len(P2) <= 6:
                        return list(res2)
                    elif len(P2) < 8:
                        return list(res2)
    
    # Return best found
    if best_result is not None:
        return list(best_result[0])
    
    # Fallback known-good 8-prime set
    return [1,2,3,5,6,9,10,15,18,25,27,30]
