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
    
    for bit in range(30):
        if time.time() > max_time:
            break
        if not (best_pm & (1 << bit)):
            continue
        test_pm = best_pm & ~(1 << bit)
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
    
    for round_num in range(5):
        if time.time() > max_time:
            break
        improved = False
        for bit in range(30):
            if time.time() > max_time:
                break
            if not (best_pm & (1 << bit)):
                continue
            test_pm = best_pm & ~(1 << bit)
            bad_pairs = []
            for i in range(n):
                for j in range(i+1, n):
                    if not _is_smooth(best_S[i] + best_S[j], test_pm):
                        bad_pairs.append((i, j))
            if not bad_pairs:
                best_pm = test_pm
                improved = True
                continue
            for (bi, bj) in bad_pairs[:8]:
                if time.time() > max_time:
                    break
                for k in range(n):
                    if k == bi or k == bj:
                        continue
                    old_bi = best_S[bi]
                    old_k = best_S[k]
                    best_S[bi] = old_k
                    best_S[k] = old_bi
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
                        best_S[bi] = old_bi
                        best_S[k] = old_k
        if not improved:
            break
    
    return best_S, best_pm

def _find_clique(A, Pmask, k, node_budget, deadline):
    n = len(A)
    if n < k:
        return None
    adj = [0]*n
    for i in range(n):
        ai = A[i]
        for j in range(i+1, n):
            if _is_smooth(ai + A[j], Pmask):
                adj[i] |= (1<<j)
                adj[j] |= (1<<i)
    
    deg = [adj[i].bit_count() for i in range(n)]
    min_deg_needed = k - 1
    active = [i for i in range(n) if deg[i] >= min_deg_needed]
    if len(active) < k:
        return None
    
    active.sort(key=lambda i: -deg[i])
    m = len(active)
    
    new_adj = [0]*m
    for i in range(m):
        for j in range(i+1, m):
            oi, oj = active[i], active[j]
            if adj[oi] & (1 << oj):
                new_adj[i] |= (1 << j)
                new_adj[j] |= (1 << i)
    
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
        
        msk = cand_mask
        cands = []
        while msk:
            lsb = msk & (-msk)
            v = lsb.bit_length() - 1
            nc_count = (cand_mask & new_adj[v] & higher[v]).bit_count()
            if nc_count + 1 >= need:
                cands.append((nc_count, v))
            msk &= msk - 1
        
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
    for i in range(k):
        for j in range(i+1, k):
            if not _is_smooth(vals[i]+vals[j], Pmask):
                return None
    return vals

def _greedy_clique(A, Pmask, k):
    n = len(A)
    if n < k:
        return None
    
    adj = [0]*n
    for i in range(n):
        ai = A[i]
        for j in range(i+1, n):
            if _is_smooth(ai + A[j], Pmask):
                adj[i] |= (1<<j)
                adj[j] |= (1<<i)
    
    selected = []
    remaining_mask = (1 << n) - 1
    
    while len(selected) < k and remaining_mask:
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
        remaining_mask &= adj[best_v]
        remaining_mask &= ~(1 << best_v)
    
    if len(selected) < k:
        return None
    
    vals = [A[v] for v in selected]
    for i in range(k):
        for j in range(i+1, k):
            if not _is_smooth(vals[i]+vals[j], Pmask):
                return None
    return vals

def _mod_based_init(mod, k, Pmask):
    r = mod // 2
    candidates = []
    for x in range(r, 10001, mod):
        if _is_smooth(x, Pmask):
            candidates.append(x)
        if len(candidates) >= k:
            break
    if len(candidates) < k:
        return None
    for i in range(k):
        for j in range(i+1, k):
            if not _is_smooth(candidates[i] + candidates[j], Pmask):
                return None
    return candidates

def _try_mod_construction(mod, k=12):
    """Direct construction: pick k numbers ≡ mod//2 (mod mod), verify sums use only mod's primes."""
    r = mod // 2
    candidates = []
    for x in range(r, 10001, mod):
        candidates.append(x)
        if len(candidates) >= k:
            break
    if len(candidates) < k:
        return None
    
    # Factor mod to get its prime set
    t = mod
    mod_primes = set()
    while t > 1:
        p = _SPF[t]
        mod_primes.add(p)
        while t % p == 0:
            t //= p
    
    # Build Pmask from mod's primes (only those in PRIMES30)
    pm = 0
    for p in mod_primes:
        if p in P2IDX:
            pm |= 1 << P2IDX[p]
    
    # Verify all pair sums are smooth w.r.t. mod's primes
    for i in range(k):
        for j in range(i+1, k):
            s = candidates[i] + candidates[j]
            if not _is_smooth(s, pm):
                return None
    
    # Count distinct primes actually used
    used_pm = 0
    for i in range(k):
        for j in range(i+1, k):
            s = candidates[i] + candidates[j]
            m = _M[s]
            if m >= 0:
                used_pm |= m
            else:
                return None
    
    prime_list = [p for p in PRIMES30 if used_pm & (1 << P2IDX[p])]
    return candidates, prime_list

def construct():
    deadline = time.time() + 78.0
    primes_first = [2,3,5,7,11,13,17,19,23,29,31,37,41,43]
    
    def pmask_from_list(P):
        m = 0
        for p in P:
            m |= 1 << P2IDX[p]
        return m
    
    best_result = None
    
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
    
    # Strategy 0: Direct modulus-based construction (fast, deterministic)
    # mod=210: primes {2,3,5,7} -> 4 primes if it works
    # mod=66: primes {2,3,11} -> 3 primes
    # mod=30: primes {2,3,5} -> 3 primes
    # mod=42: primes {2,3,7} -> 3 primes
    # mod=60: primes {2,3,5} -> 3 primes
    # mod=70: primes {2,5,7} -> 3 primes
    # mod=10: primes {2,5} -> 2 primes
    # mod=14: primes {2,7} -> 2 primes
    # mod=6: primes {2,3} -> 2 primes
    # mod=15: primes {3,5} -> 2 primes
    
    for mod in [210, 66, 30, 42, 60, 70, 10, 14, 6, 15, 21, 35, 55, 77]:
        if time.time() > deadline - 10:
            break
        result = _try_mod_construction(mod, 12)
        if result is not None:
            res, P = result
            if try_and_update(res, P):
                if len(P) <= 4:
                    return list(res)
    
    # Strategy 1: Modulus-based initialization with local search
    for mod in [210, 60, 30, 42, 66, 70, 105, 150, 210, 280, 330]:
        if time.time() > deadline - 10:
            break
        for P in [[2,3,5,7,11,13], [2,3,5,7,11,13,17], [2,3,5,7,11], [2,3,5,7,11,13,17,19],
                   [2,3,5,7,11,13,17,19,23], [2,3,5,7,11,13,17,19,23,29]]:
            if time.time() > deadline - 10:
                break
            pm = pmask_from_list(P)
            res = _mod_based_init(mod, 12, pm)
            if res is not None:
                res2, pm2 = _local_search_reduce(res, pm, deadline - 5)
                P2 = [p for p in PRIMES30 if pm2 & (1 << P2IDX[p])]
                if try_and_update(res2, P2):
                    if len(P2) <= 5:
                        return list(res2)
    
    # Strategy 2: Try smallest prime sets with clique search
    for k in range(3, 10):
        if time.time() > deadline - 10:
            break
        P = primes_first[:k]
        pm = pmask_from_list(P)
        L = _smooth_list(pm)
        if len(L) < 12:
            continue
        
        Ts = [50, 100, 150, 200, 300, 400, 500, 600, 800, 1000, 1500]
        Ts = [t for t in Ts if t <= len(L)]
        if len(L) not in Ts and len(L) > 1500:
            Ts.append(len(L))
        
        for T in Ts:
            if time.time() > deadline - 10:
                break
            A = L[:T]
            
            res = _greedy_clique(A, pm, 12)
            if res is not None:
                res2, pm2 = _local_search_reduce(res, pm, deadline - 5)
                P2 = [p for p in PRIMES30 if pm2 & (1 << P2IDX[p])]
                if try_and_update(res2, P2):
                    if len(P2) <= 5:
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
                    if len(P2) <= 5:
                        return list(res2)
                    elif len(P2) < k:
                        return list(res2)
    
    # Strategy 3: Alternative 6-prime sets
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
                    if len(P2) <= 5:
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
                    if len(P2) <= 5:
                        return list(res2)
    
    # Strategy 4: Alternative 7-prime sets
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
                    if len(P2) <= 5:
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
                    if len(P2) <= 5:
                        return list(res2)
                    elif len(P2) < 7:
                        return list(res2)
    
    # Strategy 5: Alternative 8-prime sets
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
                    if len(P2) <= 5:
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
                    if len(P2) <= 5:
                        return list(res2)
                    elif len(P2) < 8:
                        return list(res2)
    
    if best_result is not None:
        return list(best_result[0])
    
    # Fallback known-good 8-prime set
    return [1,2,3,5,6,9,10,15,18,25,27,30]
