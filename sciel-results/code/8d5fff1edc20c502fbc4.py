# EVOLVE-BLOCK-START
import time
import numpy as np


def _compute_cost(a, b, p):
    """Exact integer QAP cost for permutation p (facility i at location p[i])."""
    return int(np.sum(a * b[np.ix_(p, p)]))


def _swap_delta(a, b, p, i, j):
    """
    Calculate the change in cost if we swap facilities i and j.
    Vectorized O(n) calculation.
    """
    u = p[i]
    v = p[j]
    
    # Difference in flow rows
    diff_a_rows = a[i, :] - a[j, :]
    # Difference in distance rows for locations u and v, indexed by current permutation p
    # b[v, p[k]] - b[u, p[k]] for all k
    diff_b_rows = b[v, p] - b[u, p]
    s1 = np.dot(diff_a_rows, diff_b_rows)
    
    # Difference in flow cols
    diff_a_cols = a[:, i] - a[:, j]
    # Difference in distance cols for locations u and v, indexed by current permutation p
    # b[p[k], v] - b[p[k], u] for all k
    diff_b_cols = b[p, v] - b[p, u]
    s2 = np.dot(diff_a_cols, diff_b_cols)
    
    # Corrections for k=i and k=j terms included in s1 and s2
    # The dot products include k=i and k=j. We need to adjust because the interaction
    # between i and j is handled separately or the diagonal terms are different.
    
    # Let's derive the correction carefully.
    # Cost = sum_{m,n} a[m,n] b[p[m], p[n]]
    # Swap i, j. p[i]=u, p[j]=v.
    # Terms involving i or j:
    # 1. a[i,i] b[u,u] -> a[i,i] b[v,v]
    # 2. a[j,j] b[v,v] -> a[j,j] b[u,u]
    # 3. a[i,j] b[u,v] -> a[i,j] b[v,u]
    # 4. a[j,i] b[v,u] -> a[j,i] b[u,v]
    # 5. a[i,k] b[u,p[k]] for k!=i,j -> a[i,k] b[v,p[k]]
    # 6. a[k,i] b[p[k],u] for k!=i,j -> a[k,i] b[p[k],v]
    # 7. a[j,k] b[v,p[k]] for k!=i,j -> a[j,k] b[u,p[k]]
    # 8. a[k,j] b[p[k],v] for k!=i,j -> a[k,j] b[p[k],u]
    
    # Our s1 = sum_k (a[i,k]-a[j,k]) * (b[v,p[k]] - b[u,p[k]])
    # = sum_k a[i,k]b[v,p[k]] - sum_k a[i,k]b[u,p[k]] - sum_k a[j,k]b[v,p[k]] + sum_k a[j,k]b[u,p[k]]
    # This covers terms 5,6 (partially, via col) and 7,8.
    # Specifically:
    # s1 includes k=i: (a[i,i]-a[j,i]) * (b[v,u] - b[u,u])
    # s1 includes k=j: (a[i,j]-a[j,j]) * (b[v,v] - b[u,v])
    
    # s2 = sum_k (a[k,i]-a[k,j]) * (b[p[k],v] - b[p[k],u])
    # s2 includes k=i: (a[i,i]-a[i,j]) * (b[u,v] - b[u,u])
    # s2 includes k=j: (a[j,i]-a[j,j]) * (b[v,v] - b[v,u])
    
    # The "true" delta is:
    # Delta = (a[i,i](b[v,v]-b[u,u]) + a[j,j](b[u,u]-b[v,v])) 
    #       + (a[i,j](b[v,u]-b[u,v]) + a[j,i](b[u,v]-b[v,u]))
    #       + sum_{k!=i,j} [ (a[i,k]-a[j,k])(b[v,p[k]]-b[u,p[k]]) + (a[k,i]-a[k,j])(b[p[k],v]-b[p[k],u]) ]
    
    # Let's compute the sum over all k (including i,j) which is s1+s2, and subtract the k=i,j parts, then add the correct i,j,i,j interactions.
    
    # s1_k_i = (a[i,i]-a[j,i]) * (b[v,u] - b[u,u])
    # s1_k_j = (a[i,j]-a[j,j]) * (b[v,v] - b[u,v])
    # s2_k_i = (a[i,i]-a[i,j]) * (b[u,v] - b[u,u])
    # s2_k_j = (a[j,i]-a[j,j]) * (b[v,v] - b[v,u])
    
    ai_i = a[i, i]
    aj_i = a[j, i]
    ai_j = a[i, j]
    aj_j = a[j, j]
    
    bu_u = b[u, u]
    bu_v = b[u, v]
    bv_u = b[v, u]
    bv_v = b[v, v]
    
    # Terms to subtract from s1+s2
    corr_1 = (ai_i - aj_i) * (bv_u - bu_u)
    corr_2 = (ai_j - aj_j) * (bv_v - bu_v)
    corr_3 = (ai_i - ai_j) * (bu_v - bu_u)
    corr_4 = (aj_i - aj_j) * (bv_v - bv_u)
    
    # Correct diagonal terms
    diag_delta = ai_i * (bv_v - bu_u) + aj_j * (bu_u - bv_v)
    
    # Correct interaction terms
    inter_delta = ai_j * (bv_u - bu_v) + aj_i * (bu_v - bv_u)
    
    return s1 + s2 - (corr_1 + corr_2 + corr_3 + corr_4) + diag_delta + inter_delta


def _local_search(a, b, p, deadline):
    """
    First-improvement local search using vectorized swap deltas.
    """
    n = len(p)
    improved = True
    while improved and time.time() < deadline - 2.0:
        improved = False
        for i in range(n - 1):
            if time.time() > deadline - 2.0:
                return p
            # We can vectorize the search for j for a fixed i?
            # Delta(i, j) for all j > i.
            # u = p[i]
            # diff_a_row = a[i, :] - a[j, :]  <- depends on j
            # This makes full vectorization over j difficult without broadcasting large arrays.
            # Given n=150, the inner loop is small.
            
            best_j = -1
            best_delta = 0
            u = p[i]
            for j in range(i + 1, n):
                d = _swap_delta(a, b, p, i, j)
                if d < best_delta:
                    best_delta = d
                    best_j = j
            
            if best_j != -1:
                p[i], p[best_j] = p[best_j], p[i]
                improved = True
                # Restart scan from beginning to find next first improvement?
                # Or continue? Restarting is standard for first-improvement to ensure we don't miss
                # improvements created by the swap that are earlier in the index.
                # However, restarting increases time. 
                # Let's just break and restart the outer while loop.
                break
    return p


def _perturb(p, k, rng):
    """Random k-swap perturbation."""
    n = len(p)
    p = p.copy()
    # Select 2*k distinct indices and swap them in pairs
    idx = rng.choice(n, size=min(2 * k, n), replace=False)
    for t in range(len(idx) // 2):
        i, j = idx[2 * t], idx[2 * t + 1]
        p[i], p[j] = p[j], p[i]
    return p


def solve(a, b):
    t0 = time.time()
    deadline = t0 + 145.0  # Safety margin
    n = len(a)
    a = np.ascontiguousarray(a, dtype=np.int64)
    b = np.ascontiguousarray(b, dtype=np.int64)
    rng = np.random.default_rng(42)

    best_p = None
    best_cost = np.inf

    # 1. Generate Initializations
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    
    p_rank = np.empty(n, dtype=np.int64)
    p_rank[heavy] = central
    
    p_rev = np.empty(n, dtype=np.int64)
    p_rev[heavy] = central[::-1]
    
    p_rand = rng.permutation(n).astype(np.int64)
    
    inits = [p_rank, p_rev, p_rand]
    
    # 2. Initial Optimization
    for p0 in inits:
        if time.time() > deadline - 5.0:
            break
        p = p0.copy()
        p = _local_search(a, b, p, deadline)
        c = _compute_cost(a, b, p)
        if c < best_cost:
            best_cost = c
            best_p = p.copy()
            
    # 3. Iterated Local Search (ILS)
    if best_p is not None:
        while time.time() < deadline - 10.0:
            # Perturb best solution
            # Adaptive perturbation size? Start with ~n/10 swaps
            k_swaps = max(5, n // 15) 
            p_pert = _perturb(best_p, k_swaps, rng)
            
            # Local Search
            p_opt = _local_search(a, b, p_pert, deadline)
            c_opt = _compute_cost(a, b, p_opt)
            
            if c_opt < best_cost:
                best_cost = c_opt
                best_p = p_opt.copy()
                
            # If time is running out, break
            if time.time() > deadline - 15.0:
                break

    if best_p is None:
        # Fallback
        best_p = p_rank

    return best_p.tolist()
# EVOLVE-BLOCK-END
