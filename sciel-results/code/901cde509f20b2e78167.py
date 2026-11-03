"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.linalg import eigh


def _compute_cost(a, b, p):
    """Compute QAP cost for permutation p."""
    bp = b[np.ix_(p, p)]
    return int(np.sum(a * bp))


def _delta_swap(a, b, p, i, j):
    """Compute delta cost for swapping facilities i and j in permutation p."""
    pi = p[i]
    pj = p[j]
    
    # Delta = sum_k a[i,k]*(b[pj,p[k]] - b[pi,p[k]]) + a[j,k]*(b[pi,p[k]] - b[pj,p[k]])
    #       + sum_k a[k,i]*(b[p[k],pj] - b[p[k],pi]) + a[k,j]*(b[p[k],pi] - b[p[k],pj])
    #       + a[i,j]*(b[pj,pj] - b[pi,pi]) + a[j,i]*(b[pi,pi] - b[pj,pj])
    # But b is symmetric, so b[pj,pj]=b[pi,pi] for the diagonal terms... wait no.
    # Let me be more careful.
    
    # Original contribution involving i or j:
    # sum_{k != i,j} a[i,k]*b[pi,p[k]] + a[j,k]*b[pj,p[k]]
    # + sum_{k != i,j} a[k,i]*b[p[k],pi] + a[k,j]*b[p[k],pj]
    # + a[i,j]*b[pi,pj] + a[j,i]*b[pj,pi]
    
    # After swap, i goes to pj, j goes to pi:
    # sum_{k != i,j} a[i,k]*b[pj,p[k]] + a[j,k]*b[pi,p[k]]
    # + sum_{k != i,j} a[k,i]*b[p[k],pj] + a[k,j]*b[p[k],pi]
    # + a[i,j]*b[pj,pi] + a[j,i]*b[pi,pj]
    
    # Since b is symmetric: b[pj,pi] = b[pi,pj], so the cross terms a[i,j]*b[pj,pi] + a[j,i]*b[pi,pj]
    # = (a[i,j] + a[j,i]) * b[pi,pj]
    # Original: a[i,j]*b[pi,pj] + a[j,i]*b[pj,pi] = (a[i,j] + a[j,i]) * b[pi,pj]
    # So cross terms don't change!
    
    # Delta = sum_{k != i,j} [a[i,k] - a[j,k]] * (b[pj,p[k]] - b[pi,p[k]])
    #       + sum_{k != i,j} [a[k,i] - a[k,j]] * (b[p[k],pj] - b[p[k],pi])
    
    # Since a and b are symmetric: a[i,k] = a[k,i], b[pj,p[k]] = b[p[k],pj]
    # So: Delta = 2 * sum_{k != i,j} (a[i,k] - a[j,k]) * (b[pj,p[k]] - b[pi,p[k]])
    
    # Vectorized:
    mask = np.ones(len(p), dtype=bool)
    mask[i] = False
    mask[j] = False
    ks = np.where(mask)[0]
    
    if len(ks) == 0:
        return 0
    
    diff_a = a[i, ks] - a[j, ks]
    diff_b = b[pj, p[ks]] - b[pi, p[ks]]
    
    return int(2 * np.sum(diff_a * diff_b))


def _spectral_init(a, b, n, seed=0):
    """Generate initial permutation using spectral methods."""
    rng = np.random.RandomState(seed)
    
    # Compute eigenvectors of A and B
    # Use the top few eigenvectors for embedding
    try:
        evals_a, evecs_a = eigh(a, subset_by_index=[0, min(10, n-1)])
        evals_b, evecs_b = eigh(b, subset_by_index=[0, min(10, n-1)])
    except:
        # Fallback to rank matching
        heavy = np.argsort(-a.sum(axis=1), kind="stable")
        central = np.argsort(b.sum(axis=1), kind="stable")
        p = np.empty(n, dtype=np.int64)
        p[heavy] = central
        return p
    
    # Project facilities and locations onto the eigenspace
    # Use the eigenvectors corresponding to the smallest eigenvalues of B (most central)
    # and largest eigenvalues of A (most connected)
    
    # For A, use eigenvectors with largest eigenvalues
    idx_a = np.argsort(evals_a)[::-1][:min(10, n)]
    coords_a = evecs_a[:, idx_a]
    
    # For B, use eigenvectors with smallest eigenvalues (most "central" in some sense)
    idx_b = np.argsort(evals_b)[:min(10, n)]
    coords_b = evecs_b[:, idx_b]
    
    # Match facilities to locations by minimizing distance in the embedding space
    # For each facility, find the closest location
    p = np.empty(n, dtype=np.int64)
    
    # Use a greedy matching based on the embedding
    # Facility i has coordinate coords_a[i], location j has coordinate coords_b[j]
    # We want to assign facility i to location j that minimizes ||coords_a[i] - coords_b[j]||^2
    
    # For efficiency, use a simple greedy approach
    assigned = np.zeros(n, dtype=bool)
    for i in range(n):
        if assigned[i]:
            continue
        # Find best unassigned location
        diffs = coords_b - coords_a[i]
        dists = np.sum(diffs * diffs, axis=1)
        dists[assigned] = np.inf
        j = np.argmin(dists)
        p[i] = j
        assigned[j] = True
    
    return p


def _rank_matching_init(a, b, n):
    """Simple rank-matching initialization."""
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    return p


def _random_init(n, rng):
    """Random initialization."""
    p = np.arange(n)
    rng.shuffle(p)
    return p


def _local_search(a, b, p, max_iters=50000, time_limit=None, start_time=None):
    """Local search with 2-opt swaps."""
    n = len(p)
    p = p.copy().astype(np.int64)
    cost = _compute_cost(a, b, p)
    best_p = p.copy()
    best_cost = cost
    
    if time_limit is None:
        time_limit = 100.0
    if start_time is None:
        start_time = time.time()
    
    no_improve = 0
    max_no_improve = 500
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        # Randomly pick a pair to try swapping
        i = np.random.randint(n)
        j = np.random.randint(n)
        if i == j:
            continue
        
        delta = _delta_swap(a, b, p, i, j)
        
        if delta < 0:
            # Accept the swap
            p[i], p[j] = p[j], p[i]
            cost += delta
            no_improve = 0
            if cost < best_cost:
                best_cost = cost
                best_p = p.copy()
        else:
            no_improve += 1
            if no_improve > max_no_improve:
                break
    
    return best_p, best_cost


def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Precompute some useful arrays
    a_int = a.astype(np.int64)
    b_int = b.astype(np.int64)
    
    best_p = None
    best_cost = np.inf
    
    # Generate multiple diverse initial solutions
    inits = []
    
    # 1. Spectral initialization
    p_spec = _spectral_init(a_int, b_int, n, seed=42)
    inits.append(p_spec)
    
    # 2. Rank matching
    p_rank = _rank_matching_init(a_int, b_int, n)
    inits.append(p_rank)
    
    # 3. Random initializations
    rng = np.random.RandomState(123)
    for _ in range(3):
        inits.append(_random_init(n, rng))
    
    # 4. Another spectral with different seed
    p_spec2 = _spectral_init(a_int, b_int, n, seed=99)
    inits.append(p_spec2)
    
    # Evaluate all initial solutions and keep the best few
    scored_inits = []
    for p_init in inits:
        cost = _compute_cost(a_int, b_int, p_init)
        scored_inits.append((cost, p_init))
    
    scored_inits.sort(key=lambda x: x[0])
    
    # Run local search on the top initial solutions
    num_ls = min(len(scored_inits), 5)
    time_per_ls = 100.0 / num_ls
    
    for idx in range(num_ls):
        if time.time() > deadline:
            break
        
        cost_init, p_init = scored_inits[idx]
        
        # Run local search
        p_ls, cost_ls = _local_search(
            a_int, b_int, p_init,
            max_iters=100000,
            time_limit=time_per_ls,
            start_time=time.time()
        )
        
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()
    
    # If we still have time, do a final polish on the best solution
    if time.time() < deadline - 5.0 and best_p is not None:
        remaining_time = deadline - 5.0 - time.time()
        p_polish, cost_polish = _local_search(
            a_int, b_int, best_p,
            max_iters=100000,
            time_limit=remaining_time,
            start_time=time.time()
        )
        if cost_polish < best_cost:
            best_cost = cost_polish
            best_p = p_polish.copy()
    
    if best_p is None:
        # Fallback
        best_p = _rank_matching_init(a_int, b_int, n)
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
