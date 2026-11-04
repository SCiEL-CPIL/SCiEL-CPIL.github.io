"""QAP tai256c solver: ILS with spectral init, vectorized delta, and Hungarian re-optimization."""
import numpy as np
import time
import random
from scipy.optimize import linear_sum_assignment


def _compute_cost(a, b, p):
    """Compute total QAP cost for permutation p."""
    return np.sum(a * b[np.ix_(p, p)])


def _fast_delta(a, a_sym, D, i, j):
    """
    Compute delta cost for swapping positions i and j in permutation p.
    a_sym = a + a.T (symmetric flow matrix)
    D is the current distance matrix b[p, p].
    Uses vectorized computation: S = (a_sym[i] - a_sym[j]) dot (D[j] - D[i])
    """
    # Main interaction term
    diff_a = a_sym[i] - a_sym[j]
    diff_D = D[j] - D[i]
    S = np.dot(diff_a, diff_D)
    
    # Subtract k=i and k=j self-interaction corrections
    # For k=i: terms involving a[i,i], a[i,j], a[j,i], a[j,j] with D changes
    ai_i = a[i, i]
    ai_j = a[i, j]
    aj_i = a[j, i]
    aj_j = a[j, j]
    
    Di_i = D[i, i]
    Di_j = D[i, j]
    Dj_i = D[j, i]
    Dj_j = D[j, j]
    
    # k=i contribution: when facility i moves from pos i to pos j
    # Original: a[i,i]*D[i,i] + a[i,j]*D[i,j] + a[j,i]*D[j,i] + a[j,j]*D[j,j]
    # New: a[i,i]*D[j,j] + a[i,j]*D[j,i] + a[j,i]*D[i,j] + a[j,j]*D[i,i]
    ki_orig = ai_i * Di_i + ai_j * Di_j + aj_i * Dj_i + aj_j * Dj_j
    ki_new = ai_i * Dj_j + ai_j * Dj_i + aj_i * Di_j + aj_j * Di_i
    
    # k=j contribution: when facility j moves from pos j to pos i
    # This is captured in the main S term, but we need to be careful
    # The S term already accounts for the change in interactions with all k
    
    # Actually, let's use the cleaner formulation from the inspiration:
    # delta = sum_k [a[i,k]*(D[k,j]-D[k,i]) + a[j,k]*(D[k,i]-D[k,j])]
    #        + sum_k [a[k,i]*(D[i,k]-D[j,k]) + a[k,j]*(D[j,k]-D[i,k])]
    # But this double counts. Let's use the standard 2-opt delta:
    
    # More carefully:
    # For each k != i, j:
    #   old: a[i,k]*D[p_i,k] + a[k,i]*D[k,p_i] + a[j,k]*D[p_j,k] + a[k,j]*D[k,p_j]
    #   new: a[i,k]*D[p_j,k] + a[k,i]*D[k,p_j] + a[j,k]*D[p_i,k] + a[k,j]*D[k,p_i]
    #   diff: a[i,k]*(D[p_j,k]-D[p_i,k]) + a[k,i]*(D[k,p_j]-D[k,p_i]) 
    #          + a[j,k]*(D[p_i,k]-D[p_j,k]) + a[k,j]*(D[k,p_i]-D[k,p_j])
    #        = (a[i,k]-a[j,k])*(D[p_j,k]-D[p_i,k]) + (a[k,i]-a[k,j])*(D[k,p_j]-D[k,p_i])
    
    # For k=i:
    #   old: a[i,i]*D[i,i] + a[i,j]*D[i,j]
    #   new: a[i,i]*D[j,j] + a[i,j]*D[j,i]
    #   diff: a[i,i]*(D[j,j]-D[i,i]) + a[i,j]*(D[j,i]-D[i,j])
    
    # For k=j:
    #   old: a[j,i]*D[j,i] + a[j,j]*D[j,j]
    #   new: a[j,i]*D[i,j] + a[j,j]*D[i,i]
    #   diff: a[j,i]*(D[i,j]-D[j,i]) + a[j,j]*(D[i,i]-D[j,j])
    
    # Combined:
    # delta = sum_{k!=i,j} [(a[i,k]-a[j,k])*(D[j,k]-D[i,k]) + (a[k,i]-a[k,j])*(D[k,j]-D[k,i])]
    #         + a[i,i]*(D[j,j]-D[i,i]) + a[i,j]*(D[j,i]-D[i,j])
    #         + a[j,i]*(D[i,j]-D[j,i]) + a[j,j]*(D[i,i]-D[j,j])
    
    # Note: (a[i,k]-a[j,k])*(D[j,k]-D[i,k]) + (a[k,i]-a[k,j])*(D[k,j]-D[k,i])
    #      = (a[i,k]-a[j,k])*(D[j,k]-D[i,k]) + (a[i,k]-a[j,k])*(D[j,k]-D[i,k])  [if a symmetric]
    #      = 2*(a[i,k]-a[j,k])*(D[j,k]-D[i,k])
    
    # But a may not be symmetric! So we keep both terms.
    
    # Vectorized:
    mask = np.ones(n, dtype=bool)
    mask[i] = False
    mask[j] = False
    
    term1 = np.dot(a[i, mask] - a[j, mask], D[j, mask] - D[i, mask])
    term2 = np.dot(a[mask, i] - a[mask, j], D[mask, j] - D[mask, i])
    
    corner = (ai_i * (Dj_j - Di_i) + ai_j * (Dj_i - Di_j) + 
              aj_i * (Di_j - Dj_i) + aj_j * (Di_i - Dj_j))
    
    return term1 + term2 + corner


def _local_search_2opt(a, b, p, deadline):
    """
    Fast local search with 2-opt swaps using vectorized delta.
    Returns (p, cost).
    """
    n = len(p)
    p = p.copy()
    a_sym = a + a.T
    D = b[np.ix_(p, p)].copy()
    current_cost = np.sum(a * D)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n):
            if time.time() > deadline:
                break
            for j in range(i + 1, n):
                delta = _fast_delta(a, a_sym, D, i, j)
                if delta < 0:
                    # Perform swap in p
                    p[i], p[j] = p[j], p[i]
                    
                    # Update D: swap rows i and j, then cols i and j
                    D[i], D[j] = D[j].copy(), D[i].copy()
                    D[:, i], D[:, j] = D[:, j].copy(), D[:, i].copy()
                    
                    current_cost += delta
                    improved = True
                    # Restart inner loop for correctness with new D
                    break
            if improved:
                continue
    return p, current_cost


def _hungarian_reopt(a, b, p):
    """
    Use Hungarian algorithm on a linearized cost matrix to potentially escape local optima.
    This provides a different perspective on the assignment.
    """
    n = len(p)
    # Linearize: C[i, j] = sum_k a[i,k]*b[j, p[k]] + sum_k a[k,i]*b[p[k], j]
    B_p = b[np.ix_(np.arange(n), p)]
    M1 = a @ B_p.T
    Bp2 = b[np.ix_(p, np.arange(n))]
    M2 = a.T @ Bp2
    C = M1 + M2
    row_ind, col_ind = linear_sum_assignment(C)
    new_p = np.empty(n, dtype=np.int64)
    new_p[row_ind] = col_ind
    return new_p


def _spectral_init(a, b):
    """
    Spectral bisection initialization: recursively partition facilities by flow
    and locations by distance, then assign hierarchically.
    """
    n = len(a)
    
    def fiedler_partition(matrix, size):
        """Find Fiedler vector of graph Laplacian to bisect nodes."""
        if size <= 2:
            return np.ones(size, dtype=bool)
        
        D = np.diag(matrix.sum(axis=1))
        L = D - matrix
        try:
            vals, vecs = np.linalg.eigh(L)
            idx = np.argsort(vals)
            fiedler = vecs[:, idx[1]]
        except Exception:
            fiedler = np.random.randn(size)
        
        positive = fiedler > 0
        # Balance: ensure roughly equal sizes
        n_pos = np.sum(positive)
        target = size // 2
        if n_pos > target:
            # Move some from positive to negative
            pos_indices = np.where(positive)[0]
            to_move = n_pos - target
            for idx in pos_indices[-to_move:]:
                positive[idx] = False
        elif n_pos < target:
            neg_indices = np.where(~positive)[0]
            to_move = target - n_pos
            for idx in neg_indices[:to_move]:
                positive[idx] = True
        return positive
    
    def recursive_assign(facilities, locations, depth=0):
        """Recursively bisect and assign."""
        if len(facilities) <= 1:
            if len(facilities) == 1 and len(locations) == 1:
                return {facilities[0]: locations[0]}
            return {}
        
        if len(facilities) <= 4:
            # Direct assignment using Hungarian on small subproblem
            sub_a = a[np.ix_(facilities, facilities)]
            sub_b = b[np.ix_(locations, locations)]
            C = np.zeros((len(facilities), len(locations)))
            for i in range(len(facilities)):
                for j in range(len(locations)):
                    C[i, j] = np.sum(sub_a * sub_b[np.ix_([j]*len(facilities), [j]*len(facilities))])
            # Simpler: just use cost matrix
            C2 = np.array([[np.sum(sub_a * sub_b) for _ in range(len(locations))] for _ in range(len(facilities))])
            # Actually, let's just assign in order for small cases
            result = {}
            for i, f in enumerate(facilities):
                result[f] = locations[i]
            return result
        
        # Bisect facilities by flow graph
        sub_a = a[np.ix_(facilities, facilities)]
        fac_partition = fiedler_partition(sub_a, len(facilities))
        
        # Bisect locations by distance graph (use negative distance as similarity)
        sub_b = b[np.ix_(locations, locations)]
        loc_partition = fiedler_partition(-sub_b, len(locations))
        
        # Split into two groups
        fac_group1 = [facilities[i] for i in range(len(facilities)) if fac_partition[i]]
        fac_group2 = [facilities[i] for i in range(len(facilities)) if not fac_partition[i]]
        loc_group1 = [locations[i] for i in range(len(locations)) if loc_partition[i]]
        loc_group2 = [locations[i] for i in range(len(locations)) if not loc_partition[i]]
        
        # Balance the partitions
        if len(fac_group1) > len(loc_group1) and len(fac_group2) > 0:
            diff = len(fac_group1) - len(loc_group1)
            for _ in range(min(diff, len(fac_group1))):
                fac_group2.append(fac_group1.pop())
        elif len(fac_group1) < len(loc_group1) and len(fac_group2) > 0:
            diff = len(loc_group1) - len(fac_group1)
            for _ in range(min(diff, len(fac_group2))):
                fac_group1.append(fac_group2.pop())
        
        # Ensure groups are non-empty
        if len(fac_group1) == 0:
            fac_group1 = [facilities[0]]
            fac_group2 = facilities[1:]
        if len(fac_group2) == 0:
            fac_group2 = [facilities[-1]]
            fac_group1 = facilities[:-1]
        if len(loc_group1) == 0:
            loc_group1 = [locations[0]]
            loc_group2 = locations[1:]
        if len(loc_group2) == 0:
            loc_group2 = [locations[-1]]
            loc_group1 = locations[:-1]
        
        # Recursively assign
        result = {}
        result.update(recursive_assign(fac_group1, loc_group1, depth + 1))
        result.update(recursive_assign(fac_group2, loc_group2, depth + 1))
        
        return result
    
    all_facilities = list(range(n))
    all_locations = list(range(n))
    
    assignment = recursive_assign(all_facilities, all_locations)
    
    # Build permutation
    p = np.empty(n, dtype=np.int64)
    for fac, loc in assignment.items():
        p[fac] = loc
    
    # Handle any unassigned
    assigned_locs = set(assignment.values())
    unassigned_facs = [f for f in range(n) if f not in assignment]
    unassigned_locs = [l for l in range(n) if l not in assigned_locs]
    for f, l in zip(unassigned_facs, unassigned_locs):
        p[f] = l
    
    return p


def _perturb(p, k=5):
    """Randomly swap k pairs of positions."""
    p = p.copy()
    n = len(p)
    for _ in range(k):
        i, j = random.sample(range(n), 2)
        p[i], p[j] = p[j], p[i]
    return p


def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin under 150s limit
    
    best_p = None
    best_cost = np.inf
    
    # Try multiple initializations
    inits = []
    
    # 1. Spectral initialization
    p_spec = _spectral_init(a, b)
    inits.append(p_spec)
    
    # 2. Rank-based initialization
    a_row_sums = a.sum(axis=1)
    b_col_sums = b.sum(axis=1)
    heavy_fac = np.argsort(-a_row_sums, kind='stable')
    central_loc = np.argsort(b_col_sums, kind='stable')
    p_rank = np.empty(n, dtype=np.int64)
    for i, f in enumerate(heavy_fac):
        p_rank[f] = central_loc[i]
    inits.append(p_rank)
    
    # 3. Random initialization
    rng = random.Random(42)
    p_rand = np.arange(n, dtype=np.int64)
    rng.shuffle(p_rand.tolist())
    inits.append(p_rand)
    
    # Local search on each initialization
    for p_init in inits:
        if time.time() > deadline:
            break
        remaining = deadline - time.time()
        if remaining < 5:
            break
        
        p_ls, cost_ls = _local_search_2opt(a, b, p_init, deadline - 2.0)
        
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()
    
    # Iterated Local Search with Hungarian re-optimization
    perturb_k = 5
    hungarian_interval = 10  # Try Hungarian every N iterations
    
    iteration = 0
    while time.time() < deadline - 5.0:
        iteration += 1
        
        # Perturb
        p_pert = _perturb(best_p, k=perturb_k)
        
        # Local search
        remaining = deadline - time.time()
        if remaining < 3.0:
            break
        
        p_ls, cost_ls = _local_search_2opt(a, b, p_pert, deadline - 2.0)
        
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()
            perturb_k = max(3, perturb_k - 1)  # Reduce perturbation on improvement
        else:
            # Occasionally try Hungarian re-optimization to escape local optima
            if iteration % hungarian_interval == 0 and remaining > 10.0:
                p_hung = _hungarian_reopt(a, b, best_p)
                p_hung_ls, cost_hung = _local_search_2opt(a, b, p_hung, deadline - 2.0)
                if cost_hung < best_cost:
                    best_cost = cost_hung
                    best_p = p_hung_ls.copy()
        
        # Adaptive perturbation: increase if stuck
        if iteration % 20 == 0:
            perturb_k = min(15, perturb_k + 2)
    
    return best_p.tolist()
