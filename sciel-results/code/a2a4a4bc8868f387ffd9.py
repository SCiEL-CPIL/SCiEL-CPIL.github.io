"""
Greedy Pairwise Insertion (GPI) + Local Search Polish
Strategy:
1. Construct an initial permutation using a Greedy Pairwise Insertion heuristic.
   - Start with an empty assignment.
   - At each step, evaluate the cost of inserting each unassigned facility into each available location.
   - Select the (facility, location) pair that results in the minimum increase in total cost.
   - This is a constructive heuristic that builds the solution from scratch, unlike spectral methods which relax and then round.
2. Refine the solution using a fast Local Search (LS) with swap moves.
   - Use vectorized delta calculations for speed.
   - Continue until no improvement is found or time limit is reached.
3. Multi-restart with perturbations if time permits, to escape local optima of the GPI construction.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def _qap_cost(a, b, p):
    """Compute the exact integer cost sum_{i,j} a[i,j] * b[p[i], p[j]]."""
    if p is None or len(p) != len(a):
        return np.inf
    return int(np.sum(a * b[np.ix_(p, p)]))

def _swap_delta(a, b, p, i, j):
    """
    Compute the change in cost when swapping positions i and j in permutation p.
    O(n) vectorized computation.
    """
    pi = p[i]
    pj = p[j]
    
    # Direct terms (i,j pair)
    delta = (a[i, j] - a[j, i]) * (b[pj, pi] - b[pi, pj])
    
    # Diagonal terms
    delta += a[i, i] * (b[pj, pj] - b[pi, pi])
    delta += a[j, j] * (b[pi, pi] - b[pj, pj])
    
    # Cross terms with all other k
    # Term 1: sum_{k != i,j} (a[i,k] - a[j,k]) * (b[pj, p[k]] - b[pi, p[k]])
    diff_a_row = a[i, :] - a[j, :]
    diff_b_col = b[pj, p] - b[pi, p]
    sum_T1 = np.dot(diff_a_row, diff_b_col)
    
    # Subtract k=i and k=j contributions
    sum_T1 -= (a[i, i] - a[j, i]) * (b[pj, pi] - b[pi, pi])
    sum_T1 -= (a[i, j] - a[j, j]) * (b[pj, pj] - b[pi, pj])
    
    # Term 2: sum_{k != i,j} (a[k,i] - a[k,j]) * (b[p[k], pj] - b[p[k], pi])
    diff_a_col = a[:, i] - a[:, j]
    diff_b_row = b[p, pj] - b[p, pi]
    sum_T2 = np.dot(diff_a_col, diff_b_row)
    
    # Subtract k=i and k=j contributions
    sum_T2 -= (a[i, i] - a[i, j]) * (b[pi, pj] - b[pi, pi])
    sum_T2 -= (a[j, i] - a[j, j]) * (b[pj, pj] - b[pj, pi])
    
    delta += sum_T1 + sum_T2
    return int(delta)

def _greedy_pairwise_insertion(a, b):
    """
    Construct a permutation using Greedy Pairwise Insertion.
    """
    n = len(a)
    # p will store the facility assigned to each location
    # p[location] = facility
    # We need to return p such that facility i is at location p[i]
    # The interface requires: p is a permutation of 0..n-1, facility i is at location p[i].
    # So p[i] = location.
    
    # Let's track:
    # assigned_facilities: set of facilities already placed
    # assigned_locations: set of locations already taken
    # current_perm: array of size n, current_perm[location] = facility
    # We build current_perm, then invert it to get the required format.
    
    current_perm = np.full(n, -1, dtype=np.int64)
    unassigned_facilities = list(range(n))
    unassigned_locations = list(range(n))
    
    # For efficiency, we can precompute some parts, but for n=100, O(n^3) is 1,000,000 ops, which is fine.
    # However, we can do better.
    
    # Cost of adding facility f to location l, given current assignments:
    # Delta = a[f,f]*b[l,l] + sum_{k assigned} (a[f,k]*b[l, current_perm[location_of_k]] + a[k,f]*b[current_perm[location_of_k], l])
    
    # Let's maintain a matrix M where M[f, l] is the marginal cost of placing f at l.
    # Initially, M is just a*f,f * b*l,l.
    # When we place facility f0 at location l0, we update M for all remaining f, l:
    # M[f, l] += a[f, f0]*b[l, l0] + a[f0, f]*b[l0, l]
    
    M = a * b  # This is not quite right. M[f,l] should be a[f,f]*b[l,l] initially.
    M = np.outer(np.diag(a), np.diag(b))
    
    for step in range(n):
        # Find the best (f, l) pair to minimize M[f, l] among unassigned
        # Create a mask for unassigned
        mask_f = np.array([f not in [current_perm[loc] for loc in range(n) if current_perm[loc] != -1] for f in range(n)])
        mask_l = np.array([loc not in [loc for loc in range(n) if current_perm[loc] != -1] for loc in range(n)])
        
        # Actually, simpler to keep track of unassigned lists
        if not unassigned_facilities:
            break
            
        # Evaluate all pairs
        best_f = None
        best_l = None
        best_cost = np.inf
        
        # Vectorized evaluation of M[unassigned_facilities, :][:, unassigned_locations]
        # But M is updated incrementally.
        
        # For n=100, a simple loop is fast enough (100*100 = 10k ops per step, 100 steps = 1M ops)
        # But let's use numpy for the min.
        
        # Create a copy of M masked by unassigned
        M_masked = M.copy()
        for f in unassigned_facilities:
            for l in unassigned_locations:
                pass # We need to find min over these.
        
        # More efficient:
        # Flatten the submatrix
        # But indices are sparse.
        
        # Let's just loop, it's O(n^2) per step, total O(n^3). 100^3 = 1e6, very fast in Python if optimized? 
        # No, Python loops are slow. 1e6 iterations is ~0.1s. 100 steps * 100*100 = 1e6. OK.
        
        # Better: use numpy indexing
        f_indices = np.array(unassigned_facilities)
        l_indices = np.array(unassigned_locations)
        
        # Get the submatrix
        sub_M = M[np.ix_(f_indices, l_indices)]
        
        # Find min
        min_idx = np.argmin(sub_M)
        min_f_idx = min_idx // len(l_indices)
        min_l_idx = min_idx % len(l_indices)
        
        best_f = f_indices[min_f_idx]
        best_l = l_indices[min_l_idx]
        best_cost = sub_M[min_f_idx, min_l_idx]
        
        # Place best_f at best_l
        current_perm[best_l] = best_f
        
        # Remove from unassigned
        unassigned_facilities.remove(best_f)
        unassigned_locations.remove(best_l)
        
        # Update M for remaining pairs
        # M[f, l] += a[f, best_f]*b[l, best_l] + a[best_f, f]*b[best_l, l]
        if unassigned_facilities and unassigned_locations:
            f_rem = np.array(unassigned_facilities)
            l_rem = np.array(unassigned_locations)
            
            # Term 1: a[f, best_f] * b[l, best_l]
            # a[f_rem, best_f] is a vector of size len(f_rem)
            # b[l_rem, best_l] is a vector of size len(l_rem)
            # Outer product
            t1 = np.outer(a[f_rem, best_f], b[l_rem, best_l])
            
            # Term 2: a[best_f, f] * b[best_l, l]
            t2 = np.outer(a[best_f, f_rem], b[best_l, l_rem])
            
            M[np.ix_(f_rem, l_rem)] += (t1 + t2)
            
    # Invert current_perm to get p where p[facility] = location
    p = np.empty(n, dtype=np.int64)
    for loc in range(n):
        fac = current_perm[loc]
        p[fac] = loc
        
    return p

def _local_search(a, b, p, time_limit):
    """First-improvement local search with random restart of order."""
    n = len(p)
    current_cost = _qap_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    
    start_time = time.time()
    
    # Precompute indices
    indices = np.arange(n)
    
    while time.time() - start_time < time_limit:
        improved = False
        
        # Shuffle indices to ensure we don't always start from (0,1)
        np.random.shuffle(indices)
        
        # We can vectorize the delta calculation for all pairs involving i?
        # For a fixed i, delta(i, j) for all j can be computed?
        # Let's stick to pairwise but check time frequently.
        
        for i_idx in range(n):
            if time.time() - start_time > time_limit:
                break
            i = indices[i_idx]
            
            # Compute deltas for all j > i (or all j != i)
            # To be safe, just loop j
            for j_idx in range(i_idx + 1, n):
                j = indices[j_idx]
                
                delta = _swap_delta(a, b, p, i, j)
                if delta < 0:
                    p[i], p[j] = p[j], p[i]
                    current_cost += delta
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_p = p.copy()
                    improved = True
                    break # Restart the outer loop on improvement (First Improvement)
            if improved:
                break
        
        if not improved:
            break
            
    return best_p, best_cost

def solve(a, b):
    n = len(a)
    deadline = 115.0  # Safety margin before 120s limit
    start = time.time()
    
    np.random.seed(42)
    
    best_p = None
    best_cost = np.inf
    
    # Phase 1: Greedy Pairwise Insertion
    p_init = _greedy_pairwise_insertion(a, b)
    cost_init = _qap_cost(a, b, p_init)
    best_p = p_init.copy()
    best_cost = cost_init
    
    # Phase 2: Local Search Polish
    # Allocate most of the time to LS
    ls_time = deadline - (time.time() - start) - 5.0
    if ls_time > 5.0:
        p_ls, cost_ls = _local_search(a, b, best_p, ls_time)
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()
            
    # Phase 3: Multi-restart with perturbation if time remains
    # Check time
    time_remaining = deadline - (time.time() - start)
    if time_remaining > 10.0:
        # Do a few restarts
        num_restarts = int(time_remaining / 10.0)
        for r in range(num_restarts):
            if time.time() - start > deadline - 2.0:
                break
                
            # Perturb best solution
            p_current = best_p.copy()
            num_swaps = 5 + np.random.randint(5)
            for _ in range(num_swaps):
                i = np.random.randint(n)
                j = np.random.randint(n)
                if i != j:
                    p_current[i], p_current[j] = p_current[j], p_current[i]
                    
            # Quick LS
            t_left = deadline - (time.time() - start) - 2.0
            if t_left > 2.0:
                p_r, cost_r = _local_search(a, b, p_current, t_left)
                if cost_r < best_cost:
                    best_cost = cost_r
                    best_p = p_r.copy()
                    
    return best_p.tolist()

# EVOLVE-BLOCK-END
