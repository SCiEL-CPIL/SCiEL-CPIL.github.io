"""
Tabu Search Solver for QAPLIB wil100 using Vectorized Batch Move Evaluation.

Strategy:
1. Initialization: Use a spectral-like initialization based on row sums and
   a simple greedy matching to get a decent starting point.
2. Optimization: Best-Improvement Tabu Search.
   - Neighborhood: All pairs (i, j) for swapping.
   - Move Selection: Evaluate ALL n*(n-1)/2 swaps in parallel using vectorized NumPy operations.
     This is feasible because n=100, so ~5000 evaluations per step.
     Each evaluation computes the exact delta in O(1) vectorized ops if we precompute
     certain row/column interactions, or O(n) per swap if done naively.
     However, for n=100, doing 5000 swaps * O(n) = 500k ops per step is fine in Python/NumPy if vectorized.
     Actually, let's compute the delta for all pairs (i,j) at once.
     
     Let p be the current permutation.
     We want delta(i, j) for all i < j.
     
     The cost is C(p) = sum_{u,v} A[u,v] B[p[u], p[v]].
     
     When swapping p[i] and p[j], the change is:
     delta = (A[i,j] - A[j,i]) * (B[pj, pi] - B[pi, pj])
           + A[i,i] * (B[pj, pj] - B[pi, pi]) + A[j,j] * (B[pi, pi] - B[pj, pj])
           + sum_{k != i,j} (A[i,k] - A[j,k]) * (B[pj, p[k]] - B[pi, p[k]])
           + sum_{k != i,j} (A[k,i] - A[k,j]) * (B[p[k], pj] - B[p[k], pi])
     
     This is hard to fully vectorize for ALL pairs simultaneously without huge memory usage (100x100x100).
     Instead, we will use a hybrid:
     1. Use a fast "first improvement" or "best improvement" via sampling or limited search?
     No, the prompt asks for a structural change.
     
     Alternative Structural Change: **Simulated Annealing with a smarter cooling schedule and a 
     "restart" mechanism, but using a different neighborhood: 3-opt moves (cyclic shifts) 
     in addition to 2-swaps.**
     
     Actually, let's look at the "Submodular" hint. It's often a red herring for QAP.
     Let's try **Iterated Local Search (ILS)** with a strong local search (Best Improvement 2-swap)
     and a perturbation strategy that uses "Or-opt" style moves (moving a block of k items).
     
     Why ILS?
     - Current program uses SA which gets stuck in local minima with random walks.
     - ILS focuses on escaping local minima via controlled perturbations and re-optimizing.
     - For n=100, a Best-Improvement 2-swap local search is expensive if done naively (O(n^3) per pass).
     - We can optimize the local search by keeping track of "reduced costs" or using a priority queue?
     - Or, we can use a faster local search: "First Improvement" is often faster to converge to a local optimum.
     
     Let's implement **ILS with First-Improvement Local Search** and **Block-Move Perturbations**.
     This is structurally different from SA (which accepts worse moves probabilistically) and pure LS.
"""

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def _qap_cost(a, b, p):
    """Compute the exact integer cost sum_{i,j} a[i,j] * b[p[i], p[j]]."""
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

def _block_move_delta(a, b, p, start, length, new_pos):
    """
    Compute delta for moving a block of 'length' items starting at 'start' to 'new_pos'.
    This is more complex. For simplicity and robustness, we will stick to 2-swaps
    but use a smarter search strategy: Iterated Local Search.
    """
    pass

def _first_improvement_ls(a, b, p, time_limit):
    """
    First-improvement local search using 2-swaps.
    Randomly permutes the order of checking (i,j) pairs to avoid bias.
    """
    n = len(p)
    current_cost = _qap_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    
    start_time = time.time()
    # Pre-shuffle the list of pairs to check to ensure unbiased first-improvement
    # There are n*(n-1)/2 pairs. For n=100, ~5000 pairs.
    # We can generate them on the fly or precompute indices.
    # To save time, we'll just iterate i, j but maybe randomize i?
    # Standard first improvement: scan i from 0..n-1, j from i+1..n-1.
    # If found, apply and restart scan from 0? Or continue?
    # Restarting is more thorough but slower. Continuing is faster.
    # Let's do a full pass, applying improvements as we go, until no more found.
    
    improved = True
    while improved and (time.time() - start_time < time_limit):
        improved = False
        for i in range(n):
            if time.time() - start_time > time_limit:
                break
            for j in range(i + 1, n):
                delta = _swap_delta(a, b, p, i, j)
                if delta < 0:
                    p[i], p[j] = p[j], p[i]
                    current_cost += delta
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_p = p.copy()
                    improved = True
                    # In first-improvement, we can break out of inner loop and restart outer?
                    # Or just continue to next j?
                    # Restarting outer loop ensures we don't miss interactions.
                    # But for speed, let's just break inner and let outer continue?
                    # Actually, breaking inner and restarting outer is standard "First Improvement".
                    break
            if improved:
                # We found an improvement, so we should restart the scan to find the next one
                # in the new neighborhood.
                # But this is expensive.
                # Alternative: Just continue scanning. This is "First Improvement" in a single pass.
                # Let's try a single pass with immediate application, no restart.
                # This is often sufficient for ILS.
                pass
                
    return best_p, best_cost

def _best_improvement_ls(a, b, p, time_limit):
    """
    Best-improvement local search.
    Evaluates all moves, picks the best one, applies it, repeats.
    Much slower per step but fewer steps to local optimum.
    For n=100, 5000 evaluations per step. Each eval is ~O(n) dot products.
    5000 * 100 = 500,000 ops. In Python loop, this is too slow (~0.1-0.5s per step).
    So Best Improvement is likely too slow for many iterations.
    We will stick to First Improvement or a hybrid.
    """
    n = len(p)
    current_cost = _qap_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    
    start_time = time.time()
    
    while time.time() - start_time < time_limit:
        best_delta = 0
        best_i, best_j = -1, -1
        
        # Evaluate all pairs
        # To speed this up, we can vectorize the delta calculation for all j for a fixed i?
        # Or just loop.
        for i in range(n):
            if time.time() - start_time > time_limit:
                break
            for j in range(i + 1, n):
                delta = _swap_delta(a, b, p, i, j)
                if delta < best_delta:
                    best_delta = delta
                    best_i, best_j = i, j
        
        if best_i == -1:
            break
        
        p[best_i], p[best_j] = p[best_j], p[best_i]
        current_cost += best_delta
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
            
    return best_p, best_cost

def _perturb(p, n, strength):
    """
    Perturb the permutation by performing 'strength' random swaps.
    """
    p = p.copy()
    for _ in range(strength):
        i = np.random.randint(n)
        j = np.random.randint(n)
        if i != j:
            p[i], p[j] = p[j], p[i]
    return p

def _hungarian_initial(a, b):
    """
    Initial permutation using Hungarian algorithm on a surrogate.
    """
    n = len(a)
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    
    # Simple surrogate: match high flow to low distance (or similar)
    # Cost[i, j] = -a_row_sums[i] * b_row_sums[j]
    # Normalize
    ars_max = np.max(np.abs(a_row_sums))
    brs_max = np.max(np.abs(b_row_sums))
    if ars_max > 0:
        ars_norm = a_row_sums / ars_max
    else:
        ars_norm = a_row_sums
    if brs_max > 0:
        brs_norm = b_row_sums / brs_max
    else:
        brs_norm = b_row_sums
        
    C = -np.outer(ars_norm, brs_norm)
    
    row_ind, col_ind = linear_sum_assignment(C)
    p = np.empty(n, dtype=np.int64)
    p[row_ind] = col_ind
    return p

def solve(a, b):
    n = len(a)
    deadline = 115.0  # Safety margin
    start = time.time()
    
    np.random.seed(42)
    
    # 1. Initial Solution
    p = _hungarian_initial(a, b)
    current_cost = _qap_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    
    # 2. Iterated Local Search (ILS)
    # Loop:
    #   a. Local Search (First Improvement)
    #   b. If improved best, update best
    #   c. Perturb
    #   d. Check time
    
    # Parameters for ILS
    # Strength of perturbation: start small, increase if stuck?
    # For simplicity, fixed strength or adaptive.
    perturb_strength = 5
    
    while time.time() - start < deadline:
        # Local Search Phase
        # Use First Improvement LS with a time limit for this phase
        # We don't want to spend all time in one LS.
        ls_time_limit = min(1.0, deadline - (time.time() - start) - 5.0)
        if ls_time_limit <= 0:
            break
            
        p_ls, cost_ls = _first_improvement_ls(a, b, p, ls_time_limit)
        
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()
        
        p = p_ls.copy()
        
        # Perturbation Phase
        # If we just improved, perturb less? Or more to explore?
        # Standard ILS: perturb the current solution.
        # If stuck (no improvement in LS), perturb more.
        # Let's keep it simple: fixed perturb strength.
        p = _perturb(p, n, perturb_strength)
        
        # Optional: Accept worse solutions in ILS? 
        # Basic ILS accepts the result of LS regardless of quality, as long as it's not worse than best?
        # Actually, standard ILS: 
        # 1. Optimize x0
        # 2. Perturb x0 -> x1
        # 3. Optimize x1 -> x2
        # 4. If x2 < x0, accept. Else, maybe accept with probability?
        # Let's use a simple acceptance: always accept the perturbed+optimized solution as the new current,
        # but keep track of the global best.
        
    # Final Polish: Best Improvement LS on the best solution found
    time_remaining = deadline - (time.time() - start)
    if time_remaining > 2:
        p_final, cost_final = _best_improvement_ls(a, b, best_p.copy(), time_remaining - 1)
        if cost_final < best_cost:
            best_cost = cost_final
            best_p = p_final.copy()
            
    return best_p.tolist()

# EVOLVE-BLOCK-END
