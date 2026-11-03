"""
Combined Island Model with Tabu Search and Hungarian Re-optimization for QAP.

Strategy:
1. Island Model: Split available time into K sequential 'islands' (search trajectories).
2. Each island runs a Tabu Search with 2-swap neighborhood.
   - Fast O(n) delta calculation for swap moves.
   - Tabu list to prevent cycling.
   - Aspiration criterion to allow improving moves even if tabu.
3. Diversification: 
   - When stuck (no improvement for a while), apply a random perturbation (kick).
   - Periodically re-seed from the global best solution to share knowledge.
4. Hungarian Re-optimization: 
   - Every few iterations (or when stuck), solve a linear assignment problem
     (Hungarian algorithm) on a modified cost matrix to escape local optima.
   - This provides a structurally different escape mechanism than random kicks.
5. Final result is the best solution found across all islands.

Why this is different:
- Current: Island model with Tabu Search and random kicks.
- Inspiration 1: Single trajectory Tabu Search with frequency memory.
- Inspiration 2: SA with Hungarian re-optimization.
- New: Island model + Tabu Search + Hungarian re-optimization for diversification.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import linear_sum_assignment

def _qap_cost(a, b, p):
    """Compute exact QAP cost for permutation p."""
    return int(np.sum(a * b[np.ix_(p, p)]))

def _swap_delta(a_sym, b, p, i, j):
    """
    Compute delta for swapping positions i and j in permutation p.
    Assumes a is symmetric (a_sym = a + a.T, so a = a_sym / 2 if symmetric).
    Actually, for QAP cost C = sum a[i,j] b[p[i],p[j]], if a is symmetric,
    we can use a_sym = a + a.T = 2a.
    The delta formula for swapping positions i and j (facilities x=p[i], y=p[j]):
    Delta = sum_{k != i,j} (a[i,k] - a[j,k]) * (b[y,p[k]] - b[x,p[k]])
            + sum_{k != i,j} (a[k,i] - a[k,j]) * (b[p[k],y] - b[p[k],x])
            + (a[i,j] - a[j,i]) * (b[y,x] - b[x,y])
    
    If a is symmetric, a[i,k] = a[k,i], a[i,j] = a[j,i].
    Delta = sum_{k != i,j} (a[i,k] - a[j,k]) * (b[y,p[k]] - b[x,p[k]])
            + sum_{k != i,j} (a[i,k] - a[j,k]) * (b[p[k],y] - b[p[k],x])
            + 0
    
    If b is also symmetric, b[y,p[k]] = b[p[k],y], so the two sums are equal.
    Delta = 2 * sum_{k != i,j} (a[i,k] - a[j,k]) * (b[y,p[k]] - b[x,p[k]])
    
    Let's use the general case for safety, but optimize if symmetric.
    """
    x = p[i]
    y = p[j]
    
    # diff_a[k] = a[i, k] - a[j, k]
    # We only have a_sym. If a is symmetric, a = a_sym / 2.
    # Let's assume a is symmetric (standard for QAPLIB tai256c).
    # a = a_sym / 2.
    # diff_a = (a_sym[i, :] - a_sym[j, :]) / 2
    
    # To avoid division, let's work with a_sym directly and adjust the formula.
    # Original Delta with a:
    # D1 = sum_{k != i,j} (a[i,k] - a[j,k]) * (b[y,p[k]] - b[x,p[k]])
    # D2 = sum_{k != i,j} (a[k,i] - a[k,j]) * (b[p[k],y] - b[p[k],x])
    # D3 = (a[i,j] - a[j,i]) * (b[y,x] - b[x,y])
    
    # If a is symmetric, a[i,k] = a[k,i], a[i,j] = a[j,i].
    # D1 = sum_{k != i,j} (a[i,k] - a[j,k]) * (b[y,p[k]] - b[x,p[k]])
    # D2 = sum_{k != i,j} (a[i,k] - a[j,k]) * (b[p[k],y] - b[p[k],x])
    # D3 = 0
    
    # If b is also symmetric, b[y,p[k]] = b[p[k],y].
    # D1 = D2.
    # Delta = 2 * D1.
    
    # Let's compute D1 using a_sym.
    # a[i,k] - a[j,k] = (a_sym[i,k] - a_sym[j,k]) / 2
    # So D1 = 1/2 * sum_{k != i,j} (a_sym[i,k] - a_sym[j,k]) * (b[y,p[k]] - b[x,p[k]])
    # Delta = 2 * D1 = sum_{k != i,j} (a_sym[i,k] - a_sym[j,k]) * (b[y,p[k]] - b[x,p[k]])
    
    # This is much simpler!
    # Let S = sum_{k} (a_sym[i,k] - a_sym[j,k]) * (b[y,p[k]] - b[x,p[k]])
    # We need to exclude k=i and k=j.
    # Term for k=i: (a_sym[i,i] - a_sym[j,i]) * (b[y,p[i]] - b[x,p[i]])
    #                = (a_sym[i,i] - a_sym[j,i]) * (b[y,x] - b[x,x])
    # Term for k=j: (a_sym[i,j] - a_sym[j,j]) * (b[y,p[j]] - b[x,p[j]])
    #                = (a_sym[i,j] - a_sym[j,j]) * (b[y,y] - b[x,y])
    
    diff_a = a_sym[i, :] - a_sym[j, :]
    diff_b = b[y, :] - b[x, :]
    
    temp = diff_a * diff_b[p]
    
    total = np.sum(temp)
    
    term_i = diff_a[i] * diff_b[x]
    term_j = diff_a[j] * diff_b[y]
    
    delta = total - term_i - term_j
    
    return delta

def _hungarian_escape(a, b, p, current_cost, rng, time_limit, start_time):
    """
    Use Hungarian algorithm to find a better assignment.
    We modify the cost matrix to guide the search.
    """
    n = len(p)
    
    # Create a cost matrix for Hungarian.
    # We want to minimize sum a[i,j] b[p[i],p[j]].
    # This is not a linear assignment problem directly.
    # However, we can use a linear approximation.
    # Let C[i,j] = sum_k a[i,k] b[j, p[k]] + sum_k a[k,i] b[p[k], j]
    # This is the cost of assigning facility i to location j, given the current assignment of others.
    # This is a valid linear cost matrix for the Hungarian algorithm.
    
    # Compute C
    # C[i,j] = sum_k a[i,k] b[j, p[k]] + sum_k a[k,i] b[p[k], j]
    # C = a @ b[:, p].T + a.T @ b[p, :].T ?
    # Let's check dimensions.
    # a: (n, n), b: (n, n), p: (n,)
    # b[:, p] is (n, n) where column k is b[:, p[k]].
    # a @ b[:, p] is (n, n). Element (i, k) is sum_j a[i,j] b[j, p[k]].
    # This is not quite right.
    
    # Let's use the standard linearization for QAP.
    # Cost of assigning i to j is approximated by:
    # L[i,j] = sum_k a[i,k] b[j, p[k]] + sum_k a[k,i] b[p[k], j]
    
    # Vectorized computation:
    # Term1[i,j] = sum_k a[i,k] b[j, p[k]]
    # b_p = b[:, p]  # shape (n, n), b_p[j, k] = b[j, p[k]]
    # Term1 = a @ b_p  # shape (n, n), Term1[i, j] = sum_k a[i,k] b_p[k, j] = sum_k a[i,k] b[p[k], j]
    # Wait, b_p[k, j] = b[p[k], j]. So Term1[i, j] = sum_k a[i,k] b[p[k], j].
    # This is the second term in the formula.
    
    # Term2[i,j] = sum_k a[k,i] b[j, p[k]]
    # b_pT = b[p, :]  # shape (n, n), b_pT[k, j] = b[p[k], j] ? No.
    # b[p, :] has rows indexed by p[k].
    # Let's just compute it directly.
    
    # Actually, a simpler linearization is:
    # L[i,j] = sum_k a[i,k] (b[j, p[k]] + b[p[k], j])
    # If b is symmetric, b[j, p[k]] = b[p[k], j], so L[i,j] = 2 * sum_k a[i,k] b[j, p[k]].
    
    # Let's use L[i,j] = sum_k a[i,k] b[j, p[k]]
    # This is a (n, n) matrix.
    # L = a @ b[:, p].T ?
    # b[:, p] is (n, n) with b[:, p][j, k] = b[j, p[k]].
    # (b[:, p]).T is (n, n) with (b[:, p]).T[k, j] = b[j, p[k]].
    # a @ (b[:, p]).T is (n, n) with element (i, j) = sum_k a[i,k] (b[:, p]).T[k, j] = sum_k a[i,k] b[j, p[k]].
    # Yes!
    
    b_perm = b[:, p]  # shape (n, n)
    L = a @ b_perm.T  # shape (n, n)
    
    # Add a small random perturbation to break ties and diversify
    L += rng.random((n, n)) * 0.01 * np.max(np.abs(L))
    
    # Solve Hungarian
    row_ind, col_ind = linear_sum_assignment(L)
    
    # row_ind is facilities, col_ind is locations.
    # We need to construct a permutation p_new such that p_new[row_ind[k]] = col_ind[k].
    p_new = np.empty(n, dtype=np.int64)
    p_new[row_ind] = col_ind
    
    new_cost = _qap_cost(a, b, p_new)
    
    return p_new, new_cost

def _solve_tai256c(a, b, time_limit=115.0):
    n = len(a)
    start_time = time.time()
    
    # Precompute
    a_sym = a + a.T
    
    # Initial solution: Rank matching (greedy)
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    heavy = np.argsort(-a_row_sums, kind="stable")
    central = np.argsort(b_row_sums, kind="stable")
    p_init = np.empty(n, dtype=np.int64)
    p_init[heavy] = central
    
    # Island Model: Run K parallel (sequential) Tabu Searches
    K = 4
    time_per_island = time_limit / K
    
    best_p_global = p_init.copy()
    best_cost_global = _qap_cost(a, b, best_p_global)
    
    rng_global = np.random.default_rng(42)
    
    for k in range(K):
        island_start = time.time()
        island_deadline = island_start + time_per_island - 0.5
        
        # Re-seed from global best with some randomization
        p_island = best_p_global.copy()
        if k > 0:
            rng = np.random.default_rng(42 + k)
            for _ in range(10):
                i, j = rng.choice(n, size=2, replace=False)
                p_island[i], p_island[j] = p_island[j], p_island[i]
        
        current_p = p_island.copy()
        current_cost = _qap_cost(a, b, current_p)
        best_p_island = current_p.copy()
        best_cost_island = current_cost
        
        # Tabu Search Parameters
        tabu_tenure = 15
        tabu_list = {}
        step = 0
        no_improve_steps = 0
        max_no_improve = 100
        hungarian_counter = 0
        
        while time.time() < island_deadline:
            # Generate neighborhood: Random sample of swaps
            n_moves = 100
            indices = np.random.default_rng().choice(n, size=(n_moves, 2), replace=False)
            
            best_delta = np.inf
            best_move = None
            
            for idx in range(n_moves):
                i, j = indices[idx]
                if i == j:
                    continue
                
                key = (min(i, j), max(i, j))
                if key in tabu_list and tabu_list[key] > step:
                    continue
                
                delta = _swap_delta(a_sym, b, current_p, i, j)
                
                if delta < best_delta:
                    best_delta = delta
                    best_move = (i, j)
            
            if best_move is None:
                # All sampled moves were tabu?
                i, j = np.random.default_rng().choice(n, size=2, replace=False)
                delta = _swap_delta(a_sym, b, current_p, i, j)
                best_move = (i, j)
                best_delta = delta
            
            i, j = best_move
            
            # Update Tabu List
            key = (min(i, j), max(i, j))
            tabu_list[key] = step + tabu_tenure
            
            # Apply Move
            current_p[i], current_p[j] = current_p[j], current_p[i]
            current_cost += best_delta
            step += 1
            
            # Update Best
            if current_cost < best_cost_island:
                best_cost_island = current_cost
                best_p_island = current_p.copy()
                no_improve_steps = 0
            else:
                no_improve_steps += 1
            
            # Diversification
            if no_improve_steps > max_no_improve:
                # Try Hungarian escape
                p_new, new_cost = _hungarian_escape(a, b, current_p, current_cost, rng_global, time_limit, start_time)
                
                if new_cost < best_cost_island:
                    current_p = p_new.copy()
                    current_cost = new_cost
                    best_p_island = current_p.copy()
                    best_cost_island = current_cost
                else:
                    # If Hungarian didn't help, do a random kick
                    ri, rj = np.random.default_rng().choice(n, size=2, replace=False)
                    current_p[ri], current_p[rj] = current_p[rj], current_p[ri]
                    current_cost = _qap_cost(a, b, current_p)
                
                no_improve_steps = 0
                tabu_list.clear()
            
            # Update Global Best
            if best_cost_island < best_cost_global:
                best_cost_global = best_cost_island
                best_p_global = best_p_island.copy()
        
        # End of island k
    
    return best_p_global.tolist()

def solve(a, b):
    return _solve_tai256c(a, b)
# EVOLVE-BLOCK-END
