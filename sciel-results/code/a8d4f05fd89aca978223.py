"""
Tabu Search Solver for QAPLIB wil100.

Strategy:
1. Initialization: Use spectral embedding (top eigenvectors of distance matrix)
   to project facilities and locations onto a 1D line. Sort by projection to
   create an initial permutation that respects the geometric structure.
2. Optimization: Tabu Search.
   - Neighborhood: All pairs (i, j) for swapping.
   - Move Selection: Best non-tabu move (or best move if it satisfies aspiration).
   - Tabu List: Store (i, j) pairs that were recently swapped. 
   - Aspiration: Accept a tabu move if it improves the best known solution.
   - Diversification: If stuck, occasionally perform a random swap to jump out.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.linalg import eigh

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

def _spectral_init(a, b):
    """
    Initialize permutation using spectral embedding.
    Projects facilities and locations onto the principal axis of the distance/flow space.
    """
    n = len(a)
    
    # Use the distance matrix B for location ordering.
    # For flow matrix A, we can use row sums or spectral embedding of A as well.
    # A common heuristic is to align the "centrality" or "spread" of flows with distances.
    
    # Get top eigenvector of B (symmetric part)
    # B is usually symmetric in QAP instances, but let's ensure symmetry for eigh
    B_sym = (b + b.T) / 2.0
    try:
        # We want the eigenvector corresponding to the smallest eigenvalue? 
        # Actually, for ordering, we often use the Fiedler vector (2nd smallest) or 
        # the principal component. Let's use the top eigenvector of the Laplacian 
        # or just the top eigenvector of the matrix itself if it represents similarity.
        # For distance matrices, larger values are "far". 
        # Let's project onto the first principal component of the coordinates implied by B.
        # Since we don't have coordinates, we use eigenvectors of B.
        
        # For a distance matrix, the top eigenvector often captures the main axis of separation.
        evals, evecs = eigh(B_sym)
        # Take the last eigenvector (largest eigenvalue) or the one with largest variance?
        # Usually, the Fiedler vector (2nd smallest) is good for clustering, but for ordering,
        # the principal component (largest eigenvalue) might be better if the matrix is not a Laplacian.
        # Let's try the eigenvector with the largest magnitude of spread.
        
        # Heuristic: Use the eigenvector corresponding to the largest eigenvalue of B_sym
        # or the one that maximizes the sum of squared distances.
        # Let's just take the last column (largest eigenvalue)
        v_b = evecs[:, -1]
        
        # For A, use row sums as a proxy for "importance" or "flow volume"
        # Facilities with high flow should be placed in locations that are "central" or "connected"?
        # Actually, in QAP, we want high flows to have low distances.
        # So we want to correlate high a_row_sum[i] with low b_row_sum[j]? 
        # Or high flow with central locations?
        
        # Let's refine: 
        # Sort locations by their projection on v_b.
        # Sort facilities by their row sums in A.
        # Then match them? 
        # This is a greedy assignment based on 1D order.
        
        loc_order = np.argsort(v_b)
        fac_order = np.argsort(a.sum(axis=1))
        
        # Create permutation p such that p[fac_order[k]] = loc_order[k]
        p = np.empty(n, dtype=np.int64)
        for k in range(n):
            p[fac_order[k]] = loc_order[k]
            
        return p
        
    except Exception:
        return np.arange(n)

def _tabu_search(a, b, p, time_limit, tabu_tenure=10, diversify_prob=0.05):
    """
    Tabu Search with vectorized delta evaluation.
    """
    n = len(p)
    current_cost = _qap_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    
    # Tabu list: dictionary mapping (i, j) to the iteration when it becomes non-tabu
    # We use a set of forbidden moves.
    # To be efficient, we can use a list of (i, j, expiry).
    # But checking if (i,j) is tabu for all pairs is O(n^2 * T) which is expensive.
    # Alternative: Maintain a 2D boolean matrix `is_tabu[i, j]` and a queue of expirations.
    
    is_tabu = np.zeros((n, n), dtype=bool)
    tabu_queue = []  # List of (iteration, i, j)
    current_iter = 0
    
    start_time = time.time()
    
    # Parameters
    max_iter_no_improve = 500
    iter_no_improve = 0
    
    while time.time() - start_time < time_limit:
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            break
            
        # Expire old tabu entries
        while tabu_queue and tabu_queue[0][0] <= current_iter:
            _, i, j = tabu_queue.pop(0)
            is_tabu[i, j] = False
            is_tabu[j, i] = False # Symmetric
        
        # Find best move
        best_delta = np.inf
        best_i = -1
        best_j = -1
        is_aspiration = False
        
        # Vectorized delta calculation for all pairs?
        # Computing delta for all (i, j) is O(n^3) if done naively.
        # We can compute deltas for all j for a fixed i in O(n).
        # But we need to check tabu status.
        
        # Heuristic: Sample a subset of moves if n is large? 
        # For n=100, n^2 = 10,000 moves. 
        # Computing delta for 10,000 moves per iteration might be too slow if each takes ~1ms (10s/iter).
        # We need faster delta or fewer candidates.
        
        # Let's try a hybrid: Evaluate a random subset of 100-200 moves per iteration.
        # This is a stochastic Tabu Search.
        
        num_candidates = 200
        candidates = []
        for _ in range(num_candidates):
            i = np.random.randint(n)
            j = np.random.randint(n)
            if i == j:
                continue
            # Ensure we don't pick the same pair twice in this iteration
            if (i, j) in candidates or (j, i) in candidates:
                continue
            candidates.append((i, j))
            
        for i, j in candidates:
            if is_tabu[i, j]:
                # Check aspiration
                # We can't know the new cost without computing delta.
                # If we compute delta, we can check if current_cost + delta < best_cost.
                pass 
            else:
                pass
            
            delta = _swap_delta(a, b, p, i, j)
            
            # Aspiration criterion: Accept if it improves best cost
            if is_tabu[i, j]:
                if current_cost + delta < best_cost:
                    is_aspiration = True
                else:
                    continue
            
            if delta < best_delta:
                best_delta = delta
                best_i = i
                best_j = j
                is_aspiration = is_aspiration if is_tabu[i, j] else False
                
        if best_i == -1:
            # No valid move found in candidates (all tabu and no aspiration)
            # This shouldn't happen often with random sampling and short tenure
            # Force a random move
            i = np.random.randint(n)
            j = np.random.randint(n)
            while i == j:
                j = np.random.randint(n)
            best_i, best_j = i, j
            best_delta = _swap_delta(a, b, p, i, j)
            is_aspiration = True # Allow it
            
        # Perform the move
        p[best_i], p[best_j] = p[best_j], p[best_i]
        current_cost += best_delta
        
        # Update tabu list
        is_tabu[best_i, best_j] = True
        is_tabu[best_j, best_i] = True
        tabu_queue.append((current_iter + tabu_tenure, best_i, best_j))
        
        # Update best
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
            iter_no_improve = 0
        else:
            iter_no_improve += 1
            
        # Diversification: If stuck, restart or random jump
        if iter_no_improve > max_iter_no_improve:
            # Random restart from best solution with perturbation
            p = best_p.copy()
            current_cost = best_cost
            # Perturb
            for _ in range(5):
                i = np.random.randint(n)
                j = np.random.randint(n)
                if i != j:
                    p[i], p[j] = p[j], p[i]
            current_cost = _qap_cost(a, b, p)
            iter_no_improve = 0
            # Clear tabu list
            is_tabu = np.zeros((n, n), dtype=bool)
            tabu_queue = []
            
        current_iter += 1
        
    return best_p, best_cost

def solve(a, b):
    n = len(a)
    deadline = 115.0  # Safety margin
    start = time.time()
    
    np.random.seed(42)
    
    best_p = None
    best_cost = np.inf
    
    # 1. Spectral Initialization
    p_init = _spectral_init(a, b)
    cost_init = _qap_cost(a, b, p_init)
    best_p = p_init.copy()
    best_cost = cost_init
    
    # 2. Tabu Search
    time_remaining = deadline - (time.time() - start)
    if time_remaining > 5:
        ts_time = max(0, time_remaining - 2)
        p_ts, cost_ts = _tabu_search(a, b, best_p.copy(), ts_time, tabu_tenure=15, diversify_prob=0.05)
        if cost_ts < best_cost:
            best_cost = cost_ts
            best_p = p_ts.copy()
            
    # 3. Final Polish with Local Search (First Improvement) to clean up
    time_remaining = deadline - (time.time() - start)
    if time_remaining > 2:
        # Quick local search
        p_ls = best_p.copy()
        c_ls = best_cost
        improved = True
        while improved and (time.time() - start) < deadline - 1:
            improved = False
            for i in range(n):
                for j in range(i + 1, n):
                    d = _swap_delta(a, b, p_ls, i, j)
                    if d < 0:
                        p_ls[i], p_ls[j] = p_ls[j], p_ls[i]
                        c_ls += d
                        if c_ls < best_cost:
                            best_cost = c_ls
                            best_p = p_ls.copy()
                        improved = True
                        break
                if improved:
                    break
                    
    return best_p.tolist()
# EVOLVE-BLOCK-END
