"""QAP solver for tho150 using local search with efficient swap evaluation."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def solve(a, b):
    """Solve QAP: assign facilities to locations to minimize sum a[i,j]*b[p[i],p[j]].
    
    Args:
        a: flow matrix (n x n) int64
        b: distance matrix (n x n) int64
    Returns:
        p: permutation list where facility i goes to location p[i]
    """
    n = len(a)
    deadline = time.time() + 115  # Safety margin before 150s kill
    
    # Precompute row sums for initial heuristics
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    
    # Convert to float for faster arithmetic in delta computation
    a_f = a.astype(np.float64)
    b_f = b.astype(np.float64)
    
    def compute_cost(p):
        """Compute exact cost of permutation p."""
        M = b[np.ix_(p, p)]
        return int(np.sum(a * M))
    
    def get_deltas_for_i(p, i):
        """Compute cost deltas for swapping position i with all other positions j.
        
        Returns array of length n where deltas[j] = cost change when swapping p[i] and p[j].
        """
        pi = p[i]
        # For each j != i, delta = sum_k a[i,k]*(b[pj,pk]-b[pi,pk]) + a[k,i]*(b[pk,pj]-b[pk,pi])
        #                        + sum_k a[j,k]*(b[pi,pk]-b[pj,pk]) + a[k,j]*(b[pk,pi]-b[pk,pj])
        #                        + diagonal terms + ij terms
        
        # Vectorized: for all j at once
        # b[pi, p[j]] for all j: b_f[pi, p]
        # b[p[j], p[k]] is harder - need to think differently
        
        # Alternative: compute contribution of position i to total cost
        # C_i = sum_j a[i,j]*b[pi,p[j]] + sum_k a[k,i]*b[p[k],pi]
        # After swap with j: 
        # C_i' = sum_j a[i,j]*b[pj,p[j]] + sum_k a[k,i]*b[p[k],pj]
        # C_j' = sum_j a[j,j]*b[pi,p[j]] + sum_k a[k,j]*b[p[k],pi]
        # Original: C_i + C_j + a[i,j]*b[pi,pj] + a[j,i]*b[pj,pi]
        # New: C_i' + C_j' + a[i,j]*b[pj,pi] + a[j,i]*b[pi,pj]
        
        # Let's compute using precomputed vectors
        # For position i: col_a_i = a[:, i] (column i of a), row_a_i = a[i, :]
        # cost contribution of i: row_a_i @ b[pi, p] + col_a_i @ b[p, pi]
        # But b[pi, p] = b_f[pi, p] and b[p, pi] = b_f[p, pi]
        
        row_ai = a_f[i, :]
        col_ai = a_f[:, i]
        b_pi_p = b_f[pi, p]  # b[pi, p[j]] for all j
        b_p_pi = b_f[p, pi]  # b[p[j], pi] for all j
        
        # Original contribution of i: sum_j a[i,j]*b[pi,p[j]] + sum_k a[k,i]*b[p[k],pi]
        orig_i = np.dot(row_ai, b_pi_p) + np.dot(col_ai, b_p_pi)
        
        # For each j, after swap:
        # i gets location p[j], j gets location pi
        # New contribution of i: sum_j a[i,j]*b[pj,p[j]] + sum_k a[k,i]*b[p[k],pj]
        # = np.dot(row_ai, b_f[pj, p]) + np.dot(col_ai, b_f[p, pj])
        # But pj depends on j, so we need per-j computation
        
        # More efficient: 
        # delta(j) = [sum_k a[i,k]*(b[pj,pk] - b[pi,pk]) + sum_k a[k,i]*(b[pk,pj] - b[pk,pi])]
        #          + [sum_k a[j,k]*(b[pi,pk] - b[pj,pk]) + sum_k a[k,j]*(b[pk,pi] - b[pk,pj])]
        #          + diagonal + ij terms
        
        # Let's define:
        # For all k: diff_k(j) = b[pj,pk] - b[pi,pk] for the i terms
        # This is b_f[pj, p] - b_f[pi, p] for the row part
        # and b_f[p, pj] - b_f[p, pi] for the col part
        
        # So delta(j) = row_ai @ (b_f[pj,p] - b_f[pi,p]) + col_ai @ (b_f[p,pj] - b_f[p,pi])
        #             + row_aj @ (b_f[pi,p] - b_f[pj,p]) + col_aj @ (b_f[p,pi] - b_f[p,pj])
        #             + a[i,i]*(b[pj,pj]-b[pi,pi]) + a[j,j]*(b[pi,pi]-b[pj,pj])
        #             + a[i,j]*(b[pj,pi]-b[pi,pj]) + a[j,i]*(b[pi,pj]-b[pj,pi])
        
        # Since b is symmetric: b[pj,pi]=b[pi,pj], so ij terms cancel.
        # Diagonal: (a[i,i]-a[j,j])*(b[pj,pj]-b[pi,pi])
        
        # Let me compute for all j at once:
        # b_f[p, p] gives us the full matrix. We need b_f[pj, p] for each j.
        # b_f[np.ix_(p, p)] gives us M where M[j,k] = b[p[j],p[k]]
        M = b_f[np.ix_(p, p)]
        # M[j, :] = b_f[p[j], p]
        # M[:, j] = b_f[p, p[j]]
        
        # For position i:
        # row_ai @ (M[j,:] - b_f[pi, p]) + col_ai @ (M[:,j] - b_f[p, pi])
        # For position j:
        # row_aj @ (b_f[pi, p] - M[j,:]) + col_aj @ (b_f[p, pi] - M[:,j])
        
        # = row_ai @ (M[j,:] - b_pi_p) + col_ai @ (M[:,j] - b_p_pi)
        #   - row_aj @ (M[j,:] - b_pi_p) - col_aj @ (M[:,j] - b_p_pi)
        # = (row_ai - row_aj) @ (M[j,:] - b_pi_p) + (col_ai - col_aj) @ (M[:,j] - b_p_pi)
        
        # This is still O(n) per j. Let me just do it directly for each j.
        # But we can vectorize the M[j,:] part.
        
        # Actually, let's just compute deltas for all j in a loop but with numpy operations
        deltas = np.zeros(n, dtype=np.float64)
        for j in range(n):
            if j == i:
                continue
            pj = p[j]
            # Row and col of j
            row_aj = a_f[j, :]
            col_aj = a_f[:, j]
            
            # Delta from i's perspective
            d_i = np.dot(row_ai, M[j, :] - b_pi_p) + np.dot(col_ai, M[:, j] - b_p_pi)
            # Delta from j's perspective  
            d_j = np.dot(row_aj, b_pi_p - M[j, :]) + np.dot(col_aj, b_p_pi - M[:, j])
            
            # Diagonal correction
            diag = (a_f[i, i] - a_f[j, j]) * (b_f[pj, pj] - b_f[pi, pi])
            
            deltas[j] = d_i + d_j + diag
        
        deltas[i] = 0  # No self-swap
        return deltas
    
    def local_search(p, deadline):
        """Steepest descent local search. Returns improved permutation and cost."""
        cost = compute_cost(p)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            best_delta = 0
            best_i, best_j = -1, -1
            
            for i in range(n):
                if time.time() > deadline:
                    break
                deltas = get_deltas_for_i(p, i)
                # Find best j > i to avoid double counting
                if i + 1 < n:
                    sub = deltas[i+1:]
                    best_idx = np.argmin(sub)
                    best_val = sub[best_idx]
                    if best_val < best_delta:
                        best_delta = best_val
                        best_i = i
                        best_j = i + 1 + best_idx
            
            if best_i >= 0:
                p[best_i], p[best_j] = p[best_j], p[best_i]
                cost += int(best_delta)
                improved = True
        
        return p, cost
    
    def make_initial_solution(method, rng):
        """Create an initial permutation using different strategies."""
        if method == 0:
            # Rank matching: heavy flow -> central location
            heavy = np.argsort(-a_row_sums, kind='stable')
            central = np.argsort(b_row_sums, kind='stable')
            p = np.empty(n, dtype=np.int64)
            p[heavy] = central
        elif method == 1:
            # Reverse rank
            heavy = np.argsort(a_row_sums, kind='stable')
            central = np.argsort(-b_row_sums, kind='stable')
            p = np.empty(n, dtype=np.int64)
            p[heavy] = central
        elif method == 2:
            # Random
            p = rng.permutation(n).astype(np.int64)
        elif method == 3:
            # Sort by a_row_sums / b_row_sums ratio
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio = np.where(b_row_sums > 0, a_row_sums / np.maximum(b_row_sums, 1), 0)
            fac_order = np.argsort(-ratio, kind='stable')
            loc_order = np.argsort(b_row_sums, kind='stable')
            p = np.empty(n, dtype=np.int64)
            p[fac_order] = loc_order
        else:
            p = rng.permutation(n).astype(np.int64)
        return p
    
    # Multiple initializations with ILS
    best_p = None
    best_cost = np.inf
    
    rng = np.random.default_rng(42)
    
    # Try several different initial solutions
    n_init = 8
    for init_idx in range(n_init):
        if time.time() > deadline:
            break
        
        method = init_idx % 4
        p = make_initial_solution(method, rng)
        
        # Local search
        p, cost = local_search(p, deadline)
        
        if cost < best_cost:
            best_cost = cost
            best_p = p.copy()
        
        # ILS: perturb and search again if time permits
        if time.time() < deadline - 30:
            for ils_iter in range(5):
                if time.time() > deadline:
                    break
                # Perturb: random swaps
                n_swaps = rng.integers(5, 20)
                p_perturbed = p.copy()
                for _ in range(n_swaps):
                    i, j = rng.integers(0, n, size=2)
                    if i != j:
                        p_perturbed[i], p_perturbed[j] = p_perturbed[j], p_perturbed[i]
                
                p_perturbed, cost_p = local_search(p_perturbed, deadline)
                
                if cost_p < best_cost:
                    best_cost = cost_p
                    best_p = p_perturbed.copy()
                elif cost_p < cost:
                    p = p_perturbed.copy()
                    cost = cost_p
    
    # Final verification
    if best_p is None:
        best_p = make_initial_solution(0, rng)
    
    final_cost = compute_cost(best_p)
    if final_cost < best_cost:
        best_cost = final_cost
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
