"""QAP tai256c solver: fast ILS with vectorized 2-opt, Frank-Wolfe init, and adaptive perturbation."""
import numpy as np
import time
import random
from scipy.optimize import linear_sum_assignment


def solve(a, b):
    n = len(a)
    deadline = time.time() + 140.0  # safety margin under 150s limit

    def compute_cost(p):
        return int(np.sum(a * b[np.ix_(p, p)]))

    def vectorized_local_search(p, cost, max_passes=200):
        """
        Vectorized steepest-descent 2-opt local search.
        For each position u, compute delta for swapping with all v in parallel,
        pick the best improving swap, and apply it.
        """
        p = p.copy()
        improved = True
        passes = 0
        while improved and passes < max_passes:
            improved = False
            passes += 1
            Bpp = b[np.ix_(p, p)]  # shape (n, n), Bpp[i, j] = b[p[i], p[j]]
            for u in range(n):
                if time.time() > deadline - 5:
                    return p, cost
                
                pu = p[u]
                # For swapping u with v:
                # delta = sum_{i,j} a[i,j] * (b[p[i'], p[j']] - b[p[i], p[j]])
                # where only positions u and v change.
                
                # When we swap p[u] and p[v]:
                # - For pairs where neither index is u or v: no change
                # - For pairs involving u or v: changes occur
                
                # Let's compute the delta efficiently:
                # Old contribution involving position u: sum_j a[u,j]*b[pu,p[j]] + sum_i a[i,u]*b[p[i],pu] - a[u,u]*b[pu,pu]
                # New contribution involving position u (now holding pv): sum_j a[u,j]*b[pv,p[j]] + sum_i a[i,u]*b[p[i],pv] - a[u,u]*b[pv,pv]
                # Similarly for position v (now holding pu):
                # Old: sum_j a[v,j]*b[pv,p[j]] + sum_i a[i,v]*b[p[i],pv] - a[v,v]*b[pv,pv]
                # New: sum_j a[v,j]*b[pu,p[j]] + sum_i a[i,v]*b[p[i],pu] - a[v,v]*b[pu,pu]
                
                # delta(u,v) = [new_u + new_v] - [old_u + old_v]
                
                # Let's compute for all v at once:
                # new_u[v] = sum_j a[u,j]*b[pv,p[j]] + sum_i a[i,u]*b[p[i],pv] - a[u,u]*b[pv,pv]
                # = a[u] @ Bpp[v] + a[:,u] @ Bpp[:,v] - a[u,u]*Bpp[v,v]
                
                # old_u[v] = sum_j a[u,j]*b[pu,p[j]] + sum_i a[i,u]*b[p[i],pu] - a[u,u]*b[pu,pu]
                # = a[u] @ Bpp[u] + a[:,u] @ Bpp[:,u] - a[u,u]*Bpp[u,u]
                # Note: old_u is the same for all v (doesn't depend on v)
                
                # new_v[v] = sum_j a[v,j]*b[pu,p[j]] + sum_i a[i,v]*b[p[i],pu] - a[v,v]*b[pu,pu]
                # = a[v] @ Bpp[u] + a[:,v] @ Bpp[:,u] - a[v,v]*Bpp[u,u]
                
                # old_v[v] = sum_j a[v,j]*b[pv,p[j]] + sum_i a[i,v]*b[p[i],pv] - a[v,v]*b[pv,pv]
                # = a[v] @ Bpp[v] + a[:,v] @ Bpp[:,v] - a[v,v]*Bpp[v,v]
                
                # So delta(v) = [a[u]@Bpp[v] + a[:,u]@Bpp[:,v] - a[u,u]*Bpp[v,v]]
                #              + [a[v]@Bpp[u] + a[:,v]@Bpp[:,u] - a[v,v]*Bpp[u,u]]
                #              - [a[u]@Bpp[u] + a[:,u]@Bpp[:,u] - a[u,u]*Bpp[u,u]]
                #              - [a[v]@Bpp[v] + a[:,v]@Bpp[:,v] - a[v,v]*Bpp[v,v]]
                
                # Group terms:
                # = a[u]@Bpp[v] - a[u]@Bpp[u] + a[:,u]@Bpp[:,v] - a[:,u]@Bpp[:,u]
                #   + a[v]@Bpp[u] - a[v]@Bpp[v] + a[:,v]@Bpp[:,u] - a[:,v]@Bpp[:,v]
                #   - a[u,u]*Bpp[v,v] + a[u,u]*Bpp[u,u] - a[v,v]*Bpp[u,u] + a[v,v]*Bpp[v,v]
                
                # Term by term for all v:
                T1 = a[u] @ Bpp.T  # a[u] @ Bpp[v] for all v: shape (n,)
                T2 = a[u] @ Bpp[u]  # scalar
                T3 = a[:, u] @ Bpp  # a[:,u] @ Bpp[:,v] for all v: shape (n,)
                T4 = a[:, u] @ Bpp[:, u]  # scalar
                T5 = a @ Bpp[u]  # a[v] @ Bpp[u] for all v: shape (n,)
                T6 = np.sum(a * Bpp, axis=1)  # a[v] @ Bpp[v] for all v: shape (n,)
                T7 = np.diag(a) * Bpp[u, :]  # a[v,v]*Bpp[u,u]... wait
                # -a[v,v]*Bpp[u,u] for all v: shape (n,)
                T8 = np.diag(a) * Bpp[u, u]  # scalar * ones? No: a[v,v] * Bpp[u,u]
                # Actually: -a[v,v]*Bpp[u,u] = -np.diag(a) * Bpp[u,u]
                T9 = -np.diag(a) * Bpp[u, u]  # shape (n,)
                T10 = -a[u, u] * Bpp[np.arange(n), np.arange(n)]  # -a[u,u]*Bpp[v,v]
                T11 = a[u, u] * Bpp[u, u]  # scalar
                T12 = np.diag(a) * Bpp[np.arange(n), np.arange(n)]  # a[v,v]*Bpp[v,v]
                
                # Wait, let me re-derive more carefully.
                # delta(v) = a[u]@Bpp[v] - a[u]@Bpp[u] 
                #          + a[:,u]@Bpp[:,v] - a[:,u]@Bpp[:,u]
                #          + a[v]@Bpp[u] - a[v]@Bpp[v]
                #          + a[:,v]@Bpp[:,u] - a[:,v]@Bpp[:,v]
                #          - a[u,u]*Bpp[v,v] + a[u,u]*Bpp[u,u] 
                #          - a[v,v]*Bpp[u,u] + a[v,v]*Bpp[v,v]
                
                # For all v:
                part1 = T1 - T2  # a[u]@Bpp[v] - a[u]@Bpp[u]
                part2 = T3 - T4  # a[:,u]@Bpp[:,v] - a[:,u]@Bpp[:,u]
                part3 = T5 - T6  # a[v]@Bpp[u] - a[v]@Bpp[v]
                # a[:,v]@Bpp[:,u] for all v: this is Bpp[:,u] @ a.T = a @ Bpp[:,u]
                part4 = a @ Bpp[:, u]  # shape (n,)
                # a[:,v]@Bpp[:,v] for all v: this is sum_i a[i,v]*Bpp[i,v] = np.sum(a * Bpp, axis=0)
                part4_minus = np.sum(a * Bpp, axis=0)  # shape (n,)
                part4 = part4 - part4_minus
                
                part5 = -a[u, u] * Bpp[np.arange(n), np.arange(n)]  # -a[u,u]*Bpp[v,v]
                part6 = a[u, u] * Bpp[u, u]  # scalar
                part7 = -np.diag(a) * Bpp[u, u]  # -a[v,v]*Bpp[u,u]
                part8 = np.diag(a) * Bpp[np.arange(n), np.arange(n)]  # a[v,v]*Bpp[v,v]
                
                deltas = part1 + part2 + part3 + part4 + part5 + part6 + part7 + part8
                deltas[u] = 0  # Can't swap with self
                
                v_best = np.argmin(deltas)
                delta_best = deltas[v_best]
                
                if delta_best < 0 and v_best != u:
                    p[u], p[v_best] = p[v_best], p[u]
                    cost += int(delta_best)
                    improved = True
                    # Recompute Bpp after swap
                    break  # Restart the u-loop with new Bpp
        
        return p, cost

    def hungarian_init():
        """Initialize using Hungarian on centrality correlation matrix."""
        # Facility centrality: row sums of a
        # Location centrality: column sums of b
        a_centrality = a.sum(axis=1)
        b_centrality = b.sum(axis=1)
        # Cost matrix: minimize sum a_centrality[i] * b_centrality[p[i]]
        # This is a linear assignment problem
        C = np.outer(a_centrality, b_centrality)
        row_ind, col_ind = linear_sum_assignment(C)
        p = np.empty(n, dtype=np.int64)
        p[row_ind] = col_ind
        return p

    def frank_wolfe_init(num_iters=50):
        """Frank-Wolfe (conditional gradient) method for QAP initialization."""
        # Start with random assignment
        X = np.eye(n)  # Assignment matrix (identity as starting point)
        best_cost = float('inf')
        best_p = np.random.permutation(n)
        
        for it in range(num_iters):
            if time.time() > deadline - 30:
                break
            # Current cost
            p = np.argmax(X, axis=1)
            cost = compute_cost(p)
            if cost < best_cost:
                best_cost = cost
                best_p = p.copy()
            
            # Compute gradient
            # G[i,j] = sum_{k,l} a[k,l] * b[j,l] * X[i,k] + sum_{k,l} a[k,l] * b[k,j] * X[i,l] - corrections
            # More precisely: G = (a @ b.T) * X + (a.T @ b) * X.T
            G1 = a @ b.T
            G2 = a.T @ b
            G = G1 * X + G2 * X.T
            
            # Find best rank-1 matrix via Hungarian
            row_ind, col_ind = linear_sum_assignment(G)
            Z = np.zeros((n, n))
            Z[row_ind, col_ind] = 1
            
            # Frank-Wolfe update
            step = 2.0 / (it + 2.0)
            X = (1 - step) * X + step * Z
        
        return best_p, best_cost

    def perturbation(p, strength):
        """Apply random perturbation to permutation."""
        p = p.copy()
        n_swaps = max(2, int(n * strength))
        for _ in range(n_swaps):
            i, j = random.sample(range(n), 2)
            p[i], p[j] = p[j], p[i]
        return p

    # Initialize
    p_hung = hungarian_init()
    cost_hung = compute_cost(p_hung)
    
    p_fw, cost_fw = frank_wolfe_init(30)
    
    p_rand = np.random.permutation(n)
    cost_rand = compute_cost(p_rand)
    
    # Pick best initial
    candidates = [(p_hung, cost_hung), (p_fw, cost_fw), (p_rand, cost_rand)]
    p, cost = min(candidates, key=lambda x: x[1])
    
    best_p = p.copy()
    best_cost = cost
    
    # Local search from initial
    p, cost = vectorized_local_search(p, cost)
    if cost < best_cost:
        best_p = p.copy()
        best_cost = cost
    
    # Iterated Local Search
    perturbation_strength = 0.05
    stagnation = 0
    
    while time.time() < deadline - 15:
        # Perturb best solution
        p_pert = perturbation(best_p, perturbation_strength)
        cost_pert = compute_cost(p_pert)
        
        # Local search
        p_new, cost_new = vectorized_local_search(p_pert, cost_pert)
        
        if cost_new < best_cost:
            best_p = p_new.copy()
            best_cost = cost_new
            stagnation = 0
            # Reduce perturbation strength on success
            perturbation_strength = max(0.02, perturbation_strength * 0.8)
        else:
            stagnation += 1
            # Increase perturbation strength on stagnation
            perturbation_strength = min(0.15, perturbation_strength * 1.2)
            
            # Every 10 stagnant iterations, try a fresh Hungarian-based restart
            if stagnation % 10 == 0 and time.time() < deadline - 25:
                p_h = hungarian_init()
                cost_h = compute_cost(p_h)
                p_h, cost_h = vectorized_local_search(p_h, cost_h)
                if cost_h < best_cost:
                    best_p = p_h.copy()
                    best_cost = cost_h
                    stagnation = 0
    
    return best_p.tolist()
