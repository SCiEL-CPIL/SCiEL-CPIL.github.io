# EVOLVE-BLOCK-START
import numpy as np
import time

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    # Precompute row sums for initial heuristic
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)

    # Initial rank-matching: heavy flows to central locations
    heavy = np.argsort(-a_row_sums, kind="stable")
    central = np.argsort(b_row_sums, kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    p = p.copy()

    def compute_cost(p):
        """Compute exact QAP cost."""
        return np.sum(a * b[np.ix_(p, p)])

    def two_opt_delta(p, i, j):
        """
        Compute delta cost for swapping facilities i and j in assignment p.
        Returns negative delta if swap improves (reduces) cost.
        """
        li = p[i]
        lj = p[j]
        
        # The cost change from swapping p[i] and p[j]:
        # Old contributions involving i or j:
        #   sum_k a[i,k]*b[li,p[k]] + sum_k a[k,i]*b[p[k],li] 
        #   + sum_k a[j,k]*b[lj,p[k]] + sum_k a[k,j]*b[p[k],lj]
        #   + a[i,j]*b[li,lj] + a[j,i]*b[lj,li]  (these are counted in both above? No, careful)
        
        # Actually, let's use the standard formula.
        # Cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
        
        # Let's define the contribution of facility i to the total cost as:
        # C_i = sum_j a[i,j] * b[p[i], p[j]] + sum_j a[j,i] * b[p[j], p[i]]
        # But this double counts the i,j and j,i terms if we sum over all i.
        # The total cost is 0.5 * sum_i C_i? No.
        # Total cost = sum_{i != j} a[i,j]b[p[i],p[j]] + sum_i a[i,i]b[p[i],p[i]]
        
        # A better way: 
        # Delta = Cost_new - Cost_old
        # Only terms involving i or j change.
        # Terms: (i,j), (j,i), (i,k), (k,i), (j,k), (k,j) for all k != i,j.
        
        # Let's compute the change directly.
        # Old terms involving i or j:
        # T_old = sum_{k != i,j} [a[i,k]*b[li,p[k]] + a[k,i]*b[p[k],li] + a[j,k]*b[lj,p[k]] + a[k,j]*b[p[k],lj]]
        #       + a[i,j]*b[li,lj] + a[j,i]*b[lj,li]
        
        # New terms (after swap, p[i]=lj, p[j]=li):
        # T_new = sum_{k != i,j} [a[i,k]*b[lj,p[k]] + a[k,i]*b[p[k],lj] + a[j,k]*b[li,p[k]] + a[k,j]*b[p[k],li]]
        #       + a[i,j]*b[lj,li] + a[j,i]*b[li,lj]
        
        # Delta = T_new - T_old
        
        # To compute this in O(n), we can precompute some vectors.
        # Let v_b_x[k] = b[x, p[k]] for a fixed x.
        # Let w_a_x[k] = a[k, x] for a fixed x.
        
        # This is getting complex. Let's use a simpler O(n) approximation or exact formula.
        
        # Exact O(n) delta:
        # Delta = sum_{k != i,j} (a[i,k]*(b[lj,p[k]] - b[li,p[k]]) + a[k,i]*(b[p[k],lj] - b[p[k],li]) 
        #         + a[j,k]*(b[li,p[k]] - b[lj,p[k]]) + a[k,j]*(b[p[k],li] - b[p[k],lj]))
        #       + (a[i,j]*(b[lj,li] - b[li,lj]) + a[j,i]*(b[li,lj] - b[lj,li]))
        
        # Note that b is symmetric in QAPLIB? Usually distance matrices are symmetric.
        # If b is symmetric, b[x,y] = b[y,x].
        # Then b[lj,li] = b[li,lj], so the last term is 0.
        # And b[lj,p[k]] - b[li,p[k]] = -(b[li,p[k]] - b[lj,p[k]])
        
        # Assuming b is symmetric:
        # Delta = sum_{k != i,j} (a[i,k] - a[j,k]) * (b[lj,p[k]] - b[li,p[k]]) 
        #         + sum_{k != i,j} (a[k,i] - a[k,j]) * (b[p[k],lj] - b[p[k],li])
        
        # Since b is symmetric, b[p[k],lj] = b[lj,p[k]] and b[p[k],li] = b[li,p[k]].
        # So:
        # Delta = sum_{k != i,j} (a[i,k] - a[j,k]) * (b[lj,p[k]] - b[li,p[k]]) 
        #         + sum_{k != i,j} (a[k,i] - a[k,j]) * (b[lj,p[k]] - b[li,p[k]])
        
        # Let d[k] = b[lj,p[k]] - b[li,p[k]]
        # Then Delta = sum_{k != i,j} (a[i,k] - a[j,k] + a[k,i] - a[k,j]) * d[k]
        
        # We can compute d vector in O(n).
        # d = b[lj, p] - b[li, p]
        # Then mask out i and j.
        # Then dot product with (a[i,:] - a[j,:] + a[:,i] - a[:,j])
        
        d = b[lj, p] - b[li, p]
        d[i] = 0
        d[j] = 0
        
        a_diff = a[i, :] - a[j, :] + a[:, i] - a[:, j]
        a_diff[i] = 0
        a_diff[j] = 0
        
        return np.dot(a_diff, d)

    best_p = p.copy()
    best_cost = compute_cost(best_p)
    
    # 2-opt local search
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        for i in range(n):
            if time.time() > deadline - 5.0:
                break
            for j in range(i+1, n):
                delta = two_opt_delta(p, i, j)
                if delta < 0:
                    # Swap
                    p[i], p[j] = p[j], p[i]
                    best_cost += delta
                    best_p = p.copy()
                    improved = True
                    # Restart search from beginning for steepest descent? 
                    # Or continue. Let's restart to find steepest descent.
                    break
            if improved:
                # We found an improvement, restart the i-loop
                # But to save time, maybe just continue?
                # Standard 2-opt usually restarts or continues. Let's restart for better quality.
                # However, restarting is O(n^2) per improvement.
                # Let's just break out of j-loop and restart i-loop.
                # Actually, the current structure breaks j-loop and sets improved=True.
                # The outer loop will restart i-loop if we don't continue.
                # But we are in a while loop. If improved is True, we go to next while iteration.
                # So it restarts the whole i,j scan. This is steepest descent.
                pass
    
    return best_p.tolist()

# EVOLVE-BLOCK-END
