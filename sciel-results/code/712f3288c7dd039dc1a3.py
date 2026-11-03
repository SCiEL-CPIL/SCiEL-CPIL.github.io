"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def solve(a, b):
    n = len(a)
    deadline = time.time() + 115.0  # safety margin before 120s kill

    # Rank-matching start
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central

    def compute_cost(p):
        return int(np.sum(a * b[np.ix_(p, p)]))

    best_cost = compute_cost(p)
    best_p = p.copy()

    # Precompute symmetric flow matrix for efficient swap delta
    # cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
    # For swapping positions i and j (with current locations u=p[i], v=p[j]):
    # The change involves all terms where index i or j appears.
    
    # Let's derive the delta more carefully:
    # Old contribution involving i or j:
    #   sum_k a[i,k]*b[u,p[k]] + sum_k a[k,i]*b[p[k],u] + sum_k a[j,k]*b[v,p[k]] + sum_k a[k,j]*b[p[k],v]
    #   (note: when k=i in first sum, b[u,u]; when k=j in first sum, b[u,v])
    # New contribution:
    #   sum_k a[i,k]*b[v,p[k]] + sum_k a[k,i]*b[p[k],v] + sum_k a[j,k]*b[u,p[k]] + sum_k a[k,j]*b[p[k],u]
    
    # delta = new - old
    # = sum_k a[i,k]*(b[v,p[k]]-b[u,p[k]]) + sum_k a[k,i]*(b[p[k],v]-b[p[k],u])
    #   + sum_k a[j,k]*(b[u,p[k]]-b[v,p[k]]) + sum_k a[k,j]*(b[p[k],u]-b[p[k],v))
    
    # Let D[k] = b[v,p[k]] - b[u,p[k]]  (row difference)
    # Let E[k] = b[p[k],v] - b[p[k],u]  (col difference)
    
    # delta = sum_k a[i,k]*D[k] + sum_k a[k,i]*E[k] - sum_k a[j,k]*D[k] - sum_k a[k,j]*E[k]
    #       = sum_k (a[i,k]-a[j,k])*D[k] + sum_k (a[k,i]-a[k,j])*E[k]

    # Precompute nothing globally that depends on p since p changes.
    # But we can precompute a.T for faster column access if needed, though a[:,i] is already O(n).

    def swap_delta(i, j, p, a, b):
        u = p[i]
        v = p[j]
        
        # D[k] = b[v, p[k]] - b[u, p[k]]
        D = b[v, p] - b[u, p]
        
        # E[k] = b[p[k], v] - b[p[k], u]
        E = b[p, v] - b[p, u]
        
        # delta = sum_k (a[i,k]-a[j,k])*D[k] + sum_k (a[k,i]-a[k,j])*E[k]
        term1 = np.dot(a[i, :] - a[j, :], D)
        term2 = np.dot(a[:, i] - a[:, j], E)
        
        return term1 + term2

    # 2-opt local search
    improved = True
    no_improve_count = 0
    while improved and time.time() < deadline:
        improved = False
        no_improve_count += 1
        # Limit passes to avoid infinite loops
        if no_improve_count > 5:
            break
        
        for i in range(n - 1):
            if time.time() >= deadline:
                break
            for j in range(i + 1, n):
                if time.time() >= deadline:
                    break
                d = swap_delta(i, j, p, a, b)
                if d < 0:
                    p[i], p[j] = p[j], p[i]
                    best_cost += d
                    improved = True
                    no_improve_count = 0
                    if best_cost < compute_cost(p):
                        pass  # trust incremental if consistent

    # Final verification
    final_cost = compute_cost(p)
    if final_cost < best_cost:
        best_cost = final_cost
        best_p = p.copy()
    else:
        # Revert to best
        p = best_p

    return p.tolist()
# EVOLVE-BLOCK-END
