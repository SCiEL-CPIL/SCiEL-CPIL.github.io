"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math


def solve(a, b):
    n = len(a)
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)

    # --- Greedy construction for a good initial solution
    # Sort facilities by total flow descending
    flow_sums = a.sum(axis=1) + a.sum(axis=0)
    order = np.argsort(-flow_sums, kind="stable")
    p = np.full(n, -1, dtype=np.int64)
    used = np.zeros(n, dtype=bool)
    
    for idx in range(n):
        f = order[idx]
        best_loc = -1
        best_cost = np.iinfo(np.int64).max
        for loc in range(n):
            if used[loc]:
                continue
            # Incremental cost: sum over already-placed facilities
            inc = 0
            for k in range(idx):
                g = order[k]
                loc_g = p[g]
                inc += a[f, g] * b[loc, loc_g] + a[g, f] * b[loc_g, loc]
            if inc < best_cost:
                best_cost = inc
                best_loc = loc
        p[f] = best_loc
        used[best_loc] = True

    def total_cost(p):
        return np.sum(a * b[np.ix_(p, p)])

    current_cost = total_cost(p)
    best_p = p.copy()
    best_cost = current_cost

    t0 = time.time()
    time_limit = 110.0

    # Simulated Annealing with better parameters
    T = 50000.0
    T_min = 1.0
    cooling_rate = 0.99995  # Slower cooling
    
    rng = np.random.default_rng(42)
    
    # Precompute row/col sums of a for faster delta
    a_row_sum = a.sum(axis=1)
    a_col_sum = a.sum(axis=0)
    
    while time.time() - t0 < time_limit:
        # Pick random i, j
        i = rng.integers(n)
        j = rng.integers(n)
        if i == j:
            continue
        
        u = p[i]
        v = p[j]
        
        # Calculate Delta for swapping positions u and v (facilities i and j)
        # New cost contribution involving i and j:
        # Old: a[i,j]*b[u,v] + a[j,i]*b[v,u] + sum_{k!=i,j} a[i,k]*b[u,p[k]] + a[k,i]*b[p[k],u]
        #                                    + sum_{k!=i,j} a[j,k]*b[v,p[k]] + a[k,j]*b[p[k],v]
        # New: a[i,j]*b[v,u] + a[j,i]*b[u,v] + sum_{k!=i,j} a[i,k]*b[v,p[k]] + a[k,i]*b[p[k],v]
        #                                    + sum_{k!=i,j} a[j,k]*b[u,p[k]] + a[k,j]*b[p[k],u]
        
        # Delta = sum_{k!=i,j} [a[i,k]*(b[v,p[k]]-b[u,p[k]]) + a[k,i]*(b[p[k],v]-b[p[k],u])
        #                         + a[j,k]*(b[u,p[k]]-b[v,p[k]]) + a[k,j]*(b[p[k],u]-b[p[k],v])]
        #       + a[i,j]*(b[v,u]-b[u,v]) + a[j,i]*(b[u,v]-b[v,u])
        
        # Simplify: a[i,k]-a[j,k] for row terms, a[k,i]-a[k,j] for col terms
        # Let diff_row[k] = a[i,k] - a[j,k]
        # Let diff_col[k] = a[k,i] - a[k,j]
        # delta = sum_{k!=i,j} diff_row[k]*(b[v,p[k]]-b[u,p[k]]) + diff_col[k]*(b[p[k],v]-b[p[k],u])
        #       + (a[i,j]-a[j,i])*(b[v,u]-b[u,v])
        
        diff_row = a[i, :] - a[j, :]
        diff_col = a[:, i] - a[:, j]
        
        # Vectorized computation
        b_v_row = b[v, p]  # shape (n,)
        b_u_row = b[u, p]  # shape (n,)
        b_col_v = b[p, v]  # shape (n,)
        b_col_u = b[p, u]  # shape (n,)
        
        # Sum over all k, then subtract k=i and k=j terms
        s1 = np.dot(diff_row, b_v_row - b_u_row)
        s2 = np.dot(diff_col, b_col_v - b_col_u)
        
        # Subtract k=i contribution (where p[i]=u, so b[v,u]-b[u,u] and b[u,v]-b[u,u])
        s1 -= diff_row[i] * (b[v, u] - b[u, u])
        s2 -= diff_col[i] * (b[u, v] - b[u, u])
        
        # Subtract k=j contribution (where p[j]=v, so b[v,v]-b[u,v] and b[v,v]-b[v,u])
        s1 -= diff_row[j] * (b[v, v] - b[u, v])
        s2 -= diff_col[j] * (b[v, v] - b[v, u])
        
        direct = (a[i, j] - a[j, i]) * (b[v, u] - b[u, v])
        
        delta = s1 + s2 + direct
        
        # Acceptance
        if delta < 0 or (T > 1e-9 and rng.random() < math.exp(-delta / T)):
            # Swap
            p[i], p[j] = p[j], p[i]
            current_cost += delta
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
        
        T *= cooling_rate
        if T < T_min:
            T = T_min

    return best_p.tolist()
# EVOLVE-BLOCK-END
