"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_cost(a, b, p):
    """Compute the QAP cost sum_{i,j} a[i,j]*b[p[i],p[j]] efficiently."""
    return int(np.sum(a * b[np.ix_(p, p)]))


def solve(a, b):
    n = len(a)
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    
    deadline = time.time() + 115  # 5s safety margin
    
    # Precompute some useful quantities
    # Row sums of a
    a_row_sum = a.sum(axis=1)
    a_col_sum = a.sum(axis=0)
    
    # For the delta computation, we need efficient access
    
    def _delta_swap(p, i, j):
        """Compute delta for swapping facilities i and j in permutation p."""
        di = p[i]
        dj = p[j]
        
        # Precomputed arrays for efficiency
        # diff_row = b[dj, :] - b[di, :]
        # diff_col = b[:, dj] - b[:, di]
        
        # Term 1: cross term between i and j
        term1 = (a[i, j] - a[j, i]) * (b[dj, di] - b[di, dj])
        
        # For efficiency, compute the four dot products
        # S_i = sum_l a[i,l] * (b[dj,p[l]] - b[di,p[l]])
        #     = a[i,:] @ (b[dj, p] - b[di, p])
        # S_j = sum_l a[j,l] * (b[di,p[l]] - b[dj,p[l]])
        #     = a[j,:] @ (b[di, p] - b[dj, p])
        # S_ki = sum_k a[k,i] * (b[p[k],dj] - b[p[k],di])
        #       = a[:,i] @ (b[p, dj] - b[p, di])
        # S_kj = sum_k a[k,j] * (b[p[k],di] - b[p[k],dj])
        #       = a[:,j] @ (b[p, di] - b[p, dj])
        
        diff1 = b[dj, p] - b[di, p]
        diff3 = b[p, dj] - b[p, di]
        
        S_i = np.dot(a[i, :], diff1)
        S_j = -np.dot(a[j, :], diff1)
        S_ki = np.dot(a[:, i], diff3)
        S_kj = -np.dot(a[:, j], diff3)
        
        # Subtract terms where l==i, l==j for S_i and S_j
        # S_i includes l=i: a[i,i]*(b[dj,di]-b[di,di]) and l=j: a[i,j]*(b[dj,dj]-b[di,dj])
        S_i -= a[i, i] * (b[dj, di] - b[di, di])
        S_i -= a[i, j] * (b[dj, dj] - b[di, dj])
        
        # S_j includes l=i: a[j,i]*(b[di,di]-b[dj,di]) and l=j: a[j,j]*(b[di,dj]-b[dj,dj])
        S_j -= a[j, i] * (b[di, di] - b[dj, di])
        S_j -= a[j, j] * (b[di, dj] - b[dj, dj])
        
        # S_ki includes k=i: a[i,i]*(b[di,dj]-b[di,di]) and k=j: a[j,i]*(b[dj,dj]-b[dj,di])
        S_ki -= a[i, i] * (b[di, dj] - b[di, di])
        S_ki -= a[j, i] * (b[dj, dj] - b[dj, di])
        
        # S_kj includes k=i: a[i,j]*(b[di,di]-b[di,dj]) and k=j: a[j,j]*(b[dj,di]-b[dj,dj])
        S_kj -= a[i, j] * (b[di, di] - b[di, dj])
        S_kj -= a[j, j] * (b[dj, di] - b[dj, dj])
        
        return term1 + S_i + S_j + S_ki + S_kj
    
    # Initial solution using rank-matching heuristic
    # Match high-flow facilities to low-distance locations
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    
    best_p = p.copy()
    best_cost = _compute_cost(a, b, p)
    
    # Tabu search with first-improvement for speed, but track best
    tabu_len = 15
    tabu = np.zeros(n, dtype=np.int64)
    tabu_count = 0
    
    # Use a priority-like approach: sample pairs to find good moves quickly
    # For n=150, full search is 11175 pairs per iteration
    
    while time.time() < deadline:
        best_delta = 0
        best_i, best_j = -1, -1
        
        # Full search for best improving non-tabu move
        for i in range(n):
            if time.time() > deadline:
                break
            for j in range(i + 1, n):
                if tabu[i] > tabu_count or tabu[j] > tabu_count:
                    continue
                d = _delta_swap(p, i, j)
                if d < best_delta:
                    best_delta = d
                    best_i, best_j = i, j
        
        if best_i == -1:
            # No improving non-tabu move; find best improving move overall
            for i in range(n):
                for j in range(i + 1, n):
                    d = _delta_swap(p, i, j)
                    if d < best_delta:
                        best_delta = d
                        best_i, best_j = i, j
            
            if best_i == -1:
                # No improving move; diversify
                # Pick a random pair and apply it
                i, j = np.random.choice(n, size=2, replace=False)
                best_i, best_j = i, j
                best_delta = _delta_swap(p, i, j)
                tabu[:] = 0
        
        # Apply the move
        p[best_i], p[best_j] = p[best_j], p[best_i]
        tabu[best_i] = tabu_count + tabu_len
        tabu[best_j] = tabu_count + tabu_len
        tabu_count += 1
        
        new_cost = best_cost + int(best_delta)
        if new_cost < best_cost:
            best_cost = new_cost
            best_p = p.copy()
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
