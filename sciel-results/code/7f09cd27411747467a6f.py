import numpy as np
import time

def solve(a, b):
    n = len(a)
    # 1. Initialization: Rank matching
    # Rows of a with largest sums -> Rows of b with smallest sums
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    
    # Convert to lists for easier manipulation if needed, but keep as array for speed
    # We will maintain p as a numpy array
    
    # Calculate initial cost
    # Cost = sum a[i,j] * b[p[i], p[j]]
    # Efficient cost calculation:
    # Let M = b[p[:, None], p[None, :]]
    # Cost = np.sum(a * M)
    
    def get_cost(p_arr):
        # p_arr is the permutation
        # b_perm is b with rows and cols permuted by p
        # We need b[p[i], p[j]]
        # This is equivalent to b[np.ix_(p_arr, p_arr)]
        b_perm = b[np.ix_(p_arr, p_arr)]
        return np.sum(a * b_perm)
    
    current_cost = get_cost(p)
    
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    # 2. Local Search: 2-opt
    # We iterate over pairs (i, j)
    # To speed up, we can precompute some things or use vectorization for delta calc
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # We can randomize the order of pairs to avoid cycling or getting stuck, 
        # or just use sequential. Sequential is deterministic and fine.
        # For better results, we might want to try all pairs and pick the best one (Best Improvement)
        # or just first improvement. First improvement is faster.
        
        # To avoid O(N^3) python loops, we can try to vectorize the delta calculation for a fixed i?
        # Or just accept the loop if it's fast enough.
        # N=150. 11025 pairs.
        # Delta calc: 
        # d = b[p[j], p] - b[p[i], p]  (vector of length N)
        # diff_a = a[i, :] - a[j, :] (vector of length N)
        # delta = np.dot(diff_a, d) - diff_a[i]*d[i] - diff_a[j]*d[j] + a[i,j]*(b[p[j],p[i]] - b[p[i],p[j]])
        # Note: d[i] = b[p[j], p[i]] - b[p[i], p[i]] = b[v, u] - b[u, u]
        # d[j] = b[p[j], p[j]] - b[p[i], p[j]] = b[v, v] - b[u, v]
        # The term a[i,k] b[v, p[k]] is included in dot.
        # We need to exclude k=i and k=j from the sum in the formula derivation?
        # Let's re-verify the formula.
        # Original: Sum_{k != i, j} (a[i,k] - a[j,k]) (b[v, p[k]] - b[u, p[k]]) + a[i,j] (b[v,u] - b[u,v])
        # Dot product Sum_{k} (a[i,k] - a[j,k]) (b[v, p[k]] - b[u, p[k]])
        # Includes k=i: (a[i,i] - a[j,i]) (b[v, u] - b[u, u])
        # Includes k=j: (a[i,j] - a[j,j]) (b[v, v] - b[u, v])
        # We need to subtract these two terms and add the correct interaction term a[i,j](b[v,u] - b[u,v]).
        
        for i in range(n):
            if time.time() > deadline:
                break
            for j in range(i + 1, n):
                if time.time() > deadline:
                    break
                
                u = p[i]
                v = p[j]
                
                # Vector d_k = b[v, p[k]] - b[u, p[k]]
                # We can get row v and row u of b, then index by p
                # b_v = b[v, :]
                # b_u = b[u, :]
                # d = b_v[p] - b_u[p]
                
                # To avoid creating new arrays every time, we can precompute b_rows?
                # b is static. p changes.
                # b[v, p] is a gather operation.
                
                diff_a = a[i, :] - a[j, :]
                
                # Compute d
                # d = b[v, p] - b[u, p]
                # This is O(N).
                d = b[v, p] - b[u, p]
                
                # Dot product
                dot_val = np.dot(diff_a, d)
                
                # Subtract terms for k=i and k=j
                # Term i: diff_a[i] * d[i]
                # Term j: diff_a[j] * d[j]
                # Add interaction term
                interaction = a[i, j] * (b[v, u] - b[u, v])
                
                delta = dot_val - diff_a[i] * d[i] - diff_a[j] * d[j] + interaction
                
                if delta < 0:
                    # Perform swap
                    p[i], p[j] = p[j], p[i]
                    current_cost += delta
                    improved = True
                    # After swap, we can restart the inner loop or just continue.
                    # Restarting ensures we don't miss immediate improvements.
                    # But for speed, just continuing is often fine.
                    # Let's restart the outer loop to be safe and effective.
                    break
            if improved:
                break
            if time.time() > deadline:
                break

    return p.tolist()
