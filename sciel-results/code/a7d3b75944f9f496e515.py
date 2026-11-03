import numpy as np
import time

def solve(a, b):
    n = len(a)
    # Rank-matching start
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    
    # Precompute S matrix where S[k, i] = a[k, i] + a[i, k]
    S = a + a.T
    
    # Current cost
    # C = sum a[i,j] b[p[i], p[j]]
    # Can compute efficiently or just track delta
    # Let's just track cost by delta updates
    # Initial cost calculation
    # Using np.einsum or dot products
    # b_p = b[np.ix_(p, p)]
    # cost = np.sum(a * b_p)
    # This is O(n^2) and fast enough once.
    b_p = b[np.ix_(p, p)]
    cost = np.sum(a * b_p)
    
    start_time = time.time()
    deadline = 115.0  # Leave 5s buffer for 120s limit
    
    # 2-Opt Local Search
    while time.time() < deadline:
        improved = False
        for i in range(n):
            if time.time() > deadline:
                break
            u = p[i]
            for j in range(i + 1, n):
                if time.time() > deadline:
                    break
                v = p[j]
                
                # B_uv[k] = b[p[k], v] - b[p[k], u]
                # We need b[:, v] - b[:, u] indexed by p
                # b_p_col_v = b_p[:, j] is b[p[k], p[j]] = b[p[k], v]
                # b_p_col_u = b_p[:, i] is b[p[k], p[i]] = b[p[k], u]
                # So B_uv = b_p[:, j] - b_p[:, i]
                
                B_uv = b_p[:, j] - b_p[:, i]
                
                # W[k] = S[k, i] - S[k, j]
                W = S[:, i] - S[:, j]
                
                delta = np.dot(W, B_uv)
                
                # Correct for k=i and k=j terms which were included in dot but should be excluded
                # The derivation: Delta = sum_{k != i, j} W[k] B_uv[k]
                # dot(W, B_uv) = sum_{all k} W[k] B_uv[k]
                # So subtract W[i]B_uv[i] and W[j]B_uv[j]
                delta -= W[i] * B_uv[i]
                delta -= W[j] * B_uv[j]
                
                if delta < 0:
                    # Perform swap
                    p[i], p[j] = p[j], p[i]
                    cost += delta
                    improved = True
                    # Update b_p to reflect the swap
                    # b_p[:, i] and b_p[:, j] are swapped
                    # b_p[i, :] and b_p[j, :] are swapped
                    b_p[:, [i, j]] = b_p[:, [j, i]]
                    b_p[[i, j], :] = b_p[[j, i], :]
                    break  # Restart j loop for this i? Or just break to next i?
                         # Breaking to next i is standard "first improvement" in a sweep.
                         # But since we modified p, the indices for subsequent j in this i-loop
                         # are still valid, but the values changed.
                         # Usually, after a swap, we restart the entire scan or at least the inner loop.
                         # Let's restart the inner loop to be safe and simple.
                         # Actually, breaking the inner loop and moving to next i is fine,
                         # but we might miss a better j for the new i?
                         # Standard 2-opt often does:
                         # for i in range(n):
                         #   for j in range(i+1, n):
                         #     if improve: swap; break inner; 
                         # This is fine.
        
        if not improved:
            break
            
    return p.tolist()
