import numpy as np
import time

def solve(a, b):
        n = len(a)
        # Initialization
        heavy = np.argsort(-a.sum(axis=1), kind="stable")
        central = np.argsort(b.sum(axis=1), kind="stable")
        p = np.empty(n, dtype=np.int64)
        p[heavy] = central
        
        start_time = time.time()
        deadline = start_time + 115.0 # 5s safety margin
        
        def compute_cost(p):
            # Cost = sum a[i,j] b[p[i], p[j]]
            # Vectorized:
            # b_p = b[np.ix_(p, p)]
            # return np.sum(a * b_p)
            return np.sum(a * b[np.ix_(p, p)])
            
        current_cost = compute_cost(p)
        best_cost = current_cost
        best_p = p.copy()
        
        while time.time() < deadline:
            improved = False
            # Sequential 2-opt
            for i in range(n):
                if time.time() > deadline:
                    break
                u = p[i]
                for j in range(i + 1, n):
                    v = p[j]
                    if u == v: continue # Should not happen
                    
                    # Calculate Delta
                    # d_uv = b[v, :] - b[u, :]
                    # d_vu = b[:, v] - b[:, u]
                    
                    # Sum over k != i, j
                    # Term 1: sum (a[i,k] - a[j,k]) * d_uv[p[k]]
                    # Term 2: sum (a[k,i] - a[k,j]) * d_vu[p[k]]
                    
                    # Vectorized calculation for this pair
                    # Create mask? No, just subtract the i,j terms from the full sum.
                    
                    d_uv = b[v, :] - b[u, :]
                    d_vu = b[:, v] - b[:, u]
                    
                    # Full sums
                    s1 = np.dot(a[i, :] - a[j, :], d_uv[p])
                    s2 = np.dot(a[:, i] - a[:, j], d_vu[p])
                    
                    # Subtract k=i and k=j terms
                    # For s1: k=i -> (a[i,i]-a[j,i]) * d_uv[u]
                    #        k=j -> (a[i,j]-a[j,j]) * d_uv[v]
                    s1 -= (a[i, i] - a[j, i]) * d_uv[u]
                    s1 -= (a[i, j] - a[j, j]) * d_uv[v]
                    
                    # For s2: k=i -> (a[i,i]-a[i,j]) * d_vu[u]
                    #        k=j -> (a[j,i]-a[j,j]) * d_vu[v]
                    s2 -= (a[i, i] - a[i, j]) * d_vu[u]
                    s2 -= (a[j, i] - a[j, j]) * d_vu[v]
                    
                    # Cross terms
                    cross = a[i, j] * (b[v, u] - b[u, v]) + a[j, i] * (b[u, v] - b[v, u])
                    
                    delta = s1 + s2 + cross
                    
                    if delta < 0:
                        # Perform swap
                        p[i], p[j] = v, u
                        current_cost += delta
                        improved = True
                        # Restart scan from 0? Or continue?
                        # Continuing is faster, but might miss improvements.
                        # Let's break inner loop to restart outer?
                        # Or just break both and restart full scan.
                        break
                if improved:
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_p = p.copy()
                    # Restart scan
                    break
            if not improved:
                # No improvement in full pass, stop? Or random restart?
                # Try a few random restarts if time permits
                break
                
        return best_p.tolist()
