import numpy as np
import time

def solve(a, b):
        n = len(a)
        start_time = time.time()
        deadline = start_time + 115.0
        
        # Initialization
        heavy = np.argsort(-a.sum(axis=1), kind="stable")
        central = np.argsort(b.sum(axis=1), kind="stable")
        p = np.empty(n, dtype=np.int64)
        p[heavy] = central
        
        def calc_cost(p):
            # sum a[i,j] * b[p[i], p[j]]
            # Use matrix multiplication?
            # A @ B_perm?
            # B_perm[i,j] = b[p[i], p[j]]
            # Cost = sum(A * B_perm)
            B_perm = b[np.ix_(p, p)]
            return np.sum(a * B_perm)
            
        current_p = p.copy()
        current_cost = calc_cost(current_p)
        best_p = current_p.copy()
        best_cost = current_cost
        
        def local_search(p, cost, deadline):
            n = len(p)
            improved = True
            while improved and time.time() < deadline:
                improved = False
                best_delta = 0
                best_i = -1
                best_j = -1
                
                # Precompute rows of b?
                # b[p[i], :] is a row.
                # We can just index inside.
                
                for i in range(n):
                    ai = a[i, :]
                    bi_row = b[p[i], :]
                    for j in range(i + 1, n):
                        aj = a[j, :]
                        bj_row = b[p[j], :]
                        
                        # delta = 2 * sum_{k != i,j} (a_ik - a_jk) * (b_jk - b_ik)
                        # Note: b_jk here means b[p[j], p[k]]
                        # So we need b[p[j], p[k]] for all k.
                        # This is not just a row of b, it's a permuted row.
                        # b[p[j], p[k]] is the k-th element of the row b[p[j], :] permuted by p?
                        # No. b is location-location.
                        # p[k] is the location of facility k.
                        # So b[p[j], p[k]] is the distance between loc(j) and loc(k).
                        # Let L_j = p[j], L_k = p[k].
                        # We need the vector v where v[k] = b[L_j, p[k]].
                        # This is b[L_j, p].
                        
                        # So:
                        # diff_a = ai - aj
                        # diff_b = b[L_j, p] - b[L_i, p]
                        # mask out i, j
                        # delta = 2 * dot(diff_a, diff_b)
                        
                        diff_a = ai - aj
                        # b[L_j, p] is b[p[j], p]
                        diff_b = b[p[j], p] - b[p[i], p]
                        
                        # Zero out i and j
                        diff_a[i] = 0
                        diff_a[j] = 0
                        diff_b[i] = 0
                        diff_b[j] = 0
                        
                        delta = 2 * np.dot(diff_a, diff_b)
                        
                        if delta < best_delta:
                            best_delta = delta
                            best_i = i
                            best_j = j
                
                if best_i != -1:
                    p[best_i], p[best_j] = p[best_j], p[best_i]
                    cost += best_delta
                    improved = True
                    
            return p, cost

        while time.time() < deadline - 5:
            current_p, current_cost = local_search(current_p, current_cost, deadline)
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = current_p.copy()
            
            # Perturb if time allows
            if time.time() > deadline - 10:
                break
            
            # Random kick: swap 3 random elements
            idx = np.random.choice(n, 3, replace=False)
            current_p = best_p.copy() # Start from best
            # Apply kick
            current_p[idx[0]], current_p[idx[1]] = current_p[idx[1]], current_p[idx[0]]
            current_p[idx[1]], current_p[idx[2]] = current_p[idx[2]], current_p[idx[1]]
            current_cost = calc_cost(current_p)
            
        return best_p.tolist()
