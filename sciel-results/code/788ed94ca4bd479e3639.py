import numpy as np
import time
import itertools

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    max_time = 140.0
    
    nm, mp, pn = n * m, m * p, p * n
    
    i_vals, j_vals, k_vals = np.indices((n, m, p))
    u_idx = (i_vals * m + j_vals).flatten()
    v_idx = (j_vals * p + k_vals).flatten()
    w_idx = (k_vals * n + i_vals).flatten()
    num_terms = n * m * p  # 24
    
    # Schoolbook
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(nm)]
    V_sb = [[0] * rank_sb for _ in range(mp)]
    W_sb = [[0] * rank_sb for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    def verify(U, V, W, r):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        for t in range(num_terms):
            s = sum(U_arr[c, u_idx[t]] * V_arr[c, v_idx[t]] * W_arr[c, w_idx[t]] for c in range(r))
            if abs(s - 1.0) > 1e-9:
                return False
        return True
    
    def to_list(arr):
        return [[round(v * 2) / 2.0 for v in row] for row in arr]
    
    candidate_vals = [-1.0, -0.5, 0.0, 0.5, 1.0]
    
    # Try to find rank-23 by merging pairs of schoolbook columns
    for c1 in range(24):
        if time.time() - start_time > max_time:
            break
        for c2 in range(c1 + 1, 24):
            if time.time() - start_time > max_time:
                break
            
            # Get the (i,j,k) for each
            i1, j1, k1 = c1 // (m*p), (c1 % (m*p)) // p, c1 % p
            i2, j2, k2 = c2 // (m*p), (c2 % (m*p)) // p, c2 % p
            
            # Positions that matter
            u_pos = sorted(set([i1*m+j1, i2*m+j2]))
            v_pos = sorted(set([j1*p+k1, j2*p+k2]))
            w_pos = sorted(set([k1*n+i1, k2*n+i2]))
            
            # Enumerate assignments
            found = False
            for ua in itertools.product(candidate_vals, repeat=len(u_pos)):
                u = [0.0]*nm
                for pos, val in zip(u_pos, ua):
                    u[pos] = val
                for va in itertools.product(candidate_vals, repeat=len(v_pos)):
                    v = [0.0]*mp
                    for pos, val in zip(v_pos, va):
                        v[pos] = val
                    for wa in itertools.product(candidate_vals, repeat=len(w_pos)):
                        w = [0.0]*pn
                        for pos, val in zip(w_pos, wa):
                            w[pos] = val
                        
                        ok = True
                        for t in range(num_terms):
                            prod = u[u_idx[t]] * v[v_idx[t]] * w[w_idx[t]]
                            i_t = u_idx[t] // m
                            j_t = u_idx[t] % m
                            k_t = v_idx[t] % p
                            expected = 0.0
                            if (i_t, j_t, k_t) == (i1, j1, k1):
                                expected += 1.0
                            if (i_t, j_t, k_t) == (i2, j2, k2):
                                expected += 1.0
                            if abs(prod - expected) > 1e-9:
                                ok = False
                                break
                        if ok:
                            # Build rank-23 solution
                            U23 = [U_sb[c] for c in range(24) if c not in (c1, c2)]
                            V23 = [V_sb[c] for c in range(24) if c not in (c1, c2)]
                            W23 = [W_sb[c] for c in range(24) if c not in (c1, c2)]
                            U23.append(u)
                            V23.append(v)
                            W23.append(w)
                            
                            if verify(U23, V23, W23, 23):
                                return {"U": to_list(np.array(U23).T), "V": to_list(np.array(V23).T), "W": to_list(np.array(W23).T)}
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
            if found:
                continue
    
    return {"U": U_sb, "V": V_sb, "W": W_sb}
