"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 110
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n  # 6, 12, 8
    total = n*m*p  # 24
    
    def verify(U, V, W, rank):
        U = np.array(U, dtype=np.float64)
        V = np.array(V, dtype=np.float64)
        W = np.array(W, dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    # Try to find a lower-rank decomposition
    for target_rank in [20, 21, 22, 23]:
        if time.time() > deadline:
            break
        
        start = time.time()
        
        for seed in range(30):
            if time.time() > deadline or time.time() - start > 105:
                break
            
            np.random.seed(seed + target_rank * 7777)
            
            U = np.zeros((nm, target_rank))
            V = np.zeros((mp, target_rank))
            W = np.zeros((pn, target_rank))
            
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if col < target_rank:
                            U[i*m+j, col] = 1
                            V[j*p+k, col] = 1
                            W[k*n+i, col] = 1
                            col += 1
            
            for it in range(400):
                if time.time() > deadline:
                    break
                
                if verify(U, V, W, target_rank):
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                
                res = {}
                worst_key = None
                max_abs = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for c in range(target_rank):
                                s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                            r = s - 1.0
                            res[(i,j,k)] = r
                            if abs(r) > max_abs:
                                max_abs = abs(r)
                                worst_key = (i,j,k)
                
                if max_abs < 1e-9:
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                
                wi, wj, wk = worst_key
                target = -res[worst_key]
                
                best_mod = None
                best_score = float('inf')
                
                for c in range(target_rank):
                    v = V[wj*p+wk, c]
                    w = W[wk*n+wi, c]
                    if v != 0 and w != 0:
                        d = v * w
                        if d != 0:
                            delta = round(target / d)
                            if delta != 0:
                                new_u = U[wi*m+wj, c] + delta
                                help_val = abs(target - d * delta)
                                if help_val < best_score:
                                    best_score = help_val
                                    best_mod = ('U', wi*m+wj, c, new_u)
                    
                    u = U[wi*m+wj, c]
                    w = W[wk*n+wi, c]
                    if u != 0 and w != 0:
                        d = u * w
                        if d != 0:
                            delta = round(target / d)
                            if delta != 0:
                                new_v = V[wj*p+wk, c] + delta
                                help_val = abs(target - d * delta)
                                if help_val < best_score:
                                    best_score = help_val
                                    best_mod = ('V', wj*p+wk, c, new_v)
                    
                    u = U[wi*m+wj, c]
                    v = V[wj*p+wk, c]
                    if u != 0 and v != 0:
                        d = u * v
                        if d != 0:
                            delta = round(target / d)
                            if delta != 0:
                                new_w = W[wk*n+wi, c] + delta
                                help_val = abs(target - d * delta)
                                if help_val < best_score:
                                    best_score = help_val
                                    best_mod = ('W', wk*n+wi, c, new_w)
                
                if best_mod is not None:
                    mat, row, col_idx, new_val = best_mod
                    if mat == 'U':
                        U[row, col_idx] = new_val
                    elif mat == 'V':
                        V[row, col_idx] = new_val
                    else:
                        W[row, col_idx] = new_val
                else:
                    mat_choice = np.random.choice(['U', 'V', 'W'])
                    if mat_choice == 'U':
                        row = np.random.randint(nm)
                        col_idx = np.random.randint(target_rank)
                        U[row, col_idx] += np.random.choice([-1, 1])
                    elif mat_choice == 'V':
                        row = np.random.randint(mp)
                        col_idx = np.random.randint(target_rank)
                        V[row, col_idx] += np.random.choice([-1, 1])
                    else:
                        row = np.random.randint(pn)
                        col_idx = np.random.randint(target_rank)
                        W[row, col_idx] += np.random.choice([-1, 1])
    
    # Fallback: schoolbook
    rank = total
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
