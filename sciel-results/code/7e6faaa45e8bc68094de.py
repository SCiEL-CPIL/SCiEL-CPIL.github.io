import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    n, m, p = DIMENSIONS
    
    # Build target tensor: T[a,b,c] where a=i*m+j, b=j*p+k, c=k*n+i
    target = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target[(a, b, c)] = 1
    
    # Build the rank-50 solution
    rank_init = 50
    U_init = [[0] * rank_init for _ in range(n * m)]
    V_init = [[0] * rank_init for _ in range(m * p)]
    W_init = [[0] * rank_init for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_222(i0, j0, k0):
        nonlocal col
        if col >= rank_init:
            return False
            
        a = i0 * m + j0
        b = i0 * m + (j0 + 1)
        c = (i0 + 1) * m + j0
        d = (i0 + 1) * m + (j0 + 1)
        
        e = j0 * p + k0
        f = j0 * p + (k0 + 1)
        g = (j0 + 1) * p + k0
        h = (j0 + 1) * p + (k0 + 1)
        
        c00 = k0 * n + i0
        c01 = (k0 + 1) * n + i0
        c10 = k0 * n + (i0 + 1)
        c11 = (k0 + 1) * n + (i0 + 1)
        
        U_init[a][col] += 1
        U_init[d][col] += 1
        V_init[e][col] += 1
        V_init[h][col] += 1
        W_init[c00][col] += 1
        W_init[c11][col] += 1
        col += 1
        
        U_init[c][col] += 1
        U_init[d][col] += 1
        V_init[e][col] += 1
        W_init[c10][col] += 1
        W_init[c11][col] -= 1
        col += 1
        
        U_init[a][col] += 1
        V_init[f][col] += 1
        V_init[h][col] -= 1
        W_init[c01][col] += 1
        W_init[c11][col] += 1
        col += 1
        
        U_init[d][col] += 1
        V_init[g][col] += 1
        V_init[e][col] -= 1
        W_init[c00][col] += 1
        W_init[c10][col] += 1
        col += 1
        
        U_init[a][col] += 1
        U_init[b][col] += 1
        V_init[h][col] += 1
        W_init[c00][col] -= 1
        W_init[c01][col] += 1
        col += 1
        
        U_init[c][col] += 1
        U_init[a][col] -= 1
        V_init[e][col] += 1
        V_init[f][col] += 1
        W_init[c11][col] += 1
        col += 1
        
        U_init[b][col] += 1
        U_init[d][col] -= 1
        V_init[g][col] += 1
        V_init[h][col] += 1
        W_init[c00][col] += 1
        col += 1
        
        return True

    def add_schoolbook_221(i0, j0, k0):
        nonlocal col
        if col >= rank_init:
            return False
            
        a = i0 * m + j0
        b = i0 * m + (j0 + 1)
        c = (i0 + 1) * m + j0
        d = (i0 + 1) * m + (j0 + 1)
        
        e = j0 * p + k0
        g = (j0 + 1) * p + k0
        
        c00 = k0 * n + i0
        c10 = k0 * n + (i0 + 1)
        
        U_init[a][col] += 1
        V_init[e][col] += 1
        W_init[c00][col] += 1
        col += 1
        
        U_init[b][col] += 1
        V_init[g][col] += 1
        W_init[c00][col] += 1
        col += 1
        
        U_init[c][col] += 1
        V_init[e][col] += 1
        W_init[c10][col] += 1
        col += 1
        
        U_init[d][col] += 1
        V_init[g][col] += 1
        W_init[c10][col] += 1
        col += 1
        
        return True

    for j_start in [0, 2]:
        if time.time() > deadline:
            break
        for k_start in [0, 2, 4]:
            if time.time() > deadline:
                break
            if not add_strassen_222(0, j_start, k_start):
                break
        if time.time() > deadline:
            break
        if not add_schoolbook_221(0, j_start, 6):
            break
    
    # Convert to numpy
    U_np = np.array(U_init, dtype=float)
    V_np = np.array(V_init, dtype=float)
    W_np = np.array(W_init, dtype=float)
    
    # Verify
    valid = True
    for (a, b, c) in target.keys():
        s = np.dot(U_np[a], V_np[b] * W_np[c])
        if abs(s - 1) > 1e-6:
            valid = False
            break
            
    if not valid:
        # Fallback to trivial rank 56
        rank_trivial = n * m * p
        U_t = np.zeros((n * m, rank_trivial))
        V_t = np.zeros((m * p, rank_trivial))
        W_t = np.zeros((p * n, rank_trivial))
        
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U_t[a, idx] = 1
                    V_t[b, idx] = 1
                    W_t[c, idx] = 1
                    idx += 1
        return {"U": U_t, "V": V_t, "W": W_t}
        
    return {"U": U_np, "V": V_np, "W": W_np}
