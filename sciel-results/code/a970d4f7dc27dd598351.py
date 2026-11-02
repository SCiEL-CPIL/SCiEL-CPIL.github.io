import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 20s margin for 150s limit
    
    n, m, p = DIMENSIONS
    rank = 50
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_222(i0, j0, k0):
        nonlocal col
        if col >= rank:
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
        
        U[a][col] += 1
        U[d][col] += 1
        V[e][col] += 1
        V[h][col] += 1
        W[c00][col] += 1
        W[c11][col] += 1
        col += 1
        
        U[c][col] += 1
        U[d][col] += 1
        V[e][col] += 1
        W[c10][col] += 1
        W[c11][col] -= 1
        col += 1
        
        U[a][col] += 1
        V[f][col] += 1
        V[h][col] -= 1
        W[c01][col] += 1
        W[c11][col] += 1
        col += 1
        
        U[d][col] += 1
        V[g][col] += 1
        V[e][col] -= 1
        W[c00][col] += 1
        W[c10][col] += 1
        col += 1
        
        U[a][col] += 1
        U[b][col] += 1
        V[h][col] += 1
        W[c00][col] -= 1
        W[c01][col] += 1
        col += 1
        
        U[c][col] += 1
        U[a][col] -= 1
        V[e][col] += 1
        V[f][col] += 1
        W[c11][col] += 1
        col += 1
        
        U[b][col] += 1
        U[d][col] -= 1
        V[g][col] += 1
        V[h][col] += 1
        W[c00][col] += 1
        col += 1
        
        return True

    def add_schoolbook_221(i0, j0, k0):
        nonlocal col
        if col >= rank:
            return False
            
        a = i0 * m + j0
        b = i0 * m + (j0 + 1)
        c = (i0 + 1) * m + j0
        d = (i0 + 1) * m + (j0 + 1)
        
        e = j0 * p + k0
        g = (j0 + 1) * p + k0
        
        c00 = k0 * n + i0
        c10 = k0 * n + (i0 + 1)
        
        U[a][col] += 1
        V[e][col] += 1
        W[c00][col] += 1
        col += 1
        
        U[b][col] += 1
        V[g][col] += 1
        W[c00][col] += 1
        col += 1
        
        U[c][col] += 1
        V[e][col] += 1
        W[c10][col] += 1
        col += 1
        
        U[d][col] += 1
        V[g][col] += 1
        W[c10][col] += 1
        col += 1
        
        return True

    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
        
    # Block 1: Strassen for columns 0-1, rows 0-1
    for k0 in [0, 2, 4]:
        if not add_strassen_222(0, 0, k0):
            break
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
        
    # Schoolbook for column 6, rows 0-1
    if not add_schoolbook_221(0, 0, 6):
        pass
        
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
        
    # Block 2: Strassen for columns 2-3, rows 0-1
    for k0 in [0, 2, 4]:
        if not add_strassen_222(0, 2, k0):
            break
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
        
    # Schoolbook for column 6, rows 2-3 (wait, m=4, so rows 0-1 and 2-3 don't exist)
    # Actually m=4, so we have 4 columns in A. The blocks are:
    # A is 2x4: rows 0,1; cols 0,1,2,3
    # B is 4x7: rows 0,1,2,3; cols 0,1,2,3,4,5,6
    # C is 2x7: rows 0,1; cols 0,1,2,3,4,5,6
    
    # The decomposition should be:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k] + A[i,3]*B[3,k]
    
    # We can group: (A[i,0]*B[0,k] + A[i,1]*B[1,k]) + (A[i,2]*B[2,k] + A[i,3]*B[3,k])
    # Each group is a 2x2 times 2x7 product.
    
    # For the first group (j=0,1): 2x2 * 2x7
    # We can do: 2x2 * 2x2 for k=0,1; 2x2 * 2x2 for k=2,3; 2x2 * 2x2 for k=4,5; 2x2 * 2x1 for k=6
    # That's 3*7 + 4 = 25 rank for the first group.
    
    # For the second group (j=2,3): same structure, another 25 rank.
    # Total: 50 rank.
    
    # The current code does:
    # add_strassen_222(0, 0, k0) for k0 in [0,2,4]: 3 blocks of rank 7 = 21
    # add_schoolbook_221(0, 0, 6): rank 4
    # Total for first group: 25
    
    # add_strassen_222(0, 2, k0) for k0 in [0,2,4]: 3 blocks of rank 7 = 21
    # add_schoolbook_221(0, 2, 6): rank 4
    # Total for second group: 25
    
    # Grand total: 50
    
    # Now I need to add the second schoolbook
    if not add_schoolbook_221(0, 2, 6):
        pass

    # Verify the decomposition is correct
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    max_error = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                reconstructed = 0
                for r in range(rank):
                    reconstructed += U_arr[a_idx, r] * V_arr[b_idx, r] * W_arr[c_idx, r]
                if abs(reconstructed - 1) > 1e-9:
                    max_error = max(max_error, abs(reconstructed - 1))
    
    if max_error > 1e-9:
        return {"U": [[0]*rank for _ in range(n*m)], 
                "V": [[0]*rank for _ in range(m*p)], 
                "W": [[0]*rank for _ in range(p*n)]}
    
    return {"U": U, "V": V, "W": W}
