import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 110.0

    n = 4
    m = 4
    p = 4

    # Strassen 2x2 (rank 7) - verified correct
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=int)
    
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=int)
    
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=int)
    
    # Verify 2x2 Strassen
    ok2 = True
    for i in range(2):
        for j in range(2):
            for k in range(2):
                s = int(np.sum(U2[i*2+j, :] * V2[j*2+k, :] * W2[k*2+i, :]))
                if s != 1:
                    ok2 = False
                    break
            if not ok2:
                break
        if not ok2:
            break
    
    if not ok2:
        # Fallback: schoolbook 4x4 (rank 64)
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Build 4x4 rank-49 from Kronecker product with proper permutation
    U4 = np.kron(U2, U2)
    V4 = np.kron(V2, V2)
    W4 = np.kron(W2, W2)
    
    # Permutation to map Kronecker row order to standard row-major order
    # U row index target: i*4 + j where i,j in 0..3
    # Kronecker row index: (i_hi*2+j_hi)*4 + (i_lo*2+j_lo)
    # Target index: i*4+j = (i_hi*2+i_lo)*4 + (j_hi*2+j_lo) = i_hi*8 + i_lo*4 + j_hi*2 + j_lo
    # Kronecker index: (i_hi*2+j_hi)*4 + (i_lo*2+j_lo) = i_hi*8 + j_hi*4 + i_lo*2 + j_lo
    
    perm = np.zeros(16, dtype=int)
    for S in range(16):
        i = S // 4
        j = S % 4
        i_hi, i_lo = i // 2, i % 2
        j_hi, j_lo = j // 2, j % 2
        R = (i_hi * 2 + j_hi) * 4 + (i_lo * 2 + j_lo)
        perm[S] = R
    
    U4_fixed = U4[perm, :]
    V4_fixed = V4[perm, :]
    W4_fixed = W4[perm, :]
    
    # Verify rank-49
    def verify_decomposition(U, V, W, n, m, p, tol=1e-10):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    if not verify_decomposition(U4_fixed, V4_fixed, W4_fixed, n, m, p):
        # Fallback: schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Convert to lists with integer entries (Strassen entries are already integers)
    U_list = U4_fixed.tolist()
    V_list = V4_fixed.tolist()
    W_list = W4_fixed.tolist()
    
    # Ensure all entries are integers (they should be from the Kronecker product of integer matrices)
    for lst in [U_list, V_list, W_list]:
        for row in lst:
            for idx, val in enumerate(row):
                if isinstance(val, (int, np.integer)):
                    row[idx] = int(val)
                else:
                    # If it's a float that's very close to an integer, convert
                    if abs(val - round(val)) < 1e-10:
                        row[idx] = int(round(val))
    
    # Final verification with the converted lists
    def verify_lists(U, V, W, n, m, p, tol=1e-10):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = sum(U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] for c in range(len(U[0])))
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    if not verify_lists(U_list, V_list, W_list, n, m, p):
        # If conversion caused issues, fall back to original as floats
        U_list = U4_fixed.tolist()
        V_list = V4_fixed.tolist()
        W_list = W4_fixed.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
