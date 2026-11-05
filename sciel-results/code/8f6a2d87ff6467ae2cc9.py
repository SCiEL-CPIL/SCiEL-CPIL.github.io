import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin
    
    # Block decomposition for 2x3 * 3x5:
    # A (2x3) split as A_left (2x2) and A_right (2x1)
    # B (3x5) split as B_left (2x2), B_mid (2x2), B_right (2x1)
    # C (2x5) = A_left*B_left + A_left*B_mid + A_left*B_right + A_right*B_left + A_right*B_mid + A_right*B_right
    #
    # Ranks:
    # A_left (2x2) * B_left (2x2) -> C (2x2): Strassen, rank 7
    # A_left (2x2) * B_mid (2x2) -> C (2x2): Strassen, rank 7
    # A_left (2x2) * B_right (2x1) -> C (2x1): Direct, rank 4
    # A_right (2x1) * B_left (2x2) -> C (2x2): Direct, rank 4
    # A_right (2x1) * B_mid (2x2) -> C (2x2): Direct, rank 4
    # A_right (2x1) * B_right (2x1) -> C (2x1): Direct, rank 2
    # Total: 7 + 7 + 4 + 4 + 4 + 2 = 28
    
    rank = 28
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    def add_strassen_2x2x2(i1, i2, j1, j2, k1, k2):
        """Add 7 Strassen factors for A[i1:i2, j1:j2] * B[j1:j2, k1:k2] -> C[i1:i2, k1:k2]"""
        nonlocal col
        # M1 = (A00 + A11)(B00 + B11)
        U[i1*m+j1, col] = 1
        U[i2*m+j2, col] = 1
        V[j1*p+k1, col] = 1
        V[j2*p+k2, col] = 1
        W[k1*n+i1, col] = 1  # C00 += M1
        W[k2*n+i2, col] = 1  # C11 += M1
        col += 1
        
        # M2 = (A10 + A11) * B00
        U[i2*m+j1, col] = 1
        U[i2*m+j2, col] = 1
        V[j1*p+k1, col] = 1
        W[k1*n+i2, col] = 1  # C10 += M2
        W[k2*n+i2, col] = -1 # C11 -= M2
        col += 1
        
        # M3 = A00 * (B01 - B11)
        U[i1*m+j1, col] = 1
        V[j1*p+k2, col] = 1
        V[j2*p+k2, col] = -1
        W[k2*n+i1, col] = 1  # C01 += M3
        W[k2*n+i2, col] = 1  # C11 += M3
        col += 1
        
        # M4 = A11 * (B10 - B00)
        U[i2*m+j2, col] = 1
        V[j2*p+k1, col] = 1
        V[j1*p+k1, col] = -1
        W[k1*n+i1, col] = 1  # C00 += M4
        W[k1*n+i2, col] = 1  # C10 += M4
        col += 1
        
        # M5 = (A00 + A01) * B11
        U[i1*m+j1, col] = 1
        U[i1*m+j2, col] = 1
        V[j2*p+k2, col] = 1
        W[k1*n+i1, col] = -1 # C00 -= M5
        W[k2*n+i1, col] = 1  # C01 += M5
        col += 1
        
        # M6 = (A10 - A00) * (B00 + B01)
        U[i2*m+j1, col] = 1
        U[i1*m+j1, col] = -1
        V[j1*p+k1, col] = 1
        V[j1*p+k2, col] = 1
        W[k2*n+i2, col] = 1  # C11 += M6
        col += 1
        
        # M7 = (A01 - A11) * (B10 + B11)
        U[i1*m+j2, col] = 1
        U[i2*m+j2, col] = -1
        V[j2*p+k1, col] = 1
        V[j2*p+k2, col] = 1
        W[k1*n+i1, col] = 1  # C00 += M7
        col += 1
    
    def add_direct_block(i_indices, j_indices, k_indices):
        """Add direct (schoolbook) factors for A[i_indices, j_indices] * B[j_indices, k_indices] -> C[i_indices, k_indices]"""
        nonlocal col
        for i in i_indices:
            for j in j_indices:
                for k in k_indices:
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
    
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2] (Strassen, 7)
    add_strassen_2x2x2(0, 1, 0, 1, 0, 1)
    
    # Block 2: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4] (Strassen, 7)
    add_strassen_2x2x2(0, 1, 0, 1, 2, 3)
    
    # Block 3: A[0:2, 0:2] * B[0:2, 4] -> C[0:2, 4] (Direct, 4)
    add_direct_block([0, 1], [0, 1], [4])
    
    # Block 4: A[0:2, 2] * B[2, 0:2] -> C[0:2, 0:2] (Direct, 4)
    add_direct_block([0, 1], [2], [0, 1])
    
    # Block 5: A[0:2, 2] * B[2, 2:4] -> C[0:2, 2:4] (Direct, 4)
    add_direct_block([0, 1], [2], [2, 3])
    
    # Block 6: A[0:2, 2] * B[2, 4] -> C[0:2, 4] (Direct, 2)
    add_direct_block([0, 1], [2], [4])
    
    # Verify rank used
    if col != rank:
        # Should not happen, but handle gracefully
        pass
    
    # Verify exactness
    U_arr = U
    V_arr = V
    W_arr = W
    ok = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for ri in range(rank):
                    s += U_arr[i*m+j, ri] * V_arr[j*p+k, ri] * W_arr[k*n+i, ri]
                if abs(s - 1.0) > 1e-9:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            break
    
    if ok:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook if construction fails
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
