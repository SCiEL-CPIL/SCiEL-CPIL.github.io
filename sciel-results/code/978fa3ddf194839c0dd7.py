import time
import numpy as np

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    def add_strassen_2x2x2(U, V, W, rank_start, i_off, j_off, k_off):
        """Add Strassen 7-term decomposition for 2x2 * 2x2 -> 2x2
        i in [i_off, i_off+2), j in [j_off, j_off+2), k in [k_off, k_off+2)
        """
        col = rank_start
        # Standard Strassen for 2x2*2x2:
        # P1 = (A11 + A22) * (B11 + B22)
        # P2 = (A21 + A22) * B12
        # P3 = A11 * (B21 - B11)
        # P4 = A22 * (B12 - B22)
        # P5 = (A11 + A12) * B22
        # P6 = (A21 - A11) * (B11 + B12)
        # P7 = (A12 - A22) * (B21 + B22)
        # C11 = P1 + P4 - P5 + P7
        # C12 = P2 + P5
        # C21 = P3 + P4
        # C22 = P1 - P2 + P3 + P6
        
        terms = [
            # P1: (A11+A22)(B11+B22)
            ({(i_off, j_off): 1, (i_off+1, j_off+1): 1}, 
             {(j_off, k_off): 1, (j_off+1, k_off+1): 1}, 
             {(k_off, i_off): 1, (k_off+1, i_off+1): 1}),
            # P2: (A21+A22)B12
            ({(i_off+1, j_off): 1, (i_off+1, j_off+1): 1}, 
             {(j_off, k_off+1): 1}, 
             {(k_off+1, i_off): 1}),
            # P3: A11(B21-B11)
            ({(i_off, j_off): 1}, 
             {(j_off+1, k_off): 1, (j_off, k_off): -1}, 
             {(k_off, i_off+1): 1, (k_off+1, i_off+1): 1}),
            # P4: A22(B12-B22)
            ({(i_off+1, j_off+1): 1}, 
             {(j_off, k_off+1): 1, (j_off+1, k_off+1): -1}, 
             {(k_off, i_off): 1, (k_off+1, i_off+1): 1}),
            # P5: (A11+A12)B22
            ({(i_off, j_off): 1, (i_off, j_off+1): 1}, 
             {(j_off+1, k_off+1): 1}, 
             {(k_off, i_off): -1, (k_off+1, i_off): 1}),
            # P6: (A21-A11)(B11+B12)
            ({(i_off+1, j_off): 1, (i_off, j_off): -1}, 
             {(j_off, k_off): 1, (j_off, k_off+1): 1}, 
             {(k_off+1, i_off+1): 1}),
            # P7: (A12-A22)(B21+B22)
            ({(i_off, j_off+1): 1, (i_off+1, j_off+1): -1}, 
             {(j_off+1, k_off): 1, (j_off+1, k_off+1): 1}, 
             {(k_off, i_off): 1}),
        ]
        
        for U_coeffs, V_coeffs, W_coeffs in terms:
            if time.time() > deadline:
                return col
            for (ii, jj), val in U_coeffs.items():
                U[ii*m + jj][col] += val
            for (jj, kk), val in V_coeffs.items():
                V[jj*p + kk][col] += val
            for (kk, ii), val in W_coeffs.items():
                W[kk*n + ii][col] += val
            col += 1
        
        return col
    
    def add_schoolbook_block(U, V, W, rank_start, i_range, j_range, k_range):
        """Add schoolbook terms for a block"""
        col = rank_start
        for i in i_range:
            for j in j_range:
                for k in k_range:
                    if time.time() > deadline:
                        return col
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1
        return col
    
    # Block decomposition:
    # 1. A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]: Strassen 2x2*2x2 = 7 terms
    # 2. A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]: Strassen 2x2*2x2 = 7 terms
    # 3. A[0:2, 0:2] * B[0:2, 4:7] -> C[0:2, 4:7]: Schoolbook 2*2*3 = 12 terms
    # 4. A[0:2, 2:3] * B[2:3, 0:7] -> C[0:2, 0:7]: Schoolbook 2*1*7 = 14 terms
    # 5. A[2:3, 0:3] * B[0:3, 0:7] -> C[2:3, 0:7]: Schoolbook 1*3*7 = 21 terms
    # Total: 7 + 7 + 12 + 14 + 21 = 61 terms
    
    rank = 61
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    col = add_strassen_2x2x2(U, V, W, col, 0, 0, 0)
    col = add_strassen_2x2x2(U, V, W, col, 0, 0, 2)
    col = add_schoolbook_block(U, V, W, col, range(0, 2), range(0, 2), range(4, 7))
    col = add_schoolbook_block(U, V, W, col, range(0, 2), range(2, 3), range(0, 7))
    col = add_schoolbook_block(U, V, W, col, range(2, 3), range(0, 3), range(0, 7))
    
    assert col == 61, f"Expected 61 terms, got {col}"
    
    # Verify
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_true = A_test @ B_test
    C_comp = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            val = 0.0
            for j in range(m):
                for r in range(rank):
                    val += U_arr[i*m + j, r] * A_test[i, j] * V_arr[j*p + k, r] * B_test[j, k] * W_arr[k*n + i, r]
            C_comp[i, k] = val
            
    if not np.allclose(C_comp, C_true, atol=1e-9):
        # Fallback to schoolbook 63
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
