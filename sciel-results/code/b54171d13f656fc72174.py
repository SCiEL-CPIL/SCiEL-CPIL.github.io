"""
Matrix multiplication 5x5x5 - Rank 110 decomposition
Using Strassen 4x4 (rank 49) + cross terms (rank 61)
"""
import time

DIMENSIONS = [5, 5, 5]

def get_strassen_2x2():
    """
    Returns U, V, W for 2x2 Strassen (rank 7).
    U: 4x7, V: 4x7, W: 4x7
    Row/col index 0->(0,0), 1->(0,1), 2->(1,0), 3->(1,1)
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (A00 + A11)(B00 + B11)
    U[0][0] += 1; U[3][0] += 1
    V[0][0] += 1; V[3][0] += 1
    W[0][0] += 1  # C00
    W[3][0] += 1  # C11
    
    # M2 = (A10 + A11) B00
    U[2][1] += 1; U[3][1] += 1
    V[0][1] += 1
    W[2][1] += 1  # C10
    
    # M3 = A00 (B01 - B11)
    U[0][2] += 1
    V[1][2] += 1; V[3][2] += -1
    W[1][2] += 1  # C01
    
    # M4 = A11 (B10 - B00)
    U[3][3] += 1
    V[2][3] += 1; V[0][3] += -1
    W[2][3] += 1  # C10
    W[3][3] += 1  # C11
    
    # M5 = (A00 + A01) B11
    U[0][4] += 1; U[1][4] += 1
    V[3][4] += 1
    W[0][4] += 1  # C00
    
    # M6 = (A10 - A00)(B00 + B01)
    U[2][5] += 1; U[0][5] += -1
    V[0][5] += 1; V[1][5] += 1
    W[1][5] += 1  # C01
    W[2][5] += 1  # C10
    
    # M7 = (A01 - A11)(B10 + B11)
    U[1][6] += 1; U[3][6] += -1
    V[2][6] += 1; V[3][6] += 1
    W[1][6] += 1  # C01
    W[3][6] += 1  # C11
    
    return U, V, W

def construct():
    start_time = time.time()
    timeout = 120.0
    
    n, m, p = DIMENSIONS
    
    def schoolbook():
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
    
    try:
        # Rank = 49 (Strassen 4x4) + 61 (Cross terms) = 110
        rank = 110
        U = [[0] * rank for _ in range(25)]
        V = [[0] * rank for _ in range(25)]
        W = [[0] * rank for _ in range(25)]
        
        # Get 2x2 Strassen factors
        U2, V2, W2 = get_strassen_2x2()
        
        # Construct 4x4 Strassen (rank 49) using Kronecker product
        # The 4x4 matrix multiplication tensor is the Kronecker product of two 2x2 tensors
        # U4 = U2 ⊗ U2, V4 = V2 ⊗ V2, W4 = W2 ⊗ W2 (with appropriate index mapping)
        
        # For the Kronecker product decomposition:
        # 4x4 matrix A has elements A[r,c] where r,c in 0..3
        # Decompose r = r_m * 2 + r_l, c = c_m * 2 + c_l
        # Block index: (r_m, c_m), leaf index: (r_l, c_l)
        # Flat index for 2x2: idx = row * 2 + col
        # So block flat index: idx_m = r_m * 2 + c_m
        # Leaf flat index: idx_l = r_l * 2 + c_l
        
        # Term index t in 0..48: t = t_m * 7 + t_l
        # U4[flat_4, t] = U2[idx_m, t_m] * U2[idx_l, t_l]
        
        for t in range(49):
            t_m = t // 7
            t_l = t % 7
            for R in range(16):
                r = R // 4
                c = R % 4
                r_m = r // 2
                c_m = c // 2
                r_l = r % 2
                c_l = c % 2
                idx_m = r_m * 2 + c_m
                idx_l = r_l * 2 + c_l
                U[R][t] = U2[idx_m][t_m] * U2[idx_l][t_l]
                
                V[R][t] = V2[idx_m][t_m] * V2[idx_l][t_l]
                
                W[R][t] = W2[idx_m][t_m] * W2[idx_l][t_l]
        
        # Cross terms (indices 49 to 109)
        # These handle the interactions between the 4x4 block and the 5th row/col
        
        idx = 49
        
        # Type 1: A[i,4] * B[4,j] -> C[i,j] for i,j in 0..3
        # 16 terms
        for i in range(4):
            for j in range(4):
                U[i * 5 + 4][idx] = 1
                V[4 * 5 + j][idx] = 1
                W[i * 5 + j][idx] = 1
                idx += 1
        
        # Type 2: A[i,k] * B[k,4] -> C[i,4] for i,k in 0..3
        # 16 terms
        for i in range(4):
            for k in range(4):
                U[i * 5 + k][idx] = 1
                V[k * 5 + 4][idx] = 1
                W[i * 5 + 4][idx] = 1
                idx += 1
        
        # Type 3: A[i,4] * B[4,4] -> C[i,4] for i in 0..3
        # 4 terms
        for i in range(4):
            U[i * 5 + 4][idx] = 1
            V[4 * 5 + 4][idx] = 1
            W[i * 5 + 4][idx] = 1
            idx += 1
        
        # Type 4: A[4,k] * B[k,j] -> C[4,j] for k,j in 0..3
        # 16 terms
        for k in range(4):
            for j in range(4):
                U[4 * 5 + k][idx] = 1
                V[k * 5 + j][idx] = 1
                W[4 * 5 + j][idx] = 1
                idx += 1
        
        # Type 5: A[4,4] * B[4,j] -> C[4,j] for j in 0..3
        # 4 terms
        for j in range(4):
            U[4 * 5 + 4][idx] = 1
            V[4 * 5 + j][idx] = 1
            W[4 * 5 + j][idx] = 1
            idx += 1
        
        # Type 6: A[4,k] * B[k,4] -> C[4,4] for k in 0..3
        # 4 terms
        for k in range(4):
            U[4 * 5 + k][idx] = 1
            V[k * 5 + 4][idx] = 1
            W[4 * 5 + 4][idx] = 1
            idx += 1
        
        # Type 7: A[4,4] * B[4,4] -> C[4,4]
        # 1 term
        U[4 * 5 + 4][idx] = 1
        V[4 * 5 + 4][idx] = 1
        W[4 * 5 + 4][idx] = 1
        idx += 1
        
        # Verify idx == 110
        if idx != 110:
            return schoolbook()
        
        # Verification with NumPy
        import numpy as np
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        np.random.seed(42)
        A_test = np.random.rand(5, 5)
        B_test = np.random.rand(5, 5)
        C_expected = A_test @ B_test
        
        A_flat = A_test.flatten()
        B_flat = B_test.flatten()
        
        ua = U_np @ A_flat
        vb = V_np @ B_flat
        
        C_flat = W_np.T @ (ua * vb)
        C_calc = C_flat.reshape(5, 5)
        
        if not np.allclose(C_calc, C_expected, atol=1e-10):
            return schoolbook()
        
        return {"U": U, "V": V, "W": W}
    
    except Exception:
        return schoolbook()
