"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _get_strassen_2x2():
    """
    Returns U, V, W matrices for the rank-7 Strassen decomposition of 2x2x2 matrix multiplication.
    Shapes: (4, 7) each.
    Indexing:
    U: row i*2+j
    V: row j*2+k
    W: row k*2+i
    """
    # Strassen 7 products:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # M1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = -1
    
    # M2
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    # M3
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # M5
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # M6
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _verify_decomp(U, V, W, n, m, p):
    """
    Verify decomposition.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Must equal 1 for all i, j, k
    
    # Vectorized check
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U_reshaped = U.reshape(n, m, rank)
    V_reshaped = V.reshape(m, p, rank)
    W_reshaped = W.reshape(p, n, rank)
    
    # We need sum_c U[i,j,c] V[j,k,c] W[k,i,c]
    # This is a contraction.
    # Let's just loop for simplicity and safety, n=m=p=4 is small.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 4
    start_time = time.time()
    
    # Use Kronecker product of Strassen 2x2 with itself to get 4x4 rank 49.
    # U_4x4 = kron(U_2x2, U_2x2) with appropriate reshaping?
    # Actually, the tensor for 4x4x4 matmul is the Kronecker product of the 2x2x2 tensor with itself.
    # T_4 = T_2 \otimes T_2
    # If T_2 = sum_c u_c \otimes v_c \otimes w_c
    # Then T_4 = sum_{c1, c2} (u_{c1} \otimes u_{c2}) \otimes (v_{c1} \otimes v_{c2}) \otimes (w_{c1} \otimes w_{c2})
    # The rank is 7 * 7 = 49.
    
    U2, V2, W2 = _get_strassen_2x2()
    
    # Construct U4, V4, W4
    # U4 has shape (16, 49).
    # Index of U4 is (i, j) where i,j in 0..3.
    # i = i1*2 + i2, j = j1*2 + j2
    # U4[i, j, c] where c = c1*7 + c2
    # U4[i1*2+i2, j1*2+j2, c1*7+c2] = U2[i1*2+j1, c1] * U2[i2*2+j2, c2]
    
    rank4 = 49
    U4 = np.zeros((n*m, rank4))
    V4 = np.zeros((m*p, rank4))
    W4 = np.zeros((p*n, rank4))
    
    # Populate U4
    for i1 in range(2):
        for j1 in range(2):
            for i2 in range(2):
                for j2 in range(2):
                    i = i1 * 2 + i2
                    j = j1 * 2 + j2
                    row_idx = i * m + j
                    for c1 in range(7):
                        for c2 in range(7):
                            c = c1 * 7 + c2
                            U4[row_idx, c] = U2[i1 * 2 + j1, c1] * U2[i2 * 2 + j2, c2]
    
    # Populate V4
    # V4[j, k, c] = V2[j1*2+k1, c1] * V2[j2*2+k2, c2]
    # j = j1*2 + j2, k = k1*2 + k2
    for j1 in range(2):
        for k1 in range(2):
            for j2 in range(2):
                for k2 in range(2):
                    j = j1 * 2 + j2
                    k = k1 * 2 + k2
                    row_idx = j * p + k
                    for c1 in range(7):
                        for c2 in range(7):
                            c = c1 * 7 + c2
                            V4[row_idx, c] = V2[j1 * 2 + k1, c1] * V2[j2 * 2 + k2, c2]
    
    # Populate W4
    # W4[k, i, c] = W2[k1*2+i1, c1] * W2[k2*2+i2, c2]
    # k = k1*2 + k2, i = i1*2 + i2
    for k1 in range(2):
        for i1 in range(2):
            for k2 in range(2):
                for i2 in range(2):
                    k = k1 * 2 + k2
                    i = i1 * 2 + i2
                    row_idx = k * n + i
                    for c1 in range(7):
                        for c2 in range(7):
                            c = c1 * 7 + c2
                            W4[row_idx, c] = W2[k1 * 2 + i1, c1] * W2[k2 * 2 + i2, c2]
    
    # Verify
    if _verify_decomp(U4, V4, W4, n, m, p):
        # Convert to lists of lists for return interface
        return {
            "U": U4.tolist(),
            "V": V4.tolist(),
            "W": W4.tolist()
        }
    else:
        # Fallback to schoolbook rank 64
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
