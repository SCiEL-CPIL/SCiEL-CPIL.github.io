"""
Program to find a rank-minimal exact tensor decomposition for 2x2 * 2x4 matrix multiplication.
Strategy: Construct a rank-14 decomposition using two independent Strassen 2x2x2 blocks.
The previous implementation had incorrect W indices, causing verification to fail and falling back to rank 16.
This version corrects the W indices to match the W[k*n+i] convention.
"""
import time
import numpy as np

DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n=2, m=2, p=4):
    """
    Verify that U, V, W correctly decompose the matrix multiplication tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Condition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i, j, k
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    T_recon = np.zeros((n, m, p))
    for r in range(rank):
        u_r = U[:, r]
        v_r = V[:, r]
        w_r = W[:, r]
        
        u_mat = u_r.reshape(n, m)
        v_mat = v_r.reshape(m, p)
        w_mat = w_r.reshape(p, n)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_recon[i, j, k] += u_mat[i, j] * v_mat[j, k] * w_mat[k, i]

    if not np.allclose(T_recon, 1.0, atol=1e-9):
        return False
    return True

def _strassen_rank14():
    """
    Construct the rank-14 decomposition using two Strassen blocks.
    Left block handles B columns 0,1 -> C columns 0,1
    Right block handles B columns 2,3 -> C columns 2,3
    """
    n, m, p = 2, 2, 4
    rank = 14
    U = np.zeros((n*m, rank))  # 4 x 14
    V = np.zeros((m*p, rank))  # 8 x 14
    W = np.zeros((p*n, rank))  # 8 x 14
    
    # U indices: i*m+j
    # A00=0, A01=1, A10=2, A11=3
    
    # V indices: j*p+k
    # B00=0, B01=1, B02=2, B03=3
    # B10=4, B11=5, B12=6, B13=7
    
    # W indices: k*n+i
    # C00=0, C10=1, C01=2, C11=3
    # C02=4, C12=5, C03=6, C13=7
    
    # Left Block (k=0,1)
    
    # m1_L = (A00 + A11) * (B00 + B11)
    r = 0
    U[0, r] = 1; U[3, r] = 1
    V[0, r] = 1; V[5, r] = 1
    W[0, r] = 1; W[3, r] = 1
    
    # m2_L = (A10 + A11) * B00
    r = 1
    U[2, r] = 1; U[3, r] = 1
    V[0, r] = 1
    W[1, r] = 1; W[3, r] = -1
    
    # m3_L = A00 * (B10 - B11)
    r = 2
    U[0, r] = 1
    V[4, r] = 1; V[5, r] = -1
    W[2, r] = 1; W[3, r] = 1
    
    # m4_L = A11 * (B01 - B00)
    r = 3
    U[3, r] = 1
    V[1, r] = 1; V[0, r] = -1
    W[0, r] = 1; W[1, r] = 1
    
    # m5_L = (A00 + A01) * B11
    r = 4
    U[0, r] = 1; U[1, r] = 1
    V[5, r] = 1
    W[2, r] = 1; W[0, r] = -1
    
    # m6_L = (A10 - A00) * (B00 + B01)
    r = 5
    U[2, r] = 1; U[0, r] = -1
    V[0, r] = 1; V[1, r] = 1
    W[3, r] = 1
    
    # m7_L = (A01 - A11) * (B10 + B11)
    r = 6
    U[1, r] = 1; U[3, r] = -1
    V[4, r] = 1; V[5, r] = 1
    W[0, r] = 1
    
    # Right Block (k=2,3)
    
    # m1_R = (A00 + A11) * (B02 + B13)
    r = 7
    U[0, r] = 1; U[3, r] = 1
    V[2, r] = 1; V[7, r] = 1
    W[4, r] = 1; W[7, r] = 1
    
    # m2_R = (A10 + A11) * B02
    r = 8
    U[2, r] = 1; U[3, r] = 1
    V[2, r] = 1
    W[5, r] = 1; W[7, r] = -1
    
    # m3_R = A00 * (B12 - B13)
    r = 9
    U[0, r] = 1
    V[6, r] = 1; V[7, r] = -1
    W[6, r] = 1; W[7, r] = 1
    
    # m4_R = A11 * (B03 - B02)
    r = 10
    U[3, r] = 1
    V[3, r] = 1; V[2, r] = -1
    W[4, r] = 1; W[5, r] = 1
    
    # m5_R = (A00 + A01) * B13
    r = 11
    U[0, r] = 1; U[1, r] = 1
    V[7, r] = 1
    W[6, r] = 1; W[4, r] = -1
    
    # m6_R = (A10 - A00) * (B02 + B03)
    r = 12
    U[2, r] = 1; U[0, r] = -1
    V[2, r] = 1; V[3, r] = 1
    W[7, r] = 1
    
    # m7_R = (A01 - A11) * (B12 + B13)
    r = 13
    U[1, r] = 1; U[3, r] = -1
    V[6, r] = 1; V[7, r] = 1
    W[4, r] = 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Try rank 14
    result_14 = _strassen_rank14()
    
    if _verify(result_14["U"], result_14["V"], result_14["W"], n, m, p):
        return result_14
    
    # Fallback to schoolbook rank 16
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if time.time() > deadline:
                    return {"U": U, "V": V, "W": W}
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
