"""Matrix multiplication 2x3x6 - rank-minimal decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 3, 6]
_DEADLINE = 145.0  # safety margin under 150s kill
_T0 = time.time()

def _remaining():
    return _DEADLINE - (time.time() - _T0)

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-33 decomposition for 2x3 x 3x6 matrix multiplication.
    
    Strategy: Split A (2x3) into A1 (2x1) and A2 (2x2).
    Split B (3x6) into B1 (1x6) and B2 (2x6).
    C = A1*B1 + A2*B2
    A1*B1: 2x1x6, rank 12 (schoolbook)
    A2*B2: 2x2x6, split into three 2x2x2 blocks, each rank 7 (Strassen), total 21
    Total rank: 33
    
    Index conventions:
    U[i*m + j, r] for A[i][j]
    V[j*p + k, r] for B[j][k]
    W[k*n + i, r] for C[i][k]
    """
    n, m, p = 2, 3, 6
    rank = 33
    
    U = [[0.0] * rank for _ in range(n * m)]  # 6 x 33
    V = [[0.0] * rank for _ in range(m * p)]  # 18 x 33
    W = [[0.0] * rank for _ in range(p * n)]  # 12 x 33
    
    col = 0
    
    # Part 1: A1 (2x1) * B1 (1x6) => 2x6 output
    # A1 has elements A[0][0] at row 0, A[1][0] at row 1
    # B1 has elements B[0][k] for k=0..5
    # C[i][k] = A[i][0] * B[0][k]
    # U index: i*m + 0 = i*3
    # V index: 0*p + k = k
    # W index: k*n + i = k*2 + i
    for k in range(p):
        for i in range(n):
            U[i * m + 0][col] = 1.0
            V[0 * p + k][col] = 1.0
            W[k * n + i][col] = 1.0
            col += 1
    # col is now 12
    
    # Part 2: A2 (2x2) * B2 (2x6) => 2x6 output
    # A2 elements: A[0][1], A[0][2], A[1][1], A[1][2]
    # B2 elements: B[1][k], B[2][k] for k=0..5
    # C[i][k] = A[i][1]*B[1][k] + A[i][2]*B[2][k]
    # Split B2 into three 2x2 blocks: B2_0 (cols 0,1), B2_1 (cols 2,3), B2_2 (cols 4,5)
    # For each block, use Strassen with 7 multiplications
    
    # Strassen for 2x2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    for t in range(3):
        # B2 block t has columns 2t, 2t+1
        # A2 indices: (i, j) where i=0,1 and j=1,2
        # In U: row = i*m + j = i*3 + j
        # B2_block indices: (j, k') where j=1,2 and k'=0,1
        # In V: row = j*p + (2t + k') = j*6 + 2t + k'
        # C indices: (i, k) where i=0,1 and k=2t, 2t+1
        # In W: row = k*n + i = (2t+k')*2 + i
        
        # M1 = (A00+A11)(B00+B11)
        # A00 = A[0][1], A11 = A[1][2]
        # B00 = B[1][2t], B11 = B[2][2t+1]
        U[0 * m + 1][col] = 1.0  # A[0][1]
        U[1 * m + 2][col] = 1.0  # A[1][2]
        V[1 * p + 2*t][col] = 1.0  # B[1][2t]
        V[2 * p + 2*t + 1][col] = 1.0  # B[2][2t+1]
        # M1 contributes to C00 and C11
        W[2*t * n + 0][col] = 1.0  # C00
        W[(2*t + 1) * n + 1][col] = 1.0  # C11
        col += 1
        
        # M2 = (A10+A11)B00
        # A10 = A[1][1], A11 = A[1][2]
        # B00 = B[1][2t]
        U[1 * m + 1][col] = 1.0  # A[1][1]
        U[1 * m + 2][col] = 1.0  # A[1][2]
        V[1 * p + 2*t][col] = 1.0  # B[1][2t]
        # M2 contributes to C10 and C11 (with -1)
        W[2*t * n + 1][col] = 1.0  # C10
        W[(2*t + 1) * n + 1][col] = -1.0  # C11
        col += 1
        
        # M3 = A00(B01-B11)
        # A00 = A[0][1]
        # B01 = B[1][2t+1], B11 = B[2][2t+1]
        U[0 * m + 1][col] = 1.0  # A[0][1]
        V[1 * p + 2*t + 1][col] = 1.0  # B[1][2t+1]
        V[2 * p + 2*t + 1][col] = -1.0  # B[2][2t+1]
        # M3 contributes to C01 and C11
        W[(2*t + 1) * n + 0][col] = 1.0  # C01
        W[(2*t + 1) * n + 1][col] = 1.0  # C11
        col += 1
        
        # M4 = A11(B10-B00)
        # A11 = A[1][2]
        # B10 = B[2][2t], B00 = B[1][2t]
        U[1 * m + 2][col] = 1.0  # A[1][2]
        V[2 * p + 2*t][col] = 1.0  # B[2][2t]
        V[1 * p + 2*t][col] = -1.0  # B[1][2t]
        # M4 contributes to C00 and C10
        W[2*t * n + 0][col] = 1.0  # C00
        W[2*t * n + 1][col] = 1.0  # C10
        col += 1
        
        # M5 = (A00+A01)B11
        # A00 = A[0][1], A01 = A[0][2]
        # B11 = B[2][2t+1]
        U[0 * m + 1][col] = 1.0  # A[0][1]
        U[0 * m + 2][col] = 1.0  # A[0][2]
        V[2 * p + 2*t + 1][col] = 1.0  # B[2][2t+1]
        # M5 contributes to C00 (with -1) and C01
        W[2*t * n + 0][col] = -1.0  # C00
        W[(2*t + 1) * n + 0][col] = 1.0  # C01
        col += 1
        
        # M6 = (A10-A00)(B00+B01)
        # A10 = A[1][1], A00 = A[0][1]
        # B00 = B[1][2t], B01 = B[1][2t+1]
        U[1 * m + 1][col] = 1.0  # A[1][1]
        U[0 * m + 1][col] = -1.0  # A[0][1]
        V[1 * p + 2*t][col] = 1.0  # B[1][2t]
        V[1 * p + 2*t + 1][col] = 1.0  # B[1][2t+1]
        # M6 contributes to C11
        W[(2*t + 1) * n + 1][col] = 1.0  # C11
        col += 1
        
        # M7 = (A01-A11)(B10+B11)
        # A01 = A[0][2], A11 = A[1][2]
        # B10 = B[2][2t], B11 = B[2][2t+1]
        U[0 * m + 2][col] = 1.0  # A[0][2]
        U[1 * m + 2][col] = -1.0  # A[1][2]
        V[2 * p + 2*t][col] = 1.0  # B[2][2t]
        V[2 * p + 2*t + 1][col] = 1.0  # B[2][2t+1]
        # M7 contributes to C00
        W[2*t * n + 0][col] = 1.0  # C00
        col += 1
    
    # col should be 12 + 3*7 = 33
    
    # Verify the decomposition
    U_arr = np.array(U)
    V_arr = np.array(V)
    W_arr = np.array(W)
    
    # Build the tensor from the decomposition
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    T_decomp = np.zeros((n, m, p))
    for r in range(rank):
        T_decomp += np.einsum('i,j,k->ijk', U_arr[:, r], V_arr[:, r], W_arr[:, r])
    
    # The true multiplication tensor has T[i,j,k] = 1 for all i,j,k
    T_true = np.ones((n, m, p))
    
    if not np.allclose(T_decomp, T_true, atol=1e-9):
        # Fallback to schoolbook if verification fails
        rank_fb = n * m * p
        U_fb = [[0.0] * rank_fb for _ in range(n * m)]
        V_fb = [[0.0] * rank_fb for _ in range(m * p)]
        W_fb = [[0.0] * rank_fb for _ in range(p * n)]
        col_fb = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j][col_fb] = 1.0
                    V_fb[j * p + k][col_fb] = 1.0
                    W_fb[k * n + i][col_fb] = 1.0
                    col_fb += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
