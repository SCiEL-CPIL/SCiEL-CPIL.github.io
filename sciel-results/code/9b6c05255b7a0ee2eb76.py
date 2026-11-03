"""Rewritten program: construct a rank-55 exact decomposition for 2x3 * 3x10 matmul
using Strassen's algorithm on the 2x2 column-block and schoolbook for the 3rd column."""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]
N, M, P = DIMENSIONS

# Deadline guard
DEADLINE = 135.0
_start_time = time.time()

def _elapsed():
    return time.time() - _start_time

def _time_left():
    return DEADLINE - _elapsed()

def _verify(U, V, W, tol=1e-8):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor.
    
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Convention: W[k*n + i, r]
    Tensor T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    must equal 1 for all i, j, k.
    """
    rank = U.shape[1]
    U3 = U.reshape(N, M, rank)
    V3 = V.reshape(M, P, rank)
    W3 = W.reshape(P, N, rank)
    
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    expected = np.ones((N, M, P))
    return np.allclose(T, expected, atol=tol)

def _construct_rank55():
    """Construct the rank-55 decomposition using Strassen for 2x2 blocks + schoolbook for col 2.
    
    We split B into 5 blocks of size 3x2: B[:, 0:2], B[:, 2:4], ..., B[:, 8:10].
    For each block m (0..4):
      - Use Strassen's 7-multiplication algorithm for A[:, 0:2] * B[m, 0:2] (2x2 * 2x2)
      - Use schoolbook 4 multiplications for A[:, 2:3] * B[m, 2:3] (2x1 * 1x2)
    Total per block: 7 + 4 = 11 terms.
    Total rank: 5 * 11 = 55.
    """
    RANK = 55
    U = np.zeros((N * M, RANK))  # (6, 55)
    V = np.zeros((M * P, RANK))  # (30, 55)
    W = np.zeros((P * N, RANK))  # (20, 55)
    
    # Strassen's algorithm for C = A * B where A, B are 2x2:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B11)
    # M4 = A11(B01 - B10)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B10)
    # M7 = (A01 - A11)(B01 + B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For each block m, we need to assign 11 columns to U, V, W.
    # Block m handles columns 2m, 2m+1 of B (and thus of C).
    
    for m in range(5):
        base_col = m * 11  # Starting column index in U, V, W
        
        # --- Strassen part: A[:, 0:2] * B[m, 0:2] ---
        # Local indices: i in {0,1}, j_A in {0,1}, j_B in {0,1}, k_local in {0,1}
        # Global: i, j_A = j_A, j_B = j_B, k = 2m + k_local
        
        # M1 = (A00 + A11)(B00 + B11)
        # U: A00 and A11 contribute
        U[0*3+0, base_col+0] = 1.0  # A[0,0]
        U[1*3+1, base_col+0] = 1.0  # A[1,1]
        # V: B00 and B11 contribute (j_B=0,k=0 and j_B=1,k=1)
        V[0*10+(2*m+0), base_col+0] = 1.0  # B[0, 2m]
        V[1*10+(2*m+1), base_col+0] = 1.0  # B[1, 2m+1]
        # W: contributes to C[0,0] and C[1,1]
        W[(2*m+0)*2+0, base_col+0] = 1.0  # C[0, 2m]
        W[(2*m+1)*2+1, base_col+0] = 1.0  # C[1, 2m+1]
        
        # M2 = (A10 + A11)B00
        U[1*3+0, base_col+1] = 1.0  # A[1,0]
        U[1*3+1, base_col+1] = 1.0  # A[1,1]
        V[0*10+(2*m+0), base_col+1] = 1.0  # B[0, 2m]
        # W: contributes to C[1,0] and C[1,1]
        W[(2*m+0)*2+1, base_col+1] = 1.0  # C[1, 2m]
        W[(2*m+1)*2+1, base_col+1] = -1.0  # C[1, 2m+1]
        
        # M3 = A00(B10 - B11)
        U[0*3+0, base_col+2] = 1.0  # A[0,0]
        V[1*10+(2*m+0), base_col+2] = 1.0  # B[1, 2m]
        V[1*10+(2*m+1), base_col+2] = -1.0  # B[1, 2m+1]
        # W: contributes to C[0,1] and C[1,1]
        W[(2*m+1)*2+0, base_col+2] = 1.0  # C[0, 2m+1]
        W[(2*m+1)*2+1, base_col+2] = 1.0  # C[1, 2m+1]
        
        # M4 = A11(B01 - B10)
        U[1*3+1, base_col+3] = 1.0  # A[1,1]
        V[0*10+(2*m+1), base_col+3] = 1.0  # B[0, 2m+1]
        V[1*10+(2*m+0), base_col+3] = -1.0  # B[1, 2m]
        # W: contributes to C[0,0] and C[1,0]
        W[(2*m+0)*2+0, base_col+3] = 1.0  # C[0, 2m]
        W[(2*m+0)*2+1, base_col+3] = 1.0  # C[1, 2m]
        
        # M5 = (A00 + A01)B11
        U[0*3+0, base_col+4] = 1.0  # A[0,0]
        U[0*3+1, base_col+4] = 1.0  # A[0,1]
        V[1*10+(2*m+1), base_col+4] = 1.0  # B[1, 2m+1]
        # W: contributes to C[0,0] and C[0,1]
        W[(2*m+0)*2+0, base_col+4] = -1.0  # C[0, 2m]
        W[(2*m+1)*2+0, base_col+4] = 1.0  # C[0, 2m+1]
        
        # M6 = (A10 - A00)(B00 + B10)
        U[1*3+0, base_col+5] = 1.0  # A[1,0]
        U[0*3+0, base_col+5] = -1.0  # A[0,0]
        V[0*10+(2*m+0), base_col+5] = 1.0  # B[0, 2m]
        V[1*10+(2*m+0), base_col+5] = 1.0  # B[1, 2m]
        # W: contributes to C[1,1]
        W[(2*m+1)*2+1, base_col+5] = 1.0  # C[1, 2m+1]
        
        # M7 = (A01 - A11)(B01 + B11)
        U[0*3+1, base_col+6] = 1.0  # A[0,1]
        U[1*3+1, base_col+6] = -1.0  # A[1,1]
        V[0*10+(2*m+1), base_col+6] = 1.0  # B[0, 2m+1]
        V[1*10+(2*m+1), base_col+6] = 1.0  # B[1, 2m+1]
        # W: contributes to C[0,0]
        W[(2*m+0)*2+0, base_col+6] = 1.0  # C[0, 2m]
        
        # --- Schoolbook part: A[:, 2:3] * B[m, 2:3] ---
        # 4 terms: A[i,2] * B[2, 2m+k] for i in {0,1}, k in {0,1}
        
        # Term 0: A[0,2] * B[2, 2m] -> C[0, 2m]
        U[0*3+2, base_col+7] = 1.0
        V[2*10+(2*m+0), base_col+7] = 1.0
        W[(2*m+0)*2+0, base_col+7] = 1.0
        
        # Term 1: A[0,2] * B[2, 2m+1] -> C[0, 2m+1]
        U[0*3+2, base_col+8] = 1.0
        V[2*10+(2*m+1), base_col+8] = 1.0
        W[(2*m+1)*2+0, base_col+8] = 1.0
        
        # Term 2: A[1,2] * B[2, 2m] -> C[1, 2m]
        U[1*3+2, base_col+9] = 1.0
        V[2*10+(2*m+0), base_col+9] = 1.0
        W[(2*m+0)*2+1, base_col+9] = 1.0
        
        # Term 3: A[1,2] * B[2, 2m+1] -> C[1, 2m+1]
        U[1*3+2, base_col+10] = 1.0
        V[2*10+(2*m+1), base_col+10] = 1.0
        W[(2*m+1)*2+1, base_col+10] = 1.0
    
    return U, V, W

def construct():
    """Find a rank-minimal exact tensor decomposition for 2x3 * 3x10 matrix multiplication."""
    global _start_time
    _start_time = time.time()
    
    # Construct the rank-55 decomposition
    U, V, W = _construct_rank55()
    
    # Verify
    if _verify(U, V, W, tol=1e-8):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook rank 60
    RANK_SCHOOLBOOK = N * M * P
    U_fb = np.zeros((N * M, RANK_SCHOOLBOOK))
    V_fb = np.zeros((M * P, RANK_SCHOOLBOOK))
    W_fb = np.zeros((P * N, RANK_SCHOOLBOOK))
    col = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U_fb[i * M + j, col] = 1.0
                V_fb[j * P + k, col] = 1.0
                W_fb[k * N + i, col] = 1.0
                col += 1
    
    return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
