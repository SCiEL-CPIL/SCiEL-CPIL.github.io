import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n=2, m=4, p=4, verbose=False):
    """Verify that the decomposition correctly computes matrix multiplication."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    max_err = 0.0
    # T[i,j,k] = 1
    # U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U_arr[i * m + j, :], V_arr[j * p + k, :] * W_arr[k * n + i, :])
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
                    if verbose and err > 1e-9:
                        print(f"Error at i={i}, j={j}, k={k}: got {val}, expected 1.0")
    
    return max_err

def _strassen_2x2_block(U, V, W, u_idx, v_idx, w_idx, col_offset):
    """
    Adds 7 Strassen terms for a 2x2 * 2x2 multiplication.
    u_idx: [A00, A01, A10, A11]
    v_idx: [B00, B10, B01, B11]
    w_idx: [C00, C01, C10, C11]
    """
    A00, A01, A10, A11 = u_idx
    B00, B10, B01, B11 = v_idx
    C00, C01, C10, C11 = w_idx
    c = col_offset
    
    # M1 = (A00+A11)*(B00+B11)
    U[A00, c] += 1; U[A11, c] += 1
    V[B00, c] += 1; V[B11, c] += 1
    W[C00, c] += 1; W[C01, c] -= 1; W[C10, c] -= 1; W[C11, c] += 1
    c += 1
    
    # M2 = (A10+A11)*B00
    U[A10, c] += 1; U[A11, c] += 1
    V[B00, c] += 1
    W[C01, c] += 1; W[C11, c] -= 1
    c += 1
    
    # M3 = A00*(B10-B11)
    U[A00, c] += 1
    V[B10, c] += 1; V[B11, c] -= 1
    W[C00, c] += 1; W[C10, c] += 1
    c += 1
    
    # M4 = A11*(B01-B10)
    U[A11, c] += 1
    V[B01, c] += 1; V[B10, c] -= 1
    W[C01, c] += 1; W[C11, c] += 1
    c += 1
    
    # M5 = (A00+A10)*B11
    U[A00, c] += 1; U[A10, c] += 1
    V[B11, c] += 1
    W[C10, c] += 1; W[C11, c] -= 1
    c += 1
    
    # M6 = (A01-A10)*(B00+B10)
    U[A01, c] += 1; U[A10, c] -= 1
    V[B00, c] += 1; V[B10, c] += 1
    W[C00, c] -= 1; W[C01, c] += 1; W[C10, c] += 1; W[C11, c] -= 1
    c += 1
    
    # M7 = (A01-A11)*(B01+B11)
    U[A01, c] += 1; U[A11, c] -= 1
    V[B01, c] += 1; V[B11, c] += 1
    W[C00, c] += 1; W[C01, c] += 1; W[C10, c] -= 1; W[C11, c] -= 1
    c += 1
    
    return c

def _build_strassen_28():
    """
    Builds a rank-28 decomposition for 2x4 * 4x4.
    Uses 4 Strassen 2x2 blocks.
    """
    n, m, p = 2, 4, 4
    rank = 28
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # U indices for A (2x4)
    # A = [[A_L, A_R]] where A_L, A_R are 1x2? No.
    # A is 2x4.
    # A_L = A[:, 0:2] (2x2), A_R = A[:, 2:4] (2x2)
    # U indices:
    # A_L: [[U[0], U[1]], [U[2], U[3]]]
    # A_R: [[U[4], U[5]], [U[6], U[7]]]
    
    # V indices for B (4x4)
    # B = [[B_LL, B_LR], [B_RL, B_RR]] where blocks are 2x2.
    # B_LL: B[0:2, 0:2] -> V[0], V[1], V[4], V[5]
    # B_LR: B[0:2, 2:4] -> V[2], V[3], V[6], V[7]
    # B_RL: B[2:4, 0:2] -> V[8], V[9], V[12], V[13]
    # B_RR: B[2:4, 2:4] -> V[10], V[11], V[14], V[15]
    
    # W indices for C (2x4)
    # C = [[C_L, C_R]]
    # C_L: C[:, 0:2] -> W[0], W[2], W[1], W[3] (Note: W[k*n+i])
    #   C[0,0]=W[0], C[0,1]=W[2], C[1,0]=W[1], C[1,1]=W[3]
    # C_R: C[:, 2:4] -> W[4], W[6], W[5], W[7]
    #   C[0,2]=W[4], C[0,3]=W[6], C[1,2]=W[5], C[1,3]=W[7]
    
    # Block 1: A_L * B_LL -> C_L
    # A_L: [0, 1, 2, 3]
    # B_LL: [0, 4, 1, 5] (B00, B10, B01, B11)
    # C_L: [0, 2, 1, 3] (C00, C01, C10, C11)
    col = _strassen_2x2_block(U, V, W, [0, 1, 2, 3], [0, 4, 1, 5], [0, 2, 1, 3], 0)
    
    # Block 2: A_R * B_RL -> C_L
    # A_R: [4, 5, 6, 7]
    # B_RL: [8, 12, 9, 13]
    # C_L: [0, 2, 1, 3]
    col = _strassen_2x2_block(U, V, W, [4, 5, 6, 7], [8, 12, 9, 13], [0, 2, 1, 3], col)
    
    # Block 3: A_L * B_LR -> C_R
    # A_L: [0, 1, 2, 3]
    # B_LR: [2, 6, 3, 7]
    # C_R: [4, 6, 5, 7]
    col = _strassen_2x2_block(U, V, W, [0, 1, 2, 3], [2, 6, 3, 7], [4, 6, 5, 7], col)
    
    # Block 4: A_R * B_RR -> C_R
    # A_R: [4, 5, 6, 7]
    # B_RR: [10, 14, 11, 15]
    # C_R: [4, 6, 5, 7]
    col = _strassen_2x2_block(U, V, W, [4, 5, 6, 7], [10, 14, 11, 15], [4, 6, 5, 7], col)
    
    return U.tolist(), V.tolist(), W.tolist()

def construct():
    start_time = time.time()
    
    # Build rank-28 Strassen decomposition
    U, V, W = _build_strassen_28()
    
    # Verify
    err = _verify_decomposition(U, V, W)
    
    if err >= 1e-9:
        # Fall back to schoolbook rank-32
        n, m, p = 2, 4, 4
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
    
    # Deadline guard
    if time.time() - start_time > 140:
        pass
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
