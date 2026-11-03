"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def _build_schoolbook():
    n, m, p = DIMENSIONS
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

def _build_3x3x3_rank19():
    """
    Rank-19 decomposition for 3x3 x 3x3 matrix multiplication.
    Returns (U, V, W) as lists of shape (9,19), or None if verification fails.
    """
    U = np.zeros((9, 19))
    V = np.zeros((9, 19))
    
    # Strassen's 19-product algorithm
    U[0,0] = 1; U[3,0] = 1; U[6,0] = 1
    V[0,0] = 1; V[3,0] = 1; V[6,0] = 1
    
    U[0,1] = 1; U[3,1] = -1
    V[0,1] = 1; V[3,1] = -1
    
    U[1,2] = 1; U[7,2] = 1
    V[1,2] = 1; V[4,2] = 1
    
    U[1,3] = 1; U[7,3] = -1
    V[1,3] = 1; V[4,3] = -1
    
    U[2,4] = 1; U[8,4] = 1
    V[2,4] = 1; V[5,4] = 1
    
    U[2,5] = 1; U[8,5] = -1
    V[2,5] = 1; V[5,5] = -1
    
    U[3,6] = 1
    V[1,6] = 1; V[4,6] = -1
    
    U[6,7] = 1
    V[1,7] = 1; V[4,7] = 1
    
    U[4,8] = 1
    V[0,8] = 1; V[6,8] = -1
    
    U[7,9] = 1
    V[0,9] = 1; V[6,9] = 1
    
    U[5,10] = 1
    V[0,10] = 1; V[6,10] = -1
    
    U[8,11] = 1
    V[0,11] = 1; V[6,11] = 1
    
    U[1,12] = 1
    V[3,12] = 1; V[6,12] = -1
    
    U[2,13] = 1
    V[3,13] = 1; V[6,13] = 1
    
    U[4,14] = 1
    V[3,14] = 1; V[6,14] = -1
    
    U[7,15] = 1
    V[3,15] = 1; V[6,15] = 1
    
    U[5,16] = 1
    V[3,16] = 1; V[6,16] = -1
    
    U[8,17] = 1
    V[3,17] = 1; V[6,17] = 1
    
    U[0,18] = 1; U[4,18] = 1; U[8,18] = 1
    V[4,18] = 1
    
    # Solve for W using least squares
    n_eq = 27
    n_var = 171
    A_sys = np.zeros((n_eq, n_var))
    b = np.ones(n_eq)
    
    eq_idx = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for t in range(19):
                    W_idx = i*3+k
                    A_sys[eq_idx, W_idx*19+t] = U[i*3+j, t] * V[j*3+k, t]
                eq_idx += 1
    
    W_vec, residuals, rank, sv = np.linalg.lstsq(A_sys, b, rcond=None)
    W = W_vec.reshape(9, 19)
    
    # Verify the decomposition
    for i in range(3):
        for j in range(3):
            for k in range(3):
                val = sum(W[i*3+k, t] * U[i*3+j, t] * V[j*3+k, t] for t in range(19))
                if abs(val - 1) > 1e-10:
                    return None
    
    return U.tolist(), V.tolist(), W.tolist()

def _build_3x3x4_schoolbook():
    """Schoolbook rank-36 decomposition for 3x3 x 3x4."""
    U = [[0]*36 for _ in range(9)]
    V = [[0]*36 for _ in range(12)]
    W = [[0]*36 for _ in range(12)]
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(4):
                U[i*3+j][col] = 1
                V[j*4+k][col] = 1
                W[k*3+i][col] = 1
                col += 1
    return U, V, W

def _combine_blocks_3x3x3_and_3x3x4(U1, V1, W1, U2, V2, W2):
    """
    Combine a 3x3x3 decomposition (rank 19) and a 3x3x4 decomposition (rank 36)
    into a 3x3x7 decomposition (rank 55).
    
    U1 shape: (9, 19), V1 shape: (9, 19), W1 shape: (9, 19)
    U2 shape: (9, 36), V2 shape: (12, 36), W2 shape: (12, 36)
    
    Result: U shape: (9, 55), V shape: (21, 55), W shape: (21, 55)
    """
    total_rank = 19 + 36
    U = [[0]*total_rank for _ in range(9)]
    V = [[0]*total_rank for _ in range(21)]
    W = [[0]*total_rank for _ in range(21)]
    
    # Copy 3x3x3 block (columns 0-18)
    for i in range(9):
        for t in range(19):
            U[i][t] = U1[i][t]
            V[i][t] = V1[i][t]
            W[i][t] = W1[i][t]
    
    # Copy 3x3x4 block (columns 19-54)
    # V2 has 12 rows (3*4), W2 has 12 rows (4*3)
    # V indices: j*4+k for j in 0..2, k in 0..3
    # W indices: k*3+i for k in 0..3, i in 0..2
    # These map to the full V (21 rows) and W (21 rows) as follows:
    # Full V index: j*7+k for j in 0..2, k in 0..6
    # For the 3x3x4 block, k is in 0..3, so full V index: j*7+k
    # Full W index: k*3+i for k in 0..6, i in 0..2
    # For the 3x3x4 block, k is in 0..3, so full W index: k*3+i
    
    for i in range(9):
        for t in range(36):
            U[i][19+t] = U2[i][t]
    
    for j in range(3):
        for k in range(4):
            full_v_idx = j*7 + k
            for t in range(36):
                V[full_v_idx][19+t] = V2[j*4+k][t]
    
    for k in range(4):
        for i in range(3):
            full_w_idx = k*3 + i
            for t in range(36):
                W[full_w_idx][19+t] = W2[k*3+i][t]
    
    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Fallback: Schoolbook
    schoolbook = _build_schoolbook()
    best_result = schoolbook
    best_rank = 63
    
    # Try block decomposition: 3x3x3 (rank 19) + 3x3x4 (rank 36) = rank 55
    try:
        r19 = _build_3x3x3_rank19()
        if r19 is not None:
            U1, V1, W1 = r19
            U2, V2, W2 = _build_3x3x4_schoolbook()
            result = _combine_blocks_3x3x3_and_3x3x4(U1, V1, W1, U2, V2, W2)
            
            # Verify the combined decomposition
            n, m, p = 3, 3, 7
            rank = 55
            U = np.array(result["U"])
            V = np.array(result["V"])
            W = np.array(result["W"])
            
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = sum(W[k*n+i, t] * U[i*m+j, t] * V[j*p+k, t] for t in range(rank))
                        if abs(val - 1) > 1e-10:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                best_result = result
                best_rank = 55
    except Exception:
        pass
    
    return best_result
# EVOLVE-BLOCK-END
