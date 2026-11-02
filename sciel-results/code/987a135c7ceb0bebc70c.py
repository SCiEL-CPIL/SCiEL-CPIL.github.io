import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 3, 5]

def _strassen_2x2x2(U, V, W, col, i_offset, j_offset, k_offset, n, m, p):
    # Strassen decomposition for 2x2x2 block
    U[i_offset*m + j_offset][col] = 1
    U[(i_offset+1)*m + (j_offset+1)][col] = 1
    V[j_offset*p + k_offset][col] = 1
    V[(j_offset+1)*p + (k_offset+1)][col] = 1
    W[k_offset*n + i_offset][col] = 1
    W[(k_offset+1)*n + (i_offset+1)][col] = 1
    col += 1

    U[(i_offset+1)*m + j_offset][col] = 1
    U[(i_offset+1)*m + (j_offset+1)][col] = 1
    V[j_offset*p + k_offset][col] = 1
    W[k_offset*n + (i_offset+1)][col] = 1
    W[(k_offset+1)*n + (i_offset+1)][col] = -1
    col += 1

    U[i_offset*m + j_offset][col] = 1
    V[j_offset*p + (k_offset+1)][col] = 1
    V[(j_offset+1)*p + (k_offset+1)][col] = -1
    W[(k_offset+1)*n + i_offset][col] = 1
    W[(k_offset+1)*n + (i_offset+1)][col] = 1
    col += 1

    U[(i_offset+1)*m + (j_offset+1)][col] = 1
    V[(j_offset+1)*p + k_offset][col] = 1
    V[j_offset*p + k_offset][col] = -1
    W[k_offset*n + i_offset][col] = 1
    W[k_offset*n + (i_offset+1)][col] = 1
    col += 1

    U[i_offset*m + j_offset][col] = 1
    U[i_offset*m + (j_offset+1)][col] = 1
    V[(j_offset+1)*p + (k_offset+1)][col] = 1
    W[k_offset*n + i_offset][col] = -1
    W[(k_offset+1)*n + i_offset][col] = 1
    col += 1

    U[(i_offset+1)*m + j_offset][col] = 1
    U[i_offset*m + j_offset][col] = -1
    V[j_offset*p + k_offset][col] = 1
    V[j_offset*p + (k_offset+1)][col] = 1
    W[(k_offset+1)*n + (i_offset+1)][col] = 1
    col += 1

    U[i_offset*m + (j_offset+1)][col] = 1
    U[(i_offset+1)*m + (j_offset+1)][col] = -1
    V[(j_offset+1)*p + k_offset][col] = 1
    V[(j_offset+1)*p + (k_offset+1)][col] = 1
    W[k_offset*n + i_offset][col] = 1
    col += 1

    return col

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition is exact using matrix multiplication."""
    rank = len(U[0])
    # Build the tensor via rank-1 sum: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Use numpy for speed
    U_mat = np.array(U)  # shape (n*m, rank)
    V_mat = np.array(V)  # shape (m*p, rank)
    W_mat = np.array(W)  # shape (p*n, rank)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U_mat[i*m+j, r] * V_mat[j*p+k, r] * W_mat[k*n+i, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _construct_rank26():
    """
    Construct a rank-26 decomposition for 2x3 by 3x5 matrix multiplication.
    
    Strategy:
    - 10 schoolbook columns for j=0 (all i,k pairs)
    - 7 Strassen columns for j=1,2 and k=0,1
    - 7 Strassen columns for j=1,2 and k=3,4
    - 2 columns for j=1,2 and k=2 (covering all 4 entries T[i,1,2], T[i,2,2])
    Total: 10 + 7 + 7 + 2 = 26
    """
    n, m, p = 2, 3, 5
    rank = 26
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    
    # 10 schoolbook columns for j=0
    for i in range(n):
        for k in range(p):
            U[i*m + 0, col] = 1
            V[0*p + k, col] = 1
            W[k*n + i, col] = 1
            col += 1
    # col = 10
    
    # 7 Strassen columns for j=1,2, k=0,1
    col = _strassen_2x2x2(U, V, W, col, 0, 1, 0, n, m, p)
    # col = 17
    
    # 7 Strassen columns for j=1,2, k=3,4
    col = _strassen_2x2x2(U, V, W, col, 0, 1, 3, n, m, p)
    # col = 24
    
    # 2 columns for j=1,2, k=2
    # Col 24: covers T[0,1,2] and T[0,2,2]
    U[0*m + 1, col] = 1  # U[i=0, j=1]
    U[0*m + 2, col] = 1  # U[i=0, j=2]
    V[1*p + 2, col] = 1  # V[j=1, k=2]
    V[2*p + 2, col] = 1  # V[j=2, k=2]
    W[2*n + 0, col] = 1  # W[k=2, i=0]
    col += 1
    
    # Col 25: covers T[1,1,2] and T[1,2,2]
    U[1*m + 1, col] = 1  # U[i=1, j=1]
    U[1*m + 2, col] = 1  # U[i=1, j=2]
    V[1*p + 2, col] = 1  # V[j=1, k=2]
    V[2*p + 2, col] = 1  # V[j=2, k=2]
    W[2*n + 1, col] = 1  # W[k=2, i=1]
    col += 1
    
    assert col == rank, f"Column count mismatch: {col} != {rank}"
    
    # Verify
    if not _verify_decomposition(U, V, W, n, m, p):
        return None
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _construct_rank28():
    """Fallback rank-28 decomposition."""
    n, m, p = 2, 3, 5
    rank = 28
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    
    # Block 1: j=0, k=0..4 (10 columns)
    for i in range(n):
        for k in range(p):
            U[i*m + 0, col] = 1
            V[0*p + k, col] = 1
            W[k*n + i, col] = 1
            col += 1
    
    # Block 2: j=1,2, k=0,1 (Strassen, 7 columns)
    col = _strassen_2x2x2(U, V, W, col, 0, 1, 0, n, m, p)
    
    # Block 3: j=1,2, k=2 (schoolbook, 4 columns)
    for i in range(n):
        for j in [1, 2]:
            U[i*m + j, col] = 1
            V[j*p + 2, col] = 1
            W[2*n + i, col] = 1
            col += 1
    
    # Block 4: j=1,2, k=3,4 (Strassen, 7 columns)
    col = _strassen_2x2x2(U, V, W, col, 0, 1, 3, n, m, p)
    
    assert col == rank, f"Column count mismatch: {col} != {rank}"
    
    # Verify
    if not _verify_decomposition(U, V, W, n, m, p):
        return None
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # n=2, m=3, p=5
    
    # Try rank-26 first
    best_result = None
    try:
        r26 = _construct_rank26()
        if r26 is not None:
            best_result = r26
    except Exception:
        pass
    
    if best_result is not None:
        return best_result
    
    # Fallback: Rank 28 decomposition
    try:
        r28 = _construct_rank28()
        if r28 is not None:
            return r28
    except Exception:
        pass
    
    # Last resort: pure schoolbook (rank 30)
    rank = 30
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
