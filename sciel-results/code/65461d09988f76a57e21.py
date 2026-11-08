"""Program for matrix multiplication 3x3x4 tensor decomposition."""
DIMENSIONS = [3, 3, 4]

import time
import numpy as np

def _sparsify(U, V, W, max_iter=50, deadline=None):
    """
    Attempt to reduce rank by identifying redundant terms.
    Note: Simple linear dependence of stacked columns is NOT sufficient for tensor rank reduction.
    This function currently only removes zero columns to avoid invalid deletions.
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    rank = U.shape[1]
    for iteration in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        removed = False
        for c in range(rank):
            # Check if column c is zero in any matrix
            if np.allclose(U[:, c], 0) or np.allclose(V[:, c], 0) or np.allclose(W[:, c], 0):
                U = np.delete(U, c, axis=1)
                V = np.delete(V, c, axis=1)
                W = np.delete(W, c, axis=1)
                rank -= 1
                removed = True
                break
        
        if not removed:
            break
    
    return U, V, W, rank

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    n, m, p = DIMENSIONS  # n=3, m=3, p=4
    rank = 34
    U = [[0] * rank for _ in range(n * m)]  # 9 x 34
    V = [[0] * rank for _ in range(m * p)]  # 12 x 34
    W = [[0] * rank for _ in range(p * n)]  # 12 x 34
    
    col = 0
    
    # Helper to add a term
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        nonlocal col
        U[u_idx][col] += u_val
        V[v_idx][col] += v_val
        W[w_idx][col] += w_val
        col += 1

    # Strassen Left (A11 * B1L)
    # A11 indices: 0,1,3,4
    # B1L indices: 0,1,4,5
    # C1L indices: 0,3,1,4
    
    # P1
    U[0][col] = 1; U[4][col] = 1
    V[0][col] = 1; V[5][col] = 1
    W[0][col] = 1; W[4][col] = 1
    col += 1
    
    # P2
    U[3][col] = 1; U[4][col] = 1
    V[0][col] = 1
    W[1][col] = 1; W[4][col] = -1
    col += 1
    
    # P3
    U[0][col] = 1
    V[1][col] = 1; V[5][col] = -1
    W[3][col] = 1; W[4][col] = 1
    col += 1
    
    # P4
    U[4][col] = 1
    V[4][col] = 1; V[0][col] = -1
    W[0][col] = 1; W[1][col] = 1
    col += 1
    
    # P5
    U[0][col] = 1; U[1][col] = 1
    V[5][col] = 1
    W[0][col] = -1; W[3][col] = 1
    col += 1
    
    # P6
    U[3][col] = 1; U[0][col] = -1
    V[0][col] = 1; V[1][col] = 1
    W[4][col] = 1
    col += 1
    
    # P7
    U[1][col] = 1; U[4][col] = -1
    V[4][col] = 1; V[5][col] = 1
    W[0][col] = 1
    col += 1
    
    # A12 * B2L
    # A12 indices: 2, 5
    # B2L indices: 8, 9
    # C1L contribution:
    # C[0,0]->0, C[0,1]->3, C[1,0]->1, C[1,1]->4
    
    # T1: A[0,2] B[2,0] C[0,0]
    U[2][col] = 1
    V[8][col] = 1
    W[0][col] = 1
    col += 1
    
    # T2: A[0,2] B[2,1] C[0,1]
    U[2][col] = 1
    V[9][col] = 1
    W[3][col] = 1
    col += 1
    
    # T3: A[1,2] B[2,0] C[1,0]
    U[5][col] = 1
    V[8][col] = 1
    W[1][col] = 1
    col += 1
    
    # T4: A[1,2] B[2,1] C[1,1]
    U[5][col] = 1
    V[9][col] = 1
    W[4][col] = 1
    col += 1
    
    # A21 * B1L
    # A21 indices: 6, 7
    # B1L indices: 0,1,4,5
    # C2L contribution:
    # C[2,0]->2, C[2,1]->5
    
    # T1: A[2,0] B[0,0] C[2,0]
    U[6][col] = 1
    V[0][col] = 1
    W[2][col] = 1
    col += 1
    
    # T2: A[2,0] B[0,1] C[2,1]
    U[6][col] = 1
    V[1][col] = 1
    W[5][col] = 1
    col += 1
    
    # T3: A[2,1] B[1,0] C[2,0]
    U[7][col] = 1
    V[4][col] = 1
    W[2][col] = 1
    col += 1
    
    # T4: A[2,1] B[1,1] C[2,1]
    U[7][col] = 1
    V[5][col] = 1
    W[5][col] = 1
    col += 1
    
    # A22 * B2L
    # A22 index: 8
    # B2L indices: 8, 9
    # C2L contribution:
    # C[2,0]->2, C[2,1]->5
    
    # T1: A[2,2] B[2,0] C[2,0]
    U[8][col] = 1
    V[8][col] = 1
    W[2][col] = 1
    col += 1
    
    # T2: A[2,2] B[2,1] C[2,1]
    U[8][col] = 1
    V[9][col] = 1
    W[5][col] = 1
    col += 1
    
    # Strassen Right (A11 * B1R)
    # A11 indices: 0,1,3,4
    # B1R indices: 2,3,6,7
    # C1R indices: 6,9,7,10
    
    # P1
    U[0][col] = 1; U[4][col] = 1
    V[2][col] = 1; V[7][col] = 1
    W[6][col] = 1; W[10][col] = 1
    col += 1
    
    # P2
    U[3][col] = 1; U[4][col] = 1
    V[2][col] = 1
    W[7][col] = 1; W[10][col] = -1
    col += 1
    
    # P3
    U[0][col] = 1
    V[3][col] = 1; V[7][col] = -1
    W[9][col] = 1; W[10][col] = 1
    col += 1
    
    # P4
    U[4][col] = 1
    V[6][col] = 1; V[2][col] = -1
    W[6][col] = 1; W[7][col] = 1
    col += 1
    
    # P5
    U[0][col] = 1; U[1][col] = 1
    V[7][col] = 1
    W[6][col] = -1; W[9][col] = 1
    col += 1
    
    # P6
    U[3][col] = 1; U[0][col] = -1
    V[2][col] = 1; V[3][col] = 1
    W[10][col] = 1
    col += 1
    
    # P7
    U[1][col] = 1; U[4][col] = -1
    V[6][col] = 1; V[7][col] = 1
    W[6][col] = 1
    col += 1
    
    # A12 * B2R
    # A12 indices: 2, 5
    # B2R indices: 10, 11
    # C1R contribution:
    # C[0,2]->6, C[0,3]->9, C[1,2]->7, C[1,3]->10
    
    # T1: A[0,2] B[2,2] C[0,2]
    U[2][col] = 1
    V[10][col] = 1
    W[6][col] = 1
    col += 1
    
    # T2: A[0,2] B[2,3] C[0,3]
    U[2][col] = 1
    V[11][col] = 1
    W[9][col] = 1
    col += 1
    
    # T3: A[1,2] B[2,2] C[1,2]
    U[5][col] = 1
    V[10][col] = 1
    W[7][col] = 1
    col += 1
    
    # T4: A[1,2] B[2,3] C[1,3]
    U[5][col] = 1
    V[11][col] = 1
    W[10][col] = 1
    col += 1
    
    # A21 * B1R
    # A21 indices: 6, 7
    # B1R indices: 2,3,6,7
    # C2R contribution:
    # C[2,2]->8, C[2,3]->11
    
    # T1: A[2,0] B[0,2] C[2,2]
    U[6][col] = 1
    V[2][col] = 1
    W[8][col] = 1
    col += 1
    
    # T2: A[2,0] B[0,3] C[2,3]
    U[6][col] = 1
    V[3][col] = 1
    W[11][col] = 1
    col += 1
    
    # T3: A[2,1] B[1,2] C[2,2]
    U[7][col] = 1
    V[6][col] = 1
    W[8][col] = 1
    col += 1
    
    # T4: A[2,1] B[1,3] C[2,3]
    U[7][col] = 1
    V[7][col] = 1
    W[11][col] = 1
    col += 1
    
    # A22 * B2R
    # A22 index: 8
    # B2R indices: 10, 11
    # C2R contribution:
    # C[2,2]->8, C[2,3]->11
    
    # T1: A[2,2] B[2,2] C[2,2]
    U[8][col] = 1
    V[10][col] = 1
    W[8][col] = 1
    col += 1
    
    # T2: A[2,2] B[2,3] C[2,3]
    U[8][col] = 1
    V[11][col] = 1
    W[11][col] = 1
    col += 1
    
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
    
    # Try to sparsify with time limit
    U_arr, V_arr, W_arr, new_rank = _sparsify(U, V, W, max_iter=50, deadline=deadline)
    
    # Verify correctness
    U_np = np.array(U_arr)
    V_np = np.array(V_arr)
    W_np = np.array(W_arr)
    
    # Tensor: T[r, mid, c] = sum_k U[r*mid_dim + mid, k] * V[mid*p + c, k] * W[c*n + r, k]
    # where mid_dim = m (the inner dimension)
    mid_dim = m  # = 3
    
    def verify():
        # Vectorized verification
        # T[r, mid, c] = sum_k U[r*mid_dim + mid, k] * V[mid*p + c, k] * W[c*n + r, k]
        for r in range(n):
            for mid in range(mid_dim):
                for c in range(p):
                    u_row = U_np[r*mid_dim + mid, :]
                    v_row = V_np[mid*p + c, :]
                    w_row = W_np[c*n + r, :]
                    computed = np.dot(u_row, v_row * w_row)
                    if abs(computed - 1.0) > 1e-10:
                        return False
        return True
    
    if not verify():
        # Sparsification broke something, use original
        return {"U": U, "V": V, "W": W}
    
    # Convert back to lists
    U_list = U_arr.tolist()
    V_list = V_arr.tolist()
    W_list = W_arr.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
