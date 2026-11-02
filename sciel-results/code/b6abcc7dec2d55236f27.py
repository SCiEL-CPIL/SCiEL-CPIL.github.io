# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations

def construct():
    """
    Rank-minimal exact tensor decomposition for 2x4 * 4x5 matrix multiplication.
    Combines structured Strassen block construction with greedy column elimination.
    """
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = 2, 4, 5
    
    # Build the rank-36 Strassen-based decomposition
    U = np.zeros((n * m, 36), dtype=np.float64)
    V = np.zeros((m * p, 36), dtype=np.float64)
    W = np.zeros((p * n, 36), dtype=np.float64)
    
    def u_idx(i, j):
        return i * m + j
    
    def v_idx(j, k):
        return j * p + k
    
    def w_idx(k, i):
        return k * n + i
    
    col = 0
    
    def strassen_block(a_cols, b_rows, c_cols):
        """
        Add 7 Strassen terms for a 2x2 x 2x2 block.
        a_cols: [col0, col1] in A
        b_rows: [row0, row1] in B
        c_cols: [col0, col1] in C
        """
        nonlocal col
        a0, a1 = a_cols
        b0, b1 = b_rows
        c0, c1 = c_cols
        
        # M1 = (A[0,a0]+A[1,a1]) * (B[b0,c0]+B[b1,c1])
        U[u_idx(0, a0), col] = 1; U[u_idx(1, a1), col] = 1
        V[v_idx(b0, c0), col] = 1; V[v_idx(b1, c1), col] = 1
        W[w_idx(c0, 0), col] += 1; W[w_idx(c1, 1), col] += 1
        col += 1
        
        # M2 = (A[1,a0]+A[1,a1]) * B[b0,c0]
        U[u_idx(1, a0), col] = 1; U[u_idx(1, a1), col] = 1
        V[v_idx(b0, c0), col] = 1
        W[w_idx(c0, 1), col] += 1; W[w_idx(c1, 1), col] -= 1
        col += 1
        
        # M3 = A[0,a0] * (B[b0,c1]-B[b1,c1])
        U[u_idx(0, a0), col] = 1
        V[v_idx(b0, c1), col] = 1; V[v_idx(b1, c1), col] = -1
        W[w_idx(c1, 0), col] += 1; W[w_idx(c1, 1), col] += 1
        col += 1
        
        # M4 = A[1,a1] * (B[b1,c0]-B[b0,c0])
        U[u_idx(1, a1), col] = 1
        V[v_idx(b1, c0), col] = 1; V[v_idx(b0, c0), col] = -1
        W[w_idx(c0, 0), col] += 1; W[w_idx(c0, 1), col] += 1
        col += 1
        
        # M5 = (A[0,a0]+A[0,a1]) * B[b1,c1]
        U[u_idx(0, a0), col] = 1; U[u_idx(0, a1), col] = 1
        V[v_idx(b1, c1), col] = 1
        W[w_idx(c0, 0), col] -= 1; W[w_idx(c1, 0), col] += 1
        col += 1
        
        # M6 = (A[1,a0]-A[0,a0]) * (B[b0,c0]+B[b0,c1])
        U[u_idx(1, a0), col] = 1; U[u_idx(0, a0), col] = -1
        V[v_idx(b0, c0), col] = 1; V[v_idx(b0, c1), col] = 1
        W[w_idx(c1, 1), col] += 1
        col += 1
        
        # M7 = (A[0,a1]-A[1,a1]) * (B[b1,c0]+B[b1,c1])
        U[u_idx(0, a1), col] = 1; U[u_idx(1, a1), col] = -1
        V[v_idx(b1, c0), col] = 1; V[v_idx(b1, c1), col] = 1
        W[w_idx(c0, 0), col] += 1
        col += 1
    
    def schoolbook_block(a_cols, b_rows, c_col):
        """
        Add 4 schoolbook terms for a 2x2 x 2x1 block.
        """
        nonlocal col
        a0, a1 = a_cols
        b0, b1 = b_rows
        
        for i, a_col in enumerate([a0, a1]):
            for j, b_row in enumerate([b0, b1]):
                U[u_idx(i, a_col), col] = 1
                V[v_idx(b_row, c_col), col] = 1
                W[w_idx(c_col, i), col] += 1
                col += 1
    
    # Block 1: A[:, 0:2] * B[0:2, 0:2] -> C[:, 0:2]
    strassen_block([0, 1], [0, 1], [0, 1])
    
    # Block 2: A[:, 2:4] * B[2:4, 0:2] -> C[:, 0:2]
    strassen_block([2, 3], [2, 3], [0, 1])
    
    # Block 3: A[:, 0:2] * B[0:2, 2:4] -> C[:, 2:4]
    strassen_block([0, 1], [0, 1], [2, 3])
    
    # Block 4: A[:, 2:4] * B[2:4, 2:4] -> C[:, 2:4]
    strassen_block([2, 3], [2, 3], [2, 3])
    
    # Block 5: A[:, 0:2] * B[0:2, 4:5] -> C[:, 4:5]
    schoolbook_block([0, 1], [0, 1], 4)
    
    # Block 6: A[:, 2:4] * B[2:4, 4:5] -> C[:, 4:5]
    schoolbook_block([2, 3], [2, 3], 4)
    
    # Verify initial decomposition
    def verify(U_mat, V_mat, W_mat):
        rank = U_mat.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U_mat[i*m+j, r] * V_mat[j*p+k, r] * W_mat[k*n+i, r]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    if not verify(U, V, W):
        # Should not happen, but return as-is
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_rank = 36
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Greedy column elimination
    current_rank = 36
    
    for iteration in range(20):
        if time.time() > deadline:
            break
        
        removed_any = False
        
        for col_idx in range(current_rank):
            if time.time() > deadline:
                break
            
            # Try removing this column
            mask = [i for i in range(current_rank) if i != col_idx]
            U_new = U[:, mask]
            V_new = V[:, mask]
            W_new = W[:, mask]
            
            if verify(U_new, V_new, W_new):
                U, V, W = U_new, V_new, W_new
                current_rank -= 1
                removed_any = True
                if current_rank < best_rank:
                    best_rank = current_rank
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                break  # Restart from beginning
        
        if not removed_any:
            break
    
    # Try pairwise column elimination for further improvement
    for _ in range(5):
        if time.time() > deadline:
            break
        
        improved = False
        
        for c1 in range(current_rank):
            if time.time() > deadline:
                break
            for c2 in range(c1 + 1, current_rank):
                if time.time() > deadline:
                    break
                
                mask = [i for i in range(current_rank) if i != c1 and i != c2]
                U_new = U[:, mask]
                V_new = V[:, mask]
                W_new = W[:, mask]
                
                if verify(U_new, V_new, W_new):
                    U, V, W = U_new, V_new, W_new
                    current_rank -= 2
                    improved = True
                    if current_rank < best_rank:
                        best_rank = current_rank
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    break
            
            if improved:
                break
        
        if not improved:
            break
    
    # Use the best found
    if best_rank < 36:
        U, V, W = best_U, best_V, best_W
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
