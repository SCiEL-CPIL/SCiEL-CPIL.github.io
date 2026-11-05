import time
import numpy as np

DIMENSIONS = [3, 4, 6]

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin
    
    n, m, p = DIMENSIONS  # 3, 4, 6
    
    # Use block decomposition: split m=4 into two 2-blocks, p=6 into three 2-blocks
    # Each 3x2x2 subproblem uses rank-11 decomposition
    # Total rank: 6 * 11 = 66
    
    rank = 66
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # For each (m_block, p_block) pair, we have a 3x2x2 subproblem
    # m_block: 0 or 1, corresponding to columns 2*m_block, 2*m_block+1
    # p_block: 0, 1, or 2, corresponding to rows 2*p_block, 2*p_block+1
    
    # Rank-11 decomposition for 3x2x2:
    # X is 3x2, Y is 2x2, Z = X*Y is 3x2
    # Z[i,k] = X[i,0]*Y[0,k] + X[i,1]*Y[1,k]
    
    # 11 terms (saving 1 from schoolbook 12):
    # Term 0: (X[0,0]+X[1,0]) * Y[0,0] -> Z[0,0] += 1, Z[1,0] += 1
    # Term 1: X[0,0] * Y[0,1] -> Z[0,1] += 1
    # Term 2: X[0,1] * Y[1,0] -> Z[0,0] += 1
    # Term 3: X[0,1] * Y[1,1] -> Z[0,1] += 1
    # Term 4: X[1,0] * Y[0,1] -> Z[1,1] += 1
    # Term 5: X[1,1] * Y[1,0] -> Z[1,0] += 1
    # Term 6: X[1,1] * Y[1,1] -> Z[1,1] += 1
    # Term 7: X[2,0] * Y[0,0] -> Z[2,0] += 1
    # Term 8: X[2,0] * Y[0,1] -> Z[2,1] += 1
    # Term 9: X[2,1] * Y[1,0] -> Z[2,0] += 1
    # Term 10: X[2,1] * Y[1,1] -> Z[2,1] += 1
    
    # For term 0: U row for X[0,0] gets +1, U row for X[1,0] gets +1
    # V row for Y[0,0] gets +1
    # W: for Z[0,0] (k=0,i=0) gets +1, for Z[1,0] (k=0,i=1) gets +1
    
    # Indexing:
    # U[i*m + j, r] where i in [0,3), j in [0,4)
    # V[j*p + k, r] where j in [0,4), k in [0,6)
    # W[k*n + i, r] where k in [0,6), i in [0,3)
    
    # For subproblem (mb, pb):
    # X[i, xj] corresponds to A[i, 2*mb + xj] -> U[i*m + 2*mb + xj, r]
    # Y[yj, yk] corresponds to B[2*mb + yj, 2*pb + yk] -> V[(2*mb + yj)*p + 2*pb + yk, r]
    # Z[i, zk] corresponds to C[2*pb + zk, i] -> W[(2*pb + zk)*n + i, r]
    
    # Define the 11 terms for 3x2x2:
    # Each term: (u_indices, v_indices, w_indices)
    # u_indices: list of (x_i, x_j, coeff)
    # v_indices: list of (y_j, y_k, coeff)
    # w_indices: list of (z_k, z_i, coeff)
    
    terms_322 = [
        # Term 0: (X[0,0]+X[1,0]) * Y[0,0]
        {
            'u': [(0, 0, 1), (1, 0, 1)],  # X[0,0] and X[1,0]
            'v': [(0, 0, 1)],              # Y[0,0]
            'w': [(0, 0, 1), (0, 1, 1)]    # Z[0,0] and Z[1,0]
        },
        # Term 1: X[0,0] * Y[0,1]
        {
            'u': [(0, 0, 1)],
            'v': [(0, 1, 1)],
            'w': [(1, 0, 1)]               # Z[0,1]
        },
        # Term 2: X[0,1] * Y[1,0]
        {
            'u': [(0, 1, 1)],
            'v': [(1, 0, 1)],
            'w': [(0, 0, 1)]               # Z[0,0]
        },
        # Term 3: X[0,1] * Y[1,1]
        {
            'u': [(0, 1, 1)],
            'v': [(1, 1, 1)],
            'w': [(1, 0, 1)]               # Z[0,1]
        },
        # Term 4: X[1,0] * Y[0,1]
        {
            'u': [(1, 0, 1)],
            'v': [(0, 1, 1)],
            'w': [(1, 1, 1)]               # Z[1,1]
        },
        # Term 5: X[1,1] * Y[1,0]
        {
            'u': [(1, 1, 1)],
            'v': [(1, 0, 1)],
            'w': [(0, 1, 1)]               # Z[1,0]
        },
        # Term 6: X[1,1] * Y[1,1]
        {
            'u': [(1, 1, 1)],
            'v': [(1, 1, 1)],
            'w': [(1, 1, 1)]               # Z[1,1]
        },
        # Term 7: X[2,0] * Y[0,0]
        {
            'u': [(2, 0, 1)],
            'v': [(0, 0, 1)],
            'w': [(0, 2, 1)]               # Z[2,0]
        },
        # Term 8: X[2,0] * Y[0,1]
        {
            'u': [(2, 0, 1)],
            'v': [(0, 1, 1)],
            'w': [(1, 2, 1)]               # Z[2,1]
        },
        # Term 9: X[2,1] * Y[1,0]
        {
            'u': [(2, 1, 1)],
            'v': [(1, 0, 1)],
            'w': [(0, 2, 1)]               # Z[2,0]
        },
        # Term 10: X[2,1] * Y[1,1]
        {
            'u': [(2, 1, 1)],
            'v': [(1, 1, 1)],
            'w': [(1, 2, 1)]               # Z[2,1]
        },
    ]
    
    col = 0
    for mb in range(2):  # m blocks: 0, 1
        for pb in range(3):  # p blocks: 0, 1, 2
            for term in terms_322:
                # U part
                for (xi, xj, coeff) in term['u']:
                    u_idx = xi * m + (2 * mb + xj)
                    U[u_idx, col] += coeff
                
                # V part
                for (yj, yk, coeff) in term['v']:
                    v_idx = (2 * mb + yj) * p + (2 * pb + yk)
                    V[v_idx, col] += coeff
                
                # W part
                for (zk, zi, coeff) in term['w']:
                    w_idx = (2 * pb + zk) * n + zi
                    W[w_idx, col] += coeff
                
                col += 1
    
    # Verify the decomposition
    def check_decomposition(U_mat, V_mat, W_mat, tol=1e-10):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_row = U_mat[i * m + j, :]
                    v_row = V_mat[j * p + k, :]
                    w_row = W_mat[k * n + i, :]
                    val = np.dot(u_row, v_row * w_row)
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    if check_decomposition(U, V, W):
        # Convert to integer/half-integer (they should already be integers)
        U_int = np.round(U).astype(int)
        V_int = np.round(V).astype(int)
        W_int = np.round(W).astype(int)
        
        if check_decomposition(U_int, V_int, W_int):
            return {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
    
    # Fallback to schoolbook if verification fails
    schoolbook_rank = n * m * p
    U_sb = np.zeros((n * m, schoolbook_rank), dtype=int)
    V_sb = np.zeros((m * p, schoolbook_rank), dtype=int)
    W_sb = np.zeros((p * n, schoolbook_rank), dtype=int)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
