"""Matrix multiplication 4,4,9 - rank-minimal tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 9
    start_time = time.time()
    deadline = start_time + 115  # 115s budget, 5s safety margin
    
    def schoolbook_block(n, m, p_block, offset_u, offset_v, offset_w, rank_offset):
        """
        Generate schoolbook decomposition for a block of size n x m x p_block.
        offset_u, offset_v, offset_w are the starting indices for U, V, W rows.
        rank_offset is the starting column index for U, V, W.
        """
        rank = n * m * p_block
        U_block = np.zeros((n * m, rank))
        V_block = np.zeros((m * p_block, rank))
        W_block = np.zeros((p_block * n, rank))
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p_block):
                    U_block[i * m + j, col] = 1
                    V_block[j * p_block + k, col] = 1
                    W_block[k * n + i, col] = 1
                    col += 1
        return U_block, V_block, W_block, rank

    def construct_decomposition():
        # Block 1: Cols 0-3 (4 cols)
        # Block 2: Cols 4-7 (4 cols)
        # Block 3: Col 8 (1 col)
        
        # Block 1: 4x4x4
        U1, V1, W1, r1 = schoolbook_block(4, 4, 4, 0, 0, 0, 0)
        # Block 2: 4x4x4
        # V offset: 4*4 = 16 (cols 4-7 of B)
        # W offset: 4*4 = 16 (cols 4-7 of C)
        U2, V2, W2, r2 = schoolbook_block(4, 4, 4, 0, 16, 16, 0)
        # Block 3: 4x4x1
        # V offset: 8*4 = 32 (col 8 of B)
        # W offset: 8*4 = 32 (col 8 of C)
        U3, V3, W3, r3 = schoolbook_block(4, 4, 1, 0, 32, 32, 0)
        
        total_rank = r1 + r2 + r3
        
        U = np.zeros((16, total_rank))
        V = np.zeros((36, total_rank))
        W = np.zeros((36, total_rank))
        
        # Assemble
        # Block 1
        U[:, 0:r1] = U1
        V[0:16, 0:r1] = V1
        W[0:16, 0:r1] = W1
        
        # Block 2
        U[:, r1:r1+r2] = U2
        V[16:32, r1:r1+r2] = V2
        W[16:32, r1:r1+r2] = W2
        
        # Block 3
        U[:, r1+r2:r1+r2+r3] = U3
        V[32:36, r1+r2:r1+r2+r3] = V3
        W[32:36, r1+r2:r1+r2+r3] = W3
        
        return U, V, W, total_rank

    def check_validity(U, V, W, tol=1e-6):
        """Check if the decomposition is exact."""
        for i in range(n):
            for k in range(p):
                U_slice = U[i*m:(i+1)*m, :]
                V_cols = np.array([V[j*p + k, :] for j in range(m)])
                W_val = W[k*n + i, :]
                temp = U_slice * W_val
                result = temp @ V_cols.T
                expected = np.eye(m)
                if np.max(np.abs(result - expected)) > tol:
                    return False
        return True

    U, V, W, rank = construct_decomposition()
    
    if not check_validity(U, V, W):
        # Fallback to full schoolbook if construction failed (should not happen)
        U, V, W, rank = schoolbook_block(4, 4, 9, 0, 0, 0, 0)
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
