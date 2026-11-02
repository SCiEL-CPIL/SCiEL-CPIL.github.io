import numpy as np
import time
import sys

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    def verify(U, V, W):
        """Verify that the tensor decomposition is correct."""
        r = U.shape[1]
        U_r = U.reshape(n, m, r)
        V_r = V.reshape(m, p, r)
        W_r = W.reshape(p, n, r)
        
        T = np.zeros((n, m, p))
        for r_idx in range(r):
            u_col = U_r[:, :, r_idx]
            v_col = V_r[:, :, r_idx]
            w_col = W_r[:, :, r_idx]
            T += u_col[:, :, None] * v_col[None, :, :] * w_col.T[:, None, :]
            
        return np.allclose(T, np.ones((n, m, p)), atol=1e-9)
    
    # Strassen's rank-7 algorithm for 2x2x2 matrix multiplication
    # M1 = (A11+A22)(B12+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B11-B21)
    # M4 = A22(B22-B12)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # For 3x2x2 block: split into 2x2x2 (Strassen, rank 7) + 1x2x2 (trivial, rank 4)
    # Total per 3x2x2 block: 11
    # 4 such blocks for 3x4x4: total rank 44
    
    # Build the 3x4x4 decomposition using 4 blocks of 3x2x2
    # Block (bi, bj): rows of A are all 3, cols of A are 2*bi to 2*bi+1, 
    #                 rows of B are 2*bi to 2*bi+1, cols of B are 2*bj to 2*bj+1
    # Wait, let me re-think the block structure.
    
    # A is 3x4, B is 4x4, C is 3x4
    # Split A's 4 columns into two groups of 2: A[:, 0:2] and A[:, 2:4]
    # Split B's 4 rows into two groups of 2: B[0:2, :] and B[2:4, :]
    # Split B's 4 columns into two groups of 2: B[:, 0:2] and B[:, 2:4]
    
    # C = A @ B = A[:, 0:2] @ B[0:2, :] + A[:, 2:4] @ B[2:4, :]
    # Each term is (3x2) @ (2x4) = 3x4
    # We can further split the (2x4) part into (2x2) blocks
    
    # Actually, let's do it differently:
    # C[:, 0:2] = A[:, 0:2] @ B[0:2, 0:2] + A[:, 2:4] @ B[2:4, 0:2]
    # C[:, 2:4] = A[:, 0:2] @ B[0:2, 2:4] + A[:, 2:4] @ B[2:4, 2:4]
    
    # So we have 4 blocks of the form (3x2) @ (2x2) -> (3x2)
    # For each such block, we use Strassen for the top 2x2 part and trivial for the bottom 1x2 part.
    
    # Let's build this.
    # For a 3x2 @ 2x2 block:
    # A_block = [[a00, a01], [a10, a11], [a20, a21]] (3x2)
    # B_block = [[b00, b01], [b10, b11]] (2x2)
    # C_block = [[c00, c01], [c10, c11], [c20, c21]] (3x2)
    
    # Top 2x2: [[a00, a01], [a10, a11]] @ [[b00, b01], [b10, b11]]
    # Use Strassen rank 7
    # Bottom 1x2: [a20, a21] @ [[b00, b01], [b10, b11]]
    # Use trivial rank 4 (or 2? Let's check)
    # Actually, (1x2) @ (2x2) = (1x2), which is just:
    # c20 = a20*b00 + a21*b10
    # c21 = a20*b01 + a21*b11
    # This can be done with 2 terms, not 4.
    
    # So per 3x2x2 block: 7 (Strassen) + 2 (trivial) = 9
    # 4 blocks: 4 * 9 = 36
    
    # Let me verify this is correct by building it.
    
    def build_3x2x2_block(U_offset, V_offset, W_offset, a_row_start, a_col_start, b_row_start, b_col_start):
        """Build a 3x2x2 block decomposition.
        
        A_block: rows a_row_start to a_row_start+2, cols a_col_start to a_col_start+1
        B_block: rows b_row_start to b_row_start+1, cols b_col_start to b_col_start+1
        C_block: rows a_row_start to a_row_start+2, cols b_col_start to b_col_start+1
        
        Returns the number of terms used.
        """
        rank_used = 0
        
        # Strassen for top 2x2 part
        # A_top = A[a_row_start:a_row_start+2, a_col_start:a_col_start+2]
        # B_block = B[b_row_start:b_row_start+2, b_col_start:b_col_start+2]
        # C_top = C[a_row_start:a_row_start+2, b_col_start:b_col_start+2]
        
        # Strassen terms:
        # M1 = (A00+A11)(B01+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B00-B10)
        # M4 = A11(B11-B01)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # C00 = M1+M4-M5+M7
        # C01 = M3+M5
        # C10 = M2+M4
        # C11 = M1-M2+M3+M6
        
        # Let's define the indices:
        # A indices: (i, j) where i is row (0,1 for top), j is col (0,1)
        # B indices: (j, k) where j is row (0,1), k is col (0,1)
        # C indices: (i, k) where i is row (0,1), k is col (0,1)
        
        # For the full 3x4x4:
        # A global index: (a_row_start + i, a_col_start + j)
        # B global index: (b_row_start + j, b_col_start + k)
        # C global index: (a_row_start + i, b_col_start + k)
        
        # W index: (k, i) in the block, which is (b_col_start + k, a_row_start + i) globally
        
        # M1: (A00+A11)(B01+B11)
        # U: (0,0) and (1,1) -> A[a_row_start, a_col_start] and A[a_row_start+1, a_col_start+1]
        # V: (0,1) and (1,1) -> B[b_row_start, b_col_start+1] and B[b_row_start+1, b_col_start+1]
        # W: contributes to C00, C11
        # C00 = M1 + ..., C11 = M1 + ...
        # So W for M1: (0,0) gets +1, (1,1) gets +1
        
        # Let me just code this up directly.
        
        # M1
        U_offset += 0
        U[a_row_start * m + a_col_start, U_offset] = 1
        U[(a_row_start + 1) * m + (a_col_start + 1), U_offset] = 1
        V[b_row_start * p + (b_col_start + 1), U_offset] = 1
        V[(b_row_start + 1) * p + (b_col_start + 1), U_offset] = 1
        W[(b_col_start + 0) * n + (a_row_start + 0), U_offset] = 1  # C00
        W[(b_col_start + 1) * n + (a_row_start + 1), U_offset] = 1  # C11
        rank_used += 1
        
        # M2
        U_offset += 1
        U[(a_row_start + 1) * m + a_col_start, U_offset] = 1
        U[(a_row_start + 1) * m + (a_col_start + 1), U_offset] = 1
        V[b_row_start * p + b_col_start, U_offset] = 1
        W[(b_col_start + 0) * n + (a_row_start + 1), U_offset] = 1  # C10
        rank_used += 1
        
        # M3
        U_offset += 1
        U[a_row_start * m + a_col_start, U_offset] = 1
        V[b_row_start * p + b_col_start, U_offset] = 1
        V[(b_row_start + 1) * p + b_col_start, U_offset] = -1
        W[(b_col_start + 1) * n + (a_row_start + 0), U_offset] = 1  # C01
        W[(b_col_start + 1) * n + (a_row_start + 1), U_offset] = 1  # C11
        rank_used += 1
        
        # M4
        U_offset += 1
        U[(a_row_start + 1) * m + (a_col_start + 1), U_offset] = 1
        V[(b_row_start + 1) * p + (b_col_start + 1), U_offset] = 1
        V[b_row_start * p + (b_col_start + 1), U_offset] = -1
        W[(b_col_start + 0) * n + (a_row_start + 0), U_offset] = 1  # C00
        W[(b_col_start + 0) * n + (a_row_start + 1), U_offset] = 1  # C10
        rank_used += 1
        
        # M5
        U_offset += 1
        U[a_row_start * m + a_col_start, U_offset] = 1
        U[a_row_start * m + (a_col_start + 1), U_offset] = 1
        V[(b_row_start + 1) * p + (b_col_start + 1), U_offset] = 1
        W[(b_col_start + 0) * n + (a_row_start + 0), U_offset] = -1  # C00
        W[(b_col_start + 1) * n + (a_row_start + 0), U_offset] = 1  # C01
        rank_used += 1
        
        # M6
        U_offset += 1
        U[(a_row_start + 1) * m + a_col_start, U_offset] = 1
        U[a_row_start * m + a_col_start, U_offset] = -1
        V[b_row_start * p + b_col_start, U_offset] = 1
        V[b_row_start * p + (b_col_start + 1), U_offset] = 1
        W[(b_col_start + 1) * n + (a_row_start + 1), U_offset] = 1  # C11
        rank_used += 1
        
        # M7
        U_offset += 1
        U[a_row_start * m + (a_col_start + 1), U_offset] = 1
        U[(a_row_start + 1) * m + (a_col_start + 1), U_offset] = -1
        V[(b_row_start + 1) * p + b_col_start, U_offset] = 1
        V[(b_row_start + 1) * p + (b_col_start + 1), U_offset] = 1
        W[(b_col_start + 0) * n + (a_row_start + 0), U_offset] = 1  # C00
        rank_used += 1
        
        # Now the bottom row (row 2 of the 3x2 block)
        # C20 = a20*b00 + a21*b10
        # C21 = a20*b01 + a21*b11
        
        # Term for a20*b00
        U_offset += 1
        U[(a_row_start + 2) * m + a_col_start, U_offset] = 1
        V[b_row_start * p + b_col_start, U_offset] = 1
        W[(b_col_start + 0) * n + (a_row_start + 2), U_offset] = 1
        rank_used += 1
        
        # Term for a21*b10
        U_offset += 1
        U[(a_row_start + 2) * m + (a_col_start + 1), U_offset] = 1
        V[(b_row_start + 1) * p + b_col_start, U_offset] = 1
        W[(b_col_start + 0) * n + (a_row_start + 2), U_offset] = 1
        rank_used += 1
        
        # Term for a20*b01
        U_offset += 1
        U[(a_row_start + 2) * m + a_col_start, U_offset] = 1
        V[b_row_start * p + (b_col_start + 1), U_offset] = 1
        W[(b_col_start + 1) * n + (a_row_start + 2), U_offset] = 1
        rank_used += 1
        
        # Term for a21*b11
        U_offset += 1
        U[(a_row_start + 2) * m + (a_col_start + 1), U_offset] = 1
        V[(b_row_start + 1) * p + (b_col_start + 1), U_offset] = 1
        W[(b_col_start + 1) * n + (a_row_start + 2), U_offset] = 1
        rank_used += 1
        
        return rank_used
    
    # Total rank: 4 blocks * 11 terms per block = 44
    total_rank = 44
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    # Block (0,0): A[:, 0:2] @ B[0:2, 0:2] -> C[:, 0:2]
    offset = 0
    build_3x2x2_block(offset, offset, offset, 0, 0, 0, 0)
    offset += 11
    
    # Block (0,1): A[:, 0:2] @ B[0:2, 2:4] -> C[:, 2:4]
    build_3x2x2_block(offset, offset, offset, 0, 0, 0, 2)
    offset += 11
    
    # Block (1,0): A[:, 2:4] @ B[2:4, 0:2] -> C[:, 0:2]
    build_3x2x2_block(offset, offset, offset, 0, 2, 2, 0)
    offset += 11
    
    # Block (1,1): A[:, 2:4] @ B[2:4, 2:4] -> C[:, 2:4]
    build_3x2x2_block(offset, offset, offset, 0, 2, 2, 2)
    offset += 11
    
    # Verify
    if verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook if verification fails
    rank_school = n * m * p
    best_U = np.zeros((n*m, rank_school))
    best_V = np.zeros((m*p, rank_school))
    best_W = np.zeros((p*n, rank_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i*m + j, col] = 1
                best_V[j*p + k, col] = 1
                best_W[k*n + i, col] = 1
                col += 1
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
