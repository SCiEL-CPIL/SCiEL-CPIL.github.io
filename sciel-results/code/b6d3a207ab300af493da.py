import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = 2, 4, 7
    N_U = n * m  # 8
    N_V = m * p  # 28
    N_W = p * n  # 14
    
    # Total rank: 14 (for 2x4x2) * 3 + 8 (for 2x4x1) = 50
    total_rank = 50
    U = np.zeros((N_U, total_rank), dtype=np.int64)
    V = np.zeros((N_V, total_rank), dtype=np.int64)
    W = np.zeros((N_W, total_rank), dtype=np.int64)
    
    def add_strassen_block(U, V, W, offset, i_offset, j_offset, k_offset):
        """
        Adds a 2x2x2 Strassen decomposition for sub-blocks.
        U, V, W are the global matrices.
        offset: column index in U, V, W to start writing.
        i_offset: row index in A (0 or 1, but A is 2x4, so i is 0,1)
        j_offset: col index in A (0, 2 for the two 2x2 blocks)
        k_offset: col index in B (0, 1 for the 2x2 block of B)
        
        Note: The 2x4x2 block uses A[i, j_offset:j_offset+2] and B[j_offset:j_offset+2, k_offset:k_offset+2].
        """
        # Strassen 7 terms
        # M1 = A00 * (B01 + B11)
        # M2 = (A00 + A10) * B10
        # M3 = (A01 + A11) * B00
        # M4 = A11 * (B00 - B10)
        # M5 = (A00 + A11) * (B00 + B11)
        # M6 = (A01 - A10) * (B10 + B11)
        # M7 = (A00 - A01) * B01
        
        # Indices in A: i in {0,1}, j in {j_offset, j_offset+1}
        # Indices in B: j in {j_offset, j_offset+1}, k in {k_offset, k_offset+1}
        
        # Term 1: A[i_offset, j_offset] * (B[j_offset, k_offset+1] + B[j_offset+1, k_offset+1])
        # U: A[i, j]
        # V: B[j, k]
        # W: C[k, i]
        
        # We need to map local indices to global indices.
        # Local i: 0, 1. Global i: i_offset + local_i? 
        # Wait, the block is 2x2 in A. So local i is 0,1.
        # Global i is just i (since n=2).
        # Local j: 0, 1. Global j = j_offset + local_j.
        # Local k: 0, 1. Global k = k_offset + local_k.
        
        # Let's define the 7 Strassen terms for a generic 2x2x2.
        # We will add 7 columns to U, V, W.
        
        # Helper to set U, V, W for a term
        # term_idx: 0..6
        
        # M1: A00 * (B01 + B11)
        # U: A00 -> 1
        # V: B01 -> 1, B11 -> 1
        # W: C00? No, M1 contributes to C01 and C11.
        # C01 = M1 + M3. C11 = M5 - M1 + M3 + M7.
        # So W for M1 should be 1 for C01 and -1 for C11?
        # W indices: k, i.
        # C01: k=1, i=0. Index 1*n + 0 = 2.
        # C11: k=1, i=1. Index 1*n + 1 = 3.
        
        # Let's build the 7 terms explicitly.
        
        # Term 0 (M1):
        # U: A[0, 0] (local) -> Global A[i_offset, j_offset]
        U[i_offset * m + j_offset, offset + 0] = 1
        # V: B[0, 1] and B[1, 1] (local) -> Global B[j_offset, k_offset+1] and B[j_offset+1, k_offset+1]
        V[j_offset * p + k_offset + 1, offset + 0] = 1
        V[(j_offset + 1) * p + k_offset + 1, offset + 0] = 1
        # W: C[1, 0] (1) and C[1, 1] (-1)
        W[(k_offset + 1) * n + i_offset, offset + 0] = 1
        W[(k_offset + 1) * n + i_offset + 1, offset + 0] = -1
        
        # Term 1 (M2): (A00 + A10) * B10
        U[i_offset * m + j_offset, offset + 1] = 1
        U[(i_offset + 1) * m + j_offset, offset + 1] = 1
        V[(j_offset + 1) * p + k_offset, offset + 1] = 1
        W[(k_offset) * n + i_offset + 1, offset + 1] = 1
        W[(k_offset + 1) * n + i_offset + 1, offset + 1] = 1
        
        # Term 2 (M3): (A01 + A11) * B00
        U[i_offset * m + j_offset + 1, offset + 2] = 1
        U[(i_offset + 1) * m + j_offset + 1, offset + 2] = 1
        V[j_offset * p + k_offset, offset + 2] = 1
        W[(k_offset) * n + i_offset, offset + 2] = 1
        W[(k_offset + 1) * n + i_offset, offset + 2] = 1
        
        # Term 3 (M4): A11 * (B00 - B10)
        U[(i_offset + 1) * m + j_offset + 1, offset + 3] = 1
        V[j_offset * p + k_offset, offset + 3] = 1
        V[(j_offset + 1) * p + k_offset, offset + 3] = -1
        W[(k_offset) * n + i_offset, offset + 3] = 1
        W[(k_offset) * n + i_offset + 1, offset + 3] = 1
        W[(k_offset + 1) * n + i_offset, offset + 3] = -1
        W[(k_offset + 1) * n + i_offset + 1, offset + 3] = -1
        
        # Term 4 (M5): (A00 + A11) * (B00 + B11)
        U[i_offset * m + j_offset, offset + 4] = 1
        U[(i_offset + 1) * m + j_offset + 1, offset + 4] = 1
        V[j_offset * p + k_offset, offset + 4] = 1
        V[(j_offset + 1) * p + k_offset + 1, offset + 4] = 1
        W[(k_offset) * n + i_offset, offset + 4] = 1
        W[(k_offset) * n + i_offset + 1, offset + 4] = 1
        W[(k_offset + 1) * n + i_offset, offset + 4] = 1
        W[(k_offset + 1) * n + i_offset + 1, offset + 4] = 1
        
        # Term 5 (M6): (A01 - A10) * (B10 + B11)
        U[i_offset * m + j_offset + 1, offset + 5] = 1
        U[(i_offset + 1) * m + j_offset, offset + 5] = -1
        V[(j_offset + 1) * p + k_offset, offset + 5] = 1
        V[(j_offset + 1) * p + k_offset + 1, offset + 5] = 1
        W[(k_offset + 1) * n + i_offset, offset + 5] = 1
        W[(k_offset + 1) * n + i_offset + 1, offset + 5] = 1
        
        # Term 6 (M7): (A00 - A01) * B01
        U[i_offset * m + j_offset, offset + 6] = 1
        U[i_offset * m + j_offset + 1, offset + 6] = -1
        V[j_offset * p + k_offset + 1, offset + 6] = 1
        W[(k_offset + 1) * n + i_offset + 1, offset + 6] = 1
        
        return offset + 7

    def add_schoolbook_2x4x1_block(U, V, W, offset, i_offset, j_offset, k_offset):
        """
        Adds a 2x4x1 schoolbook decomposition.
        A is 2x4, B is 4x1, C is 2x1.
        """
        current_offset = offset
        for i in range(2):
            for j in range(4):
                # U: A[i, j]
                U[(i_offset + i) * m + (j_offset + j), current_offset] = 1
                # V: B[j, 0] (local k=0)
                V[(j_offset + j) * p + (k_offset), current_offset] = 1
                # W: C[0, i] (local k=0)
                W[(k_offset) * n + (i_offset + i), current_offset] = 1
                current_offset += 1
        return current_offset

    current_rank = 0
    
    # Block 1: A[:, 0:2] * B[0:2, 0:2] -> C[:, 0:2]
    # i_offset=0, j_offset=0, k_offset=0
    current_rank = add_strassen_block(U, V, W, current_rank, 0, 0, 0)
    
    # Block 2: A[:, 2:4] * B[2:4, 0:2] -> C[:, 0:2]
    # i_offset=0, j_offset=2, k_offset=0
    current_rank = add_strassen_block(U, V, W, current_rank, 0, 2, 0)
    
    # Block 3: A[:, 0:2] * B[0:2, 2:4] -> C[:, 2:4]
    # i_offset=0, j_offset=0, k_offset=2
    current_rank = add_strassen_block(U, V, W, current_rank, 0, 0, 2)
    
    # Block 4: A[:, 2:4] * B[2:4, 2:4] -> C[:, 2:4]
    # i_offset=0, j_offset=2, k_offset=2
    current_rank = add_strassen_block(U, V, W, current_rank, 0, 2, 2)
    
    # Block 5: A[:, 0:2] * B[0:2, 4:6] -> C[:, 4:6]
    # i_offset=0, j_offset=0, k_offset=4
    current_rank = add_strassen_block(U, V, W, current_rank, 0, 0, 4)
    
    # Block 6: A[:, 2:4] * B[2:4, 4:6] -> C[:, 4:6]
    # i_offset=0, j_offset=2, k_offset=4
    current_rank = add_strassen_block(U, V, W, current_rank, 0, 2, 4)
    
    # Block 7: A[:, 0:4] * B[0:4, 6:7] -> C[:, 6:7]
    # This is 2x4x1.
    # i_offset=0, j_offset=0, k_offset=6
    current_rank = add_schoolbook_2x4x1_block(U, V, W, current_rank, 0, 0, 6)
    
    # Verify current_rank is 50
    # 6 * 7 + 8 = 42 + 8 = 50.
    
    if time.time() > deadline:
        # Should not happen
        pass
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
