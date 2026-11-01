import numpy as np

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    # The minimum rank for 2x4x4 matrix multiplication is 28.
    # This is derived from the fact that it decomposes into four independent 2x2x2 multiplications,
    # each of which has a minimal rank of 7 (Strassen).
    rank = 28
    
    # U corresponds to A (size n x m = 2 x 4)
    # V corresponds to B (size m x p = 4 x 4)
    # W corresponds to C (size p x n = 4 x 2)
    # Note: The problem statement says W[k*n+i, r] which implies W is indexed by output index and rank.
    # Output C is 2x4. Let C[i, j] be the element.
    # The trace convention W[k*n+i, r] usually means:
    # If C is n x p, then C[i, j] = sum_r U[i, r] V[j, r] W[j*n+i, r] ?
    # Let's check the seed code's W indexing.
    # Seed code: w_idx = C_cols[0] * n + C_rows[0]
    # Here C is 2x4. C_rows are 0,1. C_cols are 0,1,2,3.
    # So W index is col * n + row. This matches W[k*n+i] where k=col, i=row.
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_strassen_block(block_id, A_rows, A_cols, B_rows, B_cols, C_rows, C_cols):
        """
        Adds a Strassen decomposition for a 2x2 multiplication block.
        A is n x m, B is m x p, C is n x p (locally 2x2).
        A_rows, A_cols: global indices for the 2x2 block in A.
        B_rows, B_cols: global indices for the 2x2 block in B.
        C_rows, C_cols: global indices for the 2x2 block in C.
        """
        base_col = block_id * 7
        
        # Local indices in A:
        # a00 = A[A_rows[0], A_cols[0]]
        # a01 = A[A_rows[0], A_cols[1]]
        # a10 = A[A_rows[1], A_cols[0]]
        # a11 = A[A_rows[1], A_cols[1]]
        
        # Local indices in B:
        # b00 = B[B_rows[0], B_cols[0]]
        # b01 = B[B_rows[0], B_cols[1]]
        # b10 = B[B_rows[1], B_cols[0]]
        # b11 = B[B_rows[1], B_cols[1]]
        
        # Strassen Products M1..M7
        
        # M1 = (a00 + a11) * (b00 + b11)
        col = base_col + 0
        U[A_rows[0] * m + A_cols[0]][col] += 1
        U[A_rows[1] * m + A_cols[1]][col] += 1
        V[B_rows[0] * p + B_cols[0]][col] += 1
        V[B_rows[1] * p + B_cols[1]][col] += 1
        
        # M2 = (a10 + a11) * b00
        col = base_col + 1
        U[A_rows[1] * m + A_cols[0]][col] += 1
        U[A_rows[1] * m + A_cols[1]][col] += 1
        V[B_rows[0] * p + B_cols[0]][col] += 1
        
        # M3 = a00 * (b01 - b11)
        col = base_col + 2
        U[A_rows[0] * m + A_cols[0]][col] += 1
        V[B_rows[0] * p + B_cols[1]][col] += 1
        V[B_rows[1] * p + B_cols[1]][col] -= 1
        
        # M4 = a11 * (b10 - b00)
        col = base_col + 3
        U[A_rows[1] * m + A_cols[1]][col] += 1
        V[B_rows[1] * p + B_cols[0]][col] += 1
        V[B_rows[0] * p + B_cols[0]][col] -= 1
        
        # M5 = (a00 + a01) * b11
        col = base_col + 4
        U[A_rows[0] * m + A_cols[0]][col] += 1
        U[A_rows[0] * m + A_cols[1]][col] += 1
        V[B_rows[1] * p + B_cols[1]][col] += 1
        
        # M6 = (a10 - a00) * (b00 + b01)
        col = base_col + 5
        U[A_rows[1] * m + A_cols[0]][col] += 1
        U[A_rows[0] * m + A_cols[0]][col] -= 1
        V[B_rows[0] * p + B_cols[0]][col] += 1
        V[B_rows[0] * p + B_cols[1]][col] += 1
        
        # M7 = (a01 - a11) * (b10 + b11)
        col = base_col + 6
        U[A_rows[0] * m + A_cols[1]][col] += 1
        U[A_rows[1] * m + A_cols[1]][col] -= 1
        V[B_rows[1] * p + B_cols[0]][col] += 1
        V[B_rows[1] * p + B_cols[1]][col] += 1
        
        # Reconstruct C
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # W index for C[i, j] is j * n + i
        # C[0,0] -> W[C_cols[0] * n + C_rows[0]]
        w_idx_00 = C_cols[0] * n + C_rows[0]
        W[w_idx_00][base_col + 0] += 1 # M1
        W[w_idx_00][base_col + 3] += 1 # M4
        W[w_idx_00][base_col + 4] -= 1 # -M5
        W[w_idx_00][base_col + 6] += 1 # M7
        
        # C[0,1] -> W[C_cols[1] * n + C_rows[0]]
        w_idx_01 = C_cols[1] * n + C_rows[0]
        W[w_idx_01][base_col + 2] += 1 # M3
        W[w_idx_01][base_col + 4] += 1 # M5
        
        # C[1,0] -> W[C_cols[0] * n + C_rows[1]]
        w_idx_10 = C_cols[0] * n + C_rows[1]
        W[w_idx_10][base_col + 1] += 1 # M2
        W[w_idx_10][base_col + 3] += 1 # M4
        
        # C[1,1] -> W[C_cols[1] * n + C_rows[1]]
        w_idx_11 = C_cols[1] * n + C_rows[1]
        W[w_idx_11][base_col + 0] += 1 # M1
        W[w_idx_11][base_col + 1] -= 1 # -M2
        W[w_idx_11][base_col + 2] += 1 # M3
        W[w_idx_11][base_col + 5] += 1 # M6

    # We need to compute C = A * B
    # A is 2x4: Rows 0,1. Cols 0,1,2,3.
    # B is 4x4: Rows 0,1,2,3. Cols 0,1,2,3.
    # C is 2x4: Rows 0,1. Cols 0,1,2,3.
    
    # C[:, 0:2] = A[:, 0:2] * B[0:2, 0:2] + A[:, 2:4] * B[2:4, 0:2]
    # This is sum of two 2x2 multiplications.
    
    # Block 0: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    add_strassen_block(0, [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1])
    
    # Block 1: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    # Note: This adds to the same C block. The U and V entries are distinct (different A cols, B rows),
    # so the tensor terms are disjoint.
    add_strassen_block(1, [0, 1], [2, 3], [2, 3], [0, 1], [0, 1], [0, 1])
    
    # Block 2: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    add_strassen_block(2, [0, 1], [0, 1], [0, 1], [2, 3], [0, 1], [2, 3])
    
    # Block 3: A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    add_strassen_block(3, [0, 1], [2, 3], [2, 3], [2, 3], [0, 1], [2, 3])

    return {"U": U, "V": V, "W": W}
