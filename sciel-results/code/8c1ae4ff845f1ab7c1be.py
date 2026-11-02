import numpy as np
import time

DIMENSIONS = [2, 4, 6]

def construct():
    start_time = time.time()
    DEADLINE = 135.0  # Safety margin before 150s kill

    n, m, p = DIMENSIONS
    
    # Rank-42 decomposition for 2x4x6 matrix multiplication
    # Strategy: Partition A (2x4) into two 2x2 blocks: A = [A_L, A_R]
    # Partition B (4x6) into three 2x2 column blocks: B = [B_1, B_2, B_3]
    # Note: This partitioning is not standard Strassen because the block sizes
    # for the second dimension of A (2) and first dimension of B (4) don't align
    # perfectly for a single Strassen step on 2x2 blocks of equal size.
    
    # Let's re-evaluate the partitioning.
    # A is 2x4. B is 4x6.
    # If we split the middle dimension (4) into two parts of size 2:
    # A = [A1, A2] where A1, A2 are 2x2.
    # B = [B1; B2] where B1, B2 are 2x6.
    # C = A1@B1 + A2@B2.
    
    # This is a sum of two 2x2 @ 2x6 multiplications.
    # Rank of 2x2 @ 2x6?
    # We can split the 2x6 into three 2x2 blocks: B1 = [B11, B12, B13].
    # Then A1@B1 = A1@B11 + A1@B12 + A1@B13.
    # Similarly for A2.
    # Total 6 terms of 2x2 @ 2x2.
    # Each 2x2 @ 2x2 has rank 7 (Strassen).
    # Total rank = 6 * 7 = 42.
    
    # This matches the "hierarchical block decomposition" strategy in the directive.
    
    rank = 42
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Strassen decomposition for 2x2 @ 2x2
    # Inputs: A (2x2), B (2x2)
    # Outputs: C (2x2)
    # 7 multiplications:
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We need to map this to the global indices.
    # Global A indices:
    # A_L (rows 0-1, cols 0-1): 
    #   A11 = A[0,0], A12 = A[0,1]
    #   A21 = A[1,0], A22 = A[1,1]
    # A_R (rows 0-1, cols 2-3):
    #   A11 = A[0,2], A12 = A[0,3]
    #   A21 = A[1,2], A22 = A[1,3]
    
    # Global B indices:
    # B1 (rows 0-1, cols 0-1):
    #   B11 = B[0,0], B12 = B[0,1]
    #   B21 = B[1,0], B22 = B[1,1]
    # B2 (rows 0-1, cols 2-3):
    #   B11 = B[0,2], B12 = B[0,3]
    #   B21 = B[1,2], B22 = B[1,3]
    # B3 (rows 0-1, cols 4-5):
    #   B11 = B[0,4], B12 = B[0,5]
    #   B21 = B[1,4], B22 = B[1,5]
    
    # Wait, the partition of B is by rows, not columns, if we do A=[A1, A2] and B=[B1; B2].
    # A1 is 2x2 (rows 0-1, cols 0-1). A2 is 2x2 (rows 0-1, cols 2-3).
    # B1 is 2x6 (rows 0-1, cols 0-5). B2 is 2x6 (rows 2-3, cols 0-5).
    
    # So we have:
    # Term 1: A1 @ B1. A1 is 2x2, B1 is 2x6.
    # Split B1 into three 2x2 blocks: B1_1 (cols 0-1), B1_2 (cols 2-3), B1_3 (cols 4-5).
    # A1 @ B1 = A1 @ B1_1 + A1 @ B1_2 + A1 @ B1_3.
    # Each is a 2x2 @ 2x2 multiplication.
    
    # Term 2: A2 @ B2. A2 is 2x2, B2 is 2x6.
    # Split B2 into three 2x2 blocks: B2_1 (cols 0-1), B2_2 (cols 2-3), B2_3 (cols 4-5).
    # A2 @ B2 = A2 @ B2_1 + A2 @ B2_2 + A2 @ B2_3.
    
    # Total 6 Strassen multiplications.
    
    # Let's define the Strassen steps for a generic 2x2 @ 2x2.
    # We will create a helper to add a Strassen decomposition for specific A and B blocks.
    
    def add_strassen(A_block, B_block, C_block, col_offset):
        # A_block: list of (row, col) for the 4 elements of A
        # B_block: list of (row, col) for the 4 elements of B
        # C_block: list of (row, col) for the 4 elements of C
        
        # A indices: 00, 01, 10, 11
        a00, a01, a10, a11 = A_block
        # B indices: 00, 01, 10, 11
        b00, b01, b10, b11 = B_block
        # C indices: 00, 01, 10, 11
        c00, c01, c10, c11 = C_block
        
        steps = [
            # (A coeffs, B coeffs, C coeffs)
            # M1: (A00+A11) * (B00+B11) -> C00, C11
            ([(a00, 1), (a11, 1)], [(b00, 1), (b11, 1)], [(c00, 1), (c11, 1)]),
            # M2: (A10+A11) * B00 -> C10
            ([(a10, 1), (a11, 1)], [(b00, 1)], [(c10, 1)]),
            # M3: A00 * (B01-B11) -> C01
            ([(a00, 1)], [(b01, 1), (b11, -1)], [(c01, 1)]),
            # M4: A11 * (B10-B00) -> C00, C10
            ([(a11, 1)], [(b10, 1), (b00, -1)], [(c00, 1), (c10, 1)]),
            # M5: (A00+A01) * B11 -> C00, C01
            ([(a00, 1), (a01, 1)], [(b11, 1)], [(c00, -1), (c01, 1)]),
            # M6: (A10-A00) * (B00+B01) -> C11
            ([(a10, 1), (a00, -1)], [(b00, 1), (b01, 1)], [(c11, 1)]),
            # M7: (A01-A11) * (B10+B11) -> C00, C11
            ([(a01, 1), (a11, -1)], [(b10, 1), (b11, 1)], [(c00, 1), (c11, -1)]),
        ]
        
        col = col_offset
        for a_coeffs, b_coeffs, c_coeffs in steps:
            for (ai, a_val) in a_coeffs:
                U[ai, col] += a_val
            for (bi, b_val) in b_coeffs:
                V[bi, col] += b_val
            for (ci, c_val) in c_coeffs:
                W[ci, col] += c_val
            col += 1
        return col
    
    # Define the blocks
    # A1: rows 0-1, cols 0-1
    A1 = [(0,0), (0,1), (1,0), (1,1)]
    # A2: rows 0-1, cols 2-3
    A2 = [(0,2), (0,3), (1,2), (1,3)]
    
    # B1: rows 0-1, cols 0-5 (split into 3 2x2 blocks)
    B1_1 = [(0,0), (0,1), (1,0), (1,1)]
    B1_2 = [(0,2), (0,3), (1,2), (1,3)]
    B1_3 = [(0,4), (0,5), (1,4), (1,5)]
    
    # B2: rows 2-3, cols 0-5 (split into 3 2x2 blocks)
    B2_1 = [(2,0), (2,1), (3,0), (3,1)]
    B2_2 = [(2,2), (2,3), (3,2), (3,3)]
    B2_3 = [(2,4), (2,5), (3,4), (3,5)]
    
    # C: 2x6 (split into 6 2x2 blocks)
    # C11: rows 0-1, cols 0-1
    C1_1 = [(0,0), (0,1), (1,0), (1,1)]
    # C12: rows 0-1, cols 2-3
    C1_2 = [(0,2), (0,3), (1,2), (1,3)]
    # C13: rows 0-1, cols 4-5
    C1_3 = [(0,4), (0,5), (1,4), (1,5)]
    # C21: rows 0-1, cols 0-1 (from A2@B2 part)
    C2_1 = [(0,0), (0,1), (1,0), (1,1)]
    # C22: rows 0-1, cols 2-3
    C2_2 = [(0,2), (0,3), (1,2), (1,3)]
    # C23: rows 0-1, cols 4-5
    C2_3 = [(0,4), (0,5), (1,4), (1,5)]
    
    col = 0
    # A1 @ B1 parts
    col = add_strassen(A1, B1_1, C1_1, col)
    col = add_strassen(A1, B1_2, C1_2, col)
    col = add_strassen(A1, B1_3, C1_3, col)
    
    # A2 @ B2 parts
    col = add_strassen(A2, B2_1, C2_1, col)
    col = add_strassen(A2, B2_2, C2_2, col)
    col = add_strassen(A2, B2_3, C2_3, col)
    
    # Verification
    if time.time() < DEADLINE - 10:
        np.random.seed(42)
        A = np.random.rand(n, m)
        B = np.random.rand(m, p)
        C_true = A @ B
        
        A_flat = A.flatten()
        B_flat = B.flatten()
        
        u_vals = U @ A_flat
        v_vals = V @ B_flat
        prod_vals = u_vals * v_vals
        
        W_reshaped = W.reshape(p, n, rank).transpose(1, 0, 2)
        C_decomp = np.sum(W_reshaped * prod_vals[None, None, :], axis=2)
        
        if not np.allclose(C_true, C_decomp):
            raise ValueError("Decomposition verification failed")

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    result = construct()
    print("Rank:", len(result["U"][0]))
