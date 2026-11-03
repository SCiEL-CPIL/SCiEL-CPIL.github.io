import numpy as np
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    # 1. Define Strassen decomposition for 2x2
    # U2, V2, W2 are 4x7 matrices
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1,  0, 1],
        [0, 1, 0, 0, 0,  1, 0],
        [1, 1, 0, 1, 0,  0,-1]
    ], dtype=float)
    
    V2 = np.array([
        [1, 1, 0, 0, 0, 1, 0],
        [1, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [0, 0,-1,-1, 1, 0, 1]
    ], dtype=float)
    
    W2 = np.array([
        [1, 0, 0, 1,-1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [-1,-1, 1, 0, 0, 1, 0]
    ], dtype=float)
    
    # Verify Strassen
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # U2 * A_flat, V2 * B_flat, W2 * (U2*A) * (V2*B)
    # This is standard.
    
    # 2. Define Schoolbook decomposition for 3x3
    # Rank 27
    n3 = 3
    r3 = 27
    U3 = np.zeros((9, r3))
    V3 = np.zeros((9, r3))
    W3 = np.zeros((9, r3))
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U3[i*3+j, col] = 1
                V3[j*3+k, col] = 1
                W3[k*3+i, col] = 1
                col += 1
                
    # 3. Tensor Product
    # U = U2 (Kronecker) U3
    # Shapes: U2 (4,7), U3 (9,27) -> U (36, 189)
    # Kronecker product in numpy: np.kron(U2, U3) gives shape (4*9, 7*27)
    # The row index in U is i2*9 + i3.
    # i2 is 0..1, i3 is 0..8.
    # i_total = i2*3 + i3? 
    # Wait. The tensor product of M_2 and M_3 is M_6.
    # Input A of M_6 is A[i2*3+i3, j2*3+j3].
    # The flattening order for M_6 input is row-major: (i2*3+i3)*6 + (j2*3+j3).
    # np.kron(U2, U3) produces rows indexed by i2*9 + i3.
    # This does NOT match the row-major flattening of the 6x6 matrix.
    # We need to permute the rows.
    
    U = np.kron(U2, U3)
    V = np.kron(V2, V3)
    W = np.kron(W2, W3)
    
    # Permute rows of U, V, W to match the 6x6 row-major order.
    # Current row index in U: r = i2 * 9 + i3.
    # i3 corresponds to (i3_row, i3_col) in 3x3?
    # In U3, row index is i3_flat = i3_row * 3 + i3_col.
    # So i3 = i3_row * 3 + i3_col.
    # Total row index in 6x6: i_total = i2 * 3 + i3_row.
    # j_total = j2 * 3 + j3_col.
    # Flat index in 6x6: idx = i_total * 6 + j_total.
    # We need to map the current row index (i2 * 9 + i3) to idx.
    
    def permute_rows(mat, n_sub):
        # mat shape: (n1 * n_sub^2, r)
        # n1 is dimension of first factor (2 for U2/V2/W2)
        # n_sub is dimension of second factor (3 for U3/V3/W3)
        # Current row: i1 * n_sub^2 + i_sub
        # i_sub is flattened index of n_sub x n_sub matrix.
        # i_sub = i_sub_row * n_sub + i_sub_col
        # Target row: (i1 * n_sub + i_sub_row) * (n_sub * n_sub) + (i1 * n_sub + i_sub_col) * 0? No.
        # The 6x6 matrix index is (I, J).
        # I = i1 * 3 + i_sub_row
        # J = j1 * 3 + j_sub_col (for U, J is the column of A)
        # For U, the row index represents the input A element (I, J).
        # So we need to sort rows by (I, J).
        
        n1 = mat.shape[0] // (n_sub * n_sub)
        n_total = n1 * n_sub
        new_mat = np.zeros_like(mat)
        
        for i1 in range(n1):
            for i_sub in range(n_sub * n_sub):
                i_sub_row = i_sub // n_sub
                i_sub_col = i_sub % n_sub
                
                I = i1 * n_sub + i_sub_row
                J = i1 * n_sub + i_sub_col # Wait, for U, the second index is the column of A.
                # In the tensor product, the "column" part of U comes from the second index of the sub-matrices.
                # U2 has columns for A's columns? No, U2 rows are A's elements.
                # U2[i2, k] where i2 is flat index of A (2x2).
                # i2 = i2_row * 2 + i2_col.
                # So i_sub in U3 is i3_row * 3 + i3_col.
                # The total element is A[i2_row * 3 + i3_row, i2_col * 3 + i3_col].
                # So I = i2_row * 3 + i3_row.
                # J = i2_col * 3 + i3_col.
                
                # Current row index in mat: i1 * 9 + i_sub.
                # Target row index: I * 6 + J.
                
                # We need to map (i1, i_sub) -> (I, J).
                # i1 is i2_row? No, i1 is the index in U2.
                # U2 row index i1 = i2_row * 2 + i2_col.
                # So we must decompose i1 into i2_row, i2_col.
                
                i2_row = i1 // 2
                i2_col = i1 % 2
                
                I = i2_row * 3 + i_sub_row
                J = i2_col * 3 + i_sub_col
                
                target_idx = I * 6 + J
                new_mat[target_idx] = mat[i1 * 9 + i_sub]
                
        return new_mat

    # Apply permutation
    # Check time
    if time.time() > deadline:
        # Fallback to schoolbook if too slow (unlikely)
        rank = 216
        U = np.zeros((36, rank))
        V = np.zeros((36, rank))
        W = np.zeros((36, rank))
        col = 0
        for i in range(6):
            for j in range(6):
                for k in range(6):
                    U[i*6+j, col] = 1
                    V[j*6+k, col] = 1
                    W[k*6+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    U = permute_rows(U, 3)
    V = permute_rows(V, 3)
    W = permute_rows(W, 3)
    
    # Convert to lists of floats (integers/half-integers)
    # The entries are integers, so they are exact.
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
