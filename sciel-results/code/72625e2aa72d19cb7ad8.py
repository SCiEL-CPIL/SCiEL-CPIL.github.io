import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 120.0
    
    n, m, p = 4, 4, 8
    
    # 1. Define 2x2x2 Strassen tensors (rank 7)
    # Indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # M1
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    # M2
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1
    # M3
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1
    # M4
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[1, 3] = 1; W2[2, 3] = 1
    # M5
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[1, 4] = 1
    # M6
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1
    # M7
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1; W2[3, 6] = 1
    
    # 2. Define 4x4x4 Strassen tensors (rank 49)
    # Outer combinations
    outer_A = [
        np.array([[1,0],[0,1]]),
        np.array([[0,0],[1,1]]),
        np.array([[1,0],[0,0]]),
        np.array([[0,0],[0,1]]),
        np.array([[1,1],[0,0]]),
        np.array([[-1,0],[1,0]]),
        np.array([[0,1],[-1,0]])
    ]
    outer_B = [
        np.array([[1,0],[0,1]]),
        np.array([[1,0],[0,0]]),
        np.array([[0,1],[0,-1]]),
        np.array([[-1,0],[1,0]]),
        np.array([[0,0],[0,1]]),
        np.array([[1,1],[0,0]]),
        np.array([[0,0],[1,1]])
    ]
    outer_C = [
        np.array([[1,0],[0,1]]),
        np.array([[0,0],[1,0]]),
        np.array([[0,1],[0,0]]),
        np.array([[0,1],[1,0]]),
        np.array([[-1,1],[0,0]]),
        np.array([[0,0],[0,1]]),
        np.array([[1,0],[0,1]])
    ]
    
    U44 = np.zeros((16, 49))
    V44 = np.zeros((16, 49))
    W44 = np.zeros((16, 49))
    
    for r_out in range(7):
        for r_in in range(7):
            r = r_out * 7 + r_in
            # U
            for i_block in range(2):
                for i_local in range(2):
                    i = i_block * 2 + i_local
                    # U44[i, r] = sum_{j_block} outer_A[r_out][i_block, j_block] * U2[i_local, r_in]
                    # Wait, the outer combination is applied to the blocks.
                    # The "A" input to the inner multiplication is a 2x2 matrix A' where A'[i_block, j_block] is the coefficient?
                    # No, the Strassen formula is M_r = (Sum c_A[i,j] A_ij) * (Sum c_B[i,j] B_ij).
                    # The result is a 2x2 matrix M'.
                    # M' is then combined into C using c_C.
                    # So for the inner multiplication, the "A" matrix is the 2x2 matrix of coefficients?
                    # No, the inner multiplication multiplies two 2x2 matrices.
                    # The first 2x2 matrix is formed by the linear combination of the 4 blocks of A.
                    # Let A_blocks be a 2x2 array of 2x2 matrices.
                    # A_combined = sum_{i,j} outer_A[r_out][i,j] * A_blocks[i,j].
                    # This A_combined is a 2x2 matrix.
                    # So the "A" input to the inner Strassen is this 2x2 matrix.
                    # The entries of this 2x2 matrix are linear combinations of the entries of A_blocks.
                    # Specifically, A_combined[i_local, j_local] = sum_{i_block, j_block} outer_A[r_out][i_block, j_block] * A_blocks[i_block, j_block][i_local, j_local].
                    # So U44[i, r] where i = i_block*2 + i_local? No.
                    # i indexes the flat A. i = i_block*2 + i_local is not quite right for the block structure.
                    # A is 4x4. Flat index i = row*4 + col.
                    # Block i_block, j_block contains A[row=i_block*2+rl, col=j_block*2+cl].
                    # So A[i_block*2+rl, j_block*2+cl] has flat index (i_block*2+rl)*4 + (j_block*2+cl).
                    
                    # Let's re-index.
                    # A[i, j] with i,j in 0..3.
                    # Block (bi, bj) contains A[bi*2+ri, bj*2+cj].
                    # The coefficient for A[bi*2+ri, bj*2+cj] in the combined matrix A_combined[ri, cj] is outer_A[r_out][bi, bj].
                    # So for the inner multiplication, the input A matrix has entries A_combined[ri, cj].
                    # The flat index for A_combined[ri, cj] in the 2x2 space is ri*2 + cj.
                    # The flat index for A[bi*2+ri, bj*2+cj] in the 4x4 space is (bi*2+ri)*4 + (bj*2+cj).
                    
                    # So U44[flat_44, r] = sum_{bi, bj} outer_A[r_out][bi, bj] * U2[ri*2+cj, r_in]
                    # where flat_44 = (bi*2+ri)*4 + (bj*2+cj).
                    
                    pass # Logic inside loops below
            
            # Construct U44[:, r]
            for bi in range(2):
                for bj in range(2):
                    coeff_a = outer_A[r_out][bi, bj]
                    if coeff_a == 0: continue
                    for ri in range(2):
                        for cj in range(2):
                            flat_a = (bi*2+ri)*4 + (bj*2+cj)
                            inner_idx = ri*2 + cj
                            U44[flat_a, r] += coeff_a * U2[inner_idx, r_in]
            
            # Construct V44[:, r]
            for bi in range(2):
                for bj in range(2):
                    coeff_b = outer_B[r_out][bi, bj]
                    if coeff_b == 0: continue
                    for ri in range(2):
                        for cj in range(2):
                            flat_b = (bi*2+ri)*4 + (bj*2+cj)
                            inner_idx = ri*2 + cj
                            V44[flat_b, r] += coeff_b * V2[inner_idx, r_in]
            
            # Construct W44[:, r]
            for bi in range(2):
                for bj in range(2):
                    coeff_c = outer_C[r_out][bi, bj]
                    if coeff_c == 0: continue
                    for ri in range(2):
                        for cj in range(2):
                            flat_c = (bi*2+ri)*4 + (bj*2+cj)
                            inner_idx = ri*2 + cj
                            W44[flat_c, r] += coeff_c * W2[inner_idx, r_in]
                            
    # 3. Construct 4x4x8 decomposition (rank 98)
    # Split B into B_left (cols 0-3) and B_right (cols 4-7).
    # U is same for both.
    # V is V44 for left, V44 for right (but mapped to correct B indices).
    # W is W44 for left (mapped to C cols 0-3), W44 for right (mapped to C cols 4-7).
    
    U98 = np.zeros((16, 98))
    V98 = np.zeros((32, 98))
    W98 = np.zeros((32, 98))
    
    # Left half (cols 0-3 of B, cols 0-3 of C)
    for r in range(49):
        U98[:, r] = U44[:, r]
        # V44 indices are 0..15 (4x4). B_left is 4x4. Indices match.
        V98[0:16, r] = V44[:, r]
        # W44 indices are 0..15 (4x4). C_left is 4x4.
        # C_left flat index k = row*4 + col.
        # In 4x4x8, C flat index is row*8 + col.
        # For C_left, col is 0..3. So k_48 = row*8 + col = row*4 + col + row*4.
        # Wait. k_44 = row*4 + col.
        # k_48 = row*8 + col.
        # k_48 = 2*k_44 - row*4? No.
        # Let's just map explicitly.
        for k_44 in range(16):
            row = k_44 // 4
            col = k_44 % 4
            k_48 = row * 8 + col
            W98[k_48, r] = W44[k_44, r]
            
    # Right half (cols 4-7 of B, cols 4-7 of C)
    for r in range(49):
        r_98 = 49 + r
        U98[:, r_98] = U44[:, r]
        # B_right is 4x4. Flat index in B_right is row*4 + col.
        # In 4x4x8, B flat index is row*8 + col.
        # For B_right, col is 4..7. Let col_local = col - 4.
        # k_B_right = row*4 + col_local.
        # k_B_48 = row*8 + (col_local + 4) = row*8 + col_local + 4.
        for k_B_44 in range(16):
            row = k_B_44 // 4
            col_local = k_B_44 % 4
            k_B_48 = row * 8 + col_local + 4
            V98[k_B_48, r_98] = V44[k_B_44, r]
            
        # C_right is 4x4. Flat index in C_right is row*4 + col_local.
        # In 4x4x8, C flat index is row*8 + col.
        # For C_right, col is 4..7. Let col_local = col - 4.
        # k_C_right = row*4 + col_local.
        # k_C_48 = row*8 + (col_local + 4) = row*8 + col_local + 4.
        for k_C_44 in range(16):
            row = k_C_44 // 4
            col_local = k_C_44 % 4
            k_C_48 = row * 8 + col_local + 4
            W98[k_C_48, r_98] = W44[k_C_44, r]
            
    return {"U": U98.tolist(), "V": V98.tolist(), "W": W98.tolist()}
