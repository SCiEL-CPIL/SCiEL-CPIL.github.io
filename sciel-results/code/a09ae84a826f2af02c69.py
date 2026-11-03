import numpy as np

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # We will construct the factor matrices U, V, W.
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    
    # We use a hybrid approach:
    # 1. Strassen for the first 4 columns of B (4x4 multiplication)
    # 2. Schoolbook for the remaining 3 columns of B (4x4 by 4x3 multiplication)
    
    # Strassen for 4x4 by 4x4
    # This requires 7 multiplications of 2x2 by 2x2.
    # Each 2x2 by 2x2 multiplication has rank 7 (Strassen).
    # So total rank for this part is 7 * 7 = 49.
    
    # Schoolbook for 4x4 by 4x3
    # This requires 4 * 4 * 3 = 48 multiplications.
    
    total_rank = 49 + 48
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # --- Part 1: Strassen for A * B1 (4x4 by 4x4) ---
    # B1 is columns 0..3 of B.
    # We need to decompose the 4x4 by 4x4 multiplication.
    # Let A = [[A11, A12], [A21, A22]] and B1 = [[B11, B12], [B21, B22]].
    # M1 = (A11 + A22) * (B11 + B21)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B11 - B21)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B21
    # M6 = (A21 - A11) * (B11 + B21)
    # M7 = (A12 - A22) * (B21 - B11)
    
    # Each Mi is a 2x2 by 2x2 multiplication.
    # We will use Strassen for each Mi.
    # Strassen for 2x2 by 2x2:
    # Let X = [[x11, x12], [x21, x22]] and Y = [[y11, y12], [y21, y22]].
    # S1 = (x11 + x22)(y11 + y22)
    # S2 = (x21 + x22)y11
    # S3 = x11(y11 - y21)
    # S4 = x22(y21 - y11)
    # S5 = (x11 + x12)y21
    # S6 = (x21 - x11)(y11 + y22)
    # S7 = (x12 - x22)(y21 - y11)
    # Result:
    # C11 = S1 + S4 - S5 + S7
    # C12 = S3 + S5
    # C21 = S2 + S4
    # C22 = S1 - S2 + S3 + S6
    
    # We need to map the indices.
    # A is 4x4, indices i, j.
    # B1 is 4x4, indices j, k (k=0..3).
    # C1 is 4x4, indices i, k.
    
    # Let's define a helper to add a Strassen decomposition for a 2x2 by 2x2 multiplication.
    # The inputs are 2x2 blocks of A and B.
    # We need to specify the linear combinations of the elements of A and B.
    
    # This is getting complex to index. Let's use a simpler approach:
    # Just use schoolbook for everything but with a better block structure?
    # No, 97 is a good target. Let's implement it carefully.
    
    # Actually, I can just use the schoolbook for the 4x4 by 4x3 part and Strassen for 4x4 by 4x4.
    # Let's write a function to add a rank-1 term.
    def add_term(u_idx, v_idx, w_idx, u_val, v_val, w_val):
        nonlocal col
        U[u_idx][col] += u_val
        V[v_idx][col] += v_val
        W[w_idx][col] += w_val
        col += 1
        
    # This is not efficient. Let's build the matrices directly.
    
    # Let's restart with a cleaner structure.
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    # Part 1: Strassen for 4x4 by 4x4
    # We have 7 sub-multiplications, each of rank 7.
    # Let's define the 7 sub-multiplications.
    # A blocks: A11(0,0), A12(0,1), A21(1,0), A22(1,1) -> indices 0..1, 0..1, etc.
    # B1 blocks: B11(0,0), B12(0,1), B21(1,0), B22(1,1) -> indices 0..1, 0..1, etc.
    
    # We will iterate over the 7 Strassen terms for the 4x4 by 4x4 multiplication.
    # For each term, we will add 7 rank-1 terms (Strassen for 2x2 by 2x2).
    
    # Let's define the Strassen coefficients for 2x2 by 2x2.
    # S1 = (x11 + x22)(y11 + y22)
    # S2 = (x21 + x22)y11
    # S3 = x11(y11 - y21)
    # S4 = x22(y21 - y11)
    # S5 = (x11 + x12)y21
    # S6 = (x21 - x11)(y11 + y22)
    # S7 = (x12 - x22)(y21 - y11)
    
    # This is too tedious to index manually. Let's use a generic tensor product approach.
    # The rank of 4x4 by 4x4 is at most 49.
    # The rank of 4x4 by 4x3 is 48.
    # Total 97.
    
    # Let's just use schoolbook for the whole thing but with a better block structure?
    # No, 112 is the schoolbook. 97 is better.
    
    # Let's implement the 97 rank decomposition.
    
    # Part 1: 4x4 by 4x4 (Strassen)
    # We will use the fact that the tensor for 4x4 by 4x4 is the tensor product of two 2x2 by 2x2 tensors.
    # The rank of the tensor product is at most the product of the ranks.
    # Rank(2x2 by 2x2) = 7.
    # Rank(4x4 by 4x4) <= 7 * 7 = 49.
    
    # Part 2: 4x4 by 4x3 (Schoolbook)
    # Rank = 4 * 4 * 3 = 48.
    
    # Let's build the matrices.
    
    # Part 1: Strassen for 4x4 by 4x4
    # We need to decompose the 7x7 tensor.
    # Let's use the Strassen decomposition for 2x2 by 2x2.
    # Let T be the tensor for 2x2 by 2x2.
    # T = sum_{r=1}^7 u_r v_r w_r^T
    # Then the tensor for 4x4 by 4x4 is T \otimes T.
    # (T \otimes T) = sum_{r1=1}^7 sum_{r2=1}^7 (u_{r1} \otimes u_{r2}) (v_{r1} \otimes v_{r2}) (w_{r1} \otimes w_{r2})^T
    # So we can just take the tensor product of the Strassen factors.
    
    # Let's get the Strassen factors for 2x2 by 2x2.
    # U_2x2: 4x7, V_2x2: 4x7, W_2x2: 4x7
    U_2x2 = np.zeros((4, 7))
    V_2x2 = np.zeros((4, 7))
    W_2x2 = np.zeros((4, 7))
    
    # S1 = (x11 + x22)(y11 + y22)
    U_2x2[0, 0] += 1; U_2x2[3, 0] += 1
    V_2x2[0, 0] += 1; V_2x2[3, 0] += 1
    W_2x2[0, 0] += 1; W_2x2[3, 0] += 1
    
    # S2 = (x21 + x22)y11
    U_2x2[2, 1] += 1; U_2x2[3, 1] += 1
    V_2x2[0, 1] += 1
    W_2x2[2, 1] += 1
    
    # S3 = x11(y11 - y21)
    U_2x2[0, 2] += 1
    V_2x2[0, 2] += 1; V_2x2[2, 2] -= 1
    W_2x2[1, 2] += 1
    
    # S4 = x22(y21 - y11)
    U_2x2[3, 3] += 1
    V_2x2[2, 3] += 1; V_2x2[0, 3] -= 1
    W_2x2[2, 3] += 1
    
    # S5 = (x11 + x12)y21
    U_2x2[0, 4] += 1; U_2x2[1, 4] += 1
    V_2x2[2, 4] += 1
    W_2x2[1, 4] += 1
    
    # S6 = (x21 - x11)(y11 + y22)
    U_2x2[2, 5] += 1; U_2x2[0, 5] -= 1
    V_2x2[0, 5] += 1; V_2x2[3, 5] += 1
    W_2x2[3, 5] += 1
    
    # S7 = (x12 - x22)(y21 - y11)
    U_2x2[1, 6] += 1; U_2x2[3, 6] -= 1
    V_2x2[2, 6] += 1; V_2x2[0, 6] -= 1
    W_2x2[3, 6] += 1
    
    # Now, the tensor for 4x4 by 4x4 is U_2x2 \otimes U_2x2, V_2x2 \otimes V_2x2, W_2x2 \otimes W_2x2.
    # We need to map the indices correctly.
    # A is 4x4, indices i, j.
    # B is 4x4, indices j, k.
    # C is 4x4, indices i, k.
    
    # The Kronecker product of U_2x2 (4x7) with itself gives a 16x49 matrix.
    # The rows are indexed by (i1, i2, j1, j2) where i1, j1 are the block indices and i2, j2 are the element indices.
    # Actually, the row index for A is (block_i, element_i, block_j, element_j).
    # Let's use the standard Kronecker product.
    
    U_part1 = np.kron(U_2x2, U_2x2)
    V_part1 = np.kron(V_2x2, V_2x2)
    W_part1 = np.kron(W_2x2, W_2x2)
    
    # We need to permute the columns of W_part1 to match the trace convention W[k*n+i, r].
    # The Kronecker product of W_2x2 (4x7) with itself gives a 16x49 matrix.
    # The row index is (k1, k2, i1, i2) where k1, i1 are the block indices and k2, i2 are the element indices.
    # We need to map this to (k, i) where k = k1*2 + k2 and i = i1*2 + i2.
    # So the row index in the Kronecker product is (k1*4 + i1)*2 + (k2*2 + i2)? No.
    # The Kronecker product of two vectors a (length 4) and b (length 4) is a[0]*b, a[1]*b, a[2]*b, a[3]*b.
    # So the row index is a_idx * 4 + b_idx.
    # For W_2x2, the row index is (k_block, i_block, k_elem, i_elem)? No.
    # W_2x2 has rows (k, i) where k, i are 0..1.
    # So the row index is k*2 + i.
    # The Kronecker product of W_2x2 with itself has rows (k1, i1, k2, i2) mapped to (k1*4 + i1)*2 + (k2*2 + i2)? No.
    # The Kronecker product of a 4x7 matrix with a 4x7 matrix is a 16x49 matrix.
    # The row index is (row1 * 4 + row2).
    # row1 is the row index of the first W_2x2, which is k1*2 + i1.
    # row2 is the row index of the second W_2x2, which is k2*2 + i2.
    # So the row index is (k1*2 + i1)*4 + (k2*2 + i2).
    # We need to map this to (k, i) where k = k1*2 + k2 and i = i1*2 + i2.
    # So k = k1*2 + k2, i = i1*2 + i2.
    # The target row index is k*4 + i = (k1*2 + k2)*4 + (i1*2 + i2) = k1*8 + k2*4 + i1*2 + i2.
    # The Kronecker row index is (k1*2 + i1)*4 + (k2*2 + i2) = k1*8 + i1*4 + k2*2 + i2.
    # So we need to permute the rows.
    
    # Let's create a permutation matrix.
    perm = np.zeros((16, 16))
    for k1 in range(2):
        for i1 in range(2):
            for k2 in range(2):
                for i2 in range(2):
                    kron_idx = (k1*2 + i1)*4 + (k2*2 + i2)
                    target_idx = (k1*2 + k2)*4 + (i1*2 + i2)
                    perm[target_idx, kron_idx] = 1
    
    W_part1 = perm @ W_part1
    
    # Now we have U_part1 (16x49), V_part1 (16x49), W_part1 (16x49).
    # We need to map these to the full 4x4 by 4x4 problem.
    # U_part1 rows are (i1, j1, i2, j2) mapped to (i1*4 + i2)*4 + (j1*4 + j2)? No.
    # The Kronecker product of U_2x2 (4x7) with itself has rows (i1*2 + j1)*4 + (i2*2 + j2).
    # We need to map this to (i, j) where i = i1*2 + i2 and j = j1*2 + j2.
    # So i = i1*2 + i2, j = j1*2 + j2.
    # The target row index is i*4 + j = (i1*2 + i2)*4 + (j1*2 + j2) = i1*8 + i2*4 + j1*2 + j2.
    # The Kronecker row index is (i1*2 + j1)*4 + (i2*2 + j2) = i1*8 + j1*4 + i2*2 + j2.
    # So we need to permute the rows of U_part1 and V_part1.
    
    perm_uv = np.zeros((16, 16))
    for i1 in range(2):
        for j1 in range(2):
            for i2 in range(2):
                for j2 in range(2):
                    kron_idx = (i1*2 + j1)*4 + (i2*2 + j2)
                    target_idx = (i1*2 + i2)*4 + (j1*2 + j2)
                    perm_uv[target_idx, kron_idx] = 1
    
    U_part1 = perm_uv @ U_part1
    V_part1 = perm_uv @ V_part1
    
    # Now we have the 49 rank-1 terms for the 4x4 by 4x4 multiplication.
    # We need to add these to U, V, W.
    # U has shape (16, 97). V has shape (28, 97). W has shape (28, 97).
    # U_part1 has shape (16, 49). V_part1 has shape (16, 49). W_part1 has shape (16, 49).
    # But V_part1 is for B1 which is 4x4. The full B is 4x7.
    # So V_part1 should be embedded in V.
    # V has rows (j, k) where j=0..3, k=0..6.
    # V_part1 has rows (j, k) where j=0..3, k=0..3.
    # So we can just assign V[0:16, 0:49] = V_part1.
    # But we need to be careful with the row indices.
    # V_part1 rows are (j, k) where j = j1*2 + j2 and k = k1*2 + k2.
    # The target row index is j*7 + k.
    # So we need to map (j, k) to j*7 + k.
    
    # Let's create a mapping for V_part1.
    v_map = np.zeros((16, 28))
    for j1 in range(2):
        for k1 in range(2):
            for j2 in range(2):
                for k2 in range(2):
                    kron_idx = (j1*2 + k1)*4 + (j2*2 + k2) # This is the row index in V_part1
                    j = j1*2 + j2
                    k = k1*2 + k2
                    target_idx = j*7 + k
                    v_map[target_idx, kron_idx] = 1
    
    V_part1_full = v_map @ V_part1
    
    # Similarly for W_part1.
    # W_part1 rows are (k, i) where k = k1*2 + k2 and i = i1*2 + i2.
    # The target row index is k*4 + i.
    # We already permuted W_part1 to have rows (k, i).
    # So we can just assign W[0:16, 0:49] = W_part1.
    
    U[:, 0:49] = U_part1
    V[:, 0:49] = V_part1_full
    W[:, 0:49] = W_part1
    
    # Part 2: Schoolbook for 4x4 by 4x3
    # B2 is columns 4..6 of B.
    # We need to compute A * B2.
    # Rank = 4 * 4 * 3 = 48.
    for i in range(4):
        for j in range(4):
            for k in range(3):
                col_idx = 49 + (i*4 + j)*3 + k
                U[i*4 + j, col_idx] = 1
                V[j*7 + (k+4), col_idx] = 1
                W[(k+4)*4 + i, col_idx] = 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
