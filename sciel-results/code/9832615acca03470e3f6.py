import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We will construct a rank-85 decomposition.
    # Split the 5x5 multiplication into 4x4x4, 4x4x1, 4x1x4, 4x1x1 blocks.
    # 
    # Indices:
    # i in [0, 3]
    # j in [0, 4] -> split into j' in [0, 3] and j=4
    # k in [0, 4] -> split into k' in [0, 3] and k=4
    #
    # C[i, k] = sum_j A[i, j] B[j, k]
    #
    # Block 1: j' in [0, 3], k' in [0, 3]
    # C[i, k'] = sum_{j'=0}^3 A[i, j'] B[j', k']
    # This is a 4x4x4 multiplication. Rank 49 (Strassen).
    #
    # Block 2: j' in [0, 3], k=4
    # C[i, 4] += sum_{j'=0}^3 A[i, j'] B[j', 4]
    # This is a 4x4x1 multiplication. Rank 16 (Naive).
    #
    # Block 3: j=4, k' in [0, 3]
    # C[i, k'] += A[i, 4] B[4, k']
    # This is a 4x1x4 multiplication. Rank 16 (Naive).
    #
    # Block 4: j=4, k=4
    # C[i, 4] += A[i, 4] B[4, 4]
    # This is a 4x1x1 multiplication. Rank 4 (Naive).
    #
    # Total Rank: 49 + 16 + 16 + 4 = 85.

    rank = 85
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a naive decomposition for a sub-block
    def add_naive_block(i_start, i_end, j_start, j_end, k_start, k_end, offset=0):
        nonlocal col
        for i in range(i_start, i_end):
            for j in range(j_start, j_end):
                for k in range(k_start, k_end):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
                    if col >= rank:
                        raise ValueError("Rank exceeded")

    # Helper to add Strassen decomposition for 4x4x4
    # Strassen factors for 2x2x2
    # S_U, S_V, S_W are 4x7 matrices
    # We need to map 2x2x2 to 4x4x4 by Kronecker product with 2x2 identity?
    # Or just use the known 4x4x4 Strassen factors.
    # R(4,4,4) = 49.
    # This is the Kronecker product of the 2x2x2 Strassen tensor (rank 7) with itself?
    # 7 * 7 = 49. Yes.
    # So we take the 2x2 Strassen factors and Kronecker product them with 2x2 identity?
    # No, the 4x4x4 Strassen decomposition is the tensor product of the 2x2x2 Strassen tensor with itself.
    
    # 2x2 Strassen factors (U, V, W) are 4x7.
    # Let S_U, S_V, S_W be the 4x7 factors for 2x2x2.
    # The 4x4x4 factors are S_U ⊗ I_2, S_V ⊗ I_2, S_W ⊗ I_2?
    # No, the tensor product of the tensors.
    # If T = S_U ⊗ S_V ⊗ S_W, then T ⊗ T = (S_U ⊗ S_U) ⊗ (S_V ⊗ S_V) ⊗ (S_W ⊗ S_W).
    # The factors for T ⊗ T are S_U ⊗ S_U, S_V ⊗ S_V, S_W ⊗ S_W.
    # These are 16x49 matrices.
    
    # 2x2 Strassen factors:
    # U:
    # 1 0 1 0 1 0 0
    # 0 0 1 0 0 1 0
    # 0 1 0 0 0 0 1
    # 0 0 0 1 0 0 1
    # Wait, standard Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # U (A factors):
    # Row 0 (A11): 1, 0, 1, 0, 1, -1, 0
    # Row 1 (A12): 0, 0, 0, 0, 1, 0, -1
    # Row 2 (A21): 0, 1, 0, 0, 0, 1, 0
    # Row 3 (A22): 1, 1, 0, 1, 0, 0, 1
    #
    # V (B factors):
    # Row 0 (B11): 1, 1, 0, -1, 0, 1, 0
    # Row 1 (B12): 0, 0, 0, 1, 0, 1, 0
    # Row 2 (B21): 1, 0, 1, 0, 0, 0, 1
    # Row 3 (B22): 1, 0, -1, 0, 1, 0, 1
    #
    # W (C factors):
    # Row 0 (C11): 1, 0, 0, 0, 0, 0, 0
    # Row 1 (C12): 0, 0, 0, 0, 1, 0, 0
    # Row 2 (C21): 0, 1, 1, 0, 0, 0, 0
    # Row 3 (C22): 1, -1, 0, 1, 0, 0, 0
    
    # We need the 4x4x4 factors.
    # U_44 = S_U ⊗ S_U?
    # The indices of U_44 are (i, j) where i,j in 0..3.
    # i = 2*i1 + i2, j = 2*j1 + j2.
    # U_44[i, col] = S_U[i1, c1] * S_U[i2, c2]?
    # The column index col = 7*c1 + c2.
    
    S_U = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, -1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, 1]
    ]
    S_V = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 0, 1, 0, 1, 0],
        [1, 0, 1, 0, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ]
    S_W = [
        [1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 1, 1, 0, 0, 0, 0],
        [1, -1, 0, 1, 0, 0, 0]
    ]
    
    # Construct 4x4x4 factors (16x49)
    # U_44[i, c] where i in 0..15, c in 0..48
    # i = 4*i1 + i2? No, i is index in 4x4 matrix.
    # The Kronecker product of two 2x2 matrices results in a 4x4 matrix.
    # Index mapping: i = 2*i1 + i2.
    # So i1 = i // 2, i2 = i % 2.
    
    for c1 in range(7):
        for c2 in range(7):
            col_idx = c1 * 7 + c2
            for i in range(4):
                i1 = i // 2
                i2 = i % 2
                # U factor for A
                val_u = S_U[i1][c1] * S_U[i2][c2]
                # Map i to global index in U (n*m = 4*5 = 20)
                # For the 4x4x4 block, i is 0..3, j is 0..3.
                # Global index in U is i*m + j = i*5 + j.
                # Wait, the sub-block is A[0:4, 0:4].
                # So global i is 0..3, global j is 0..3.
                # U_global[i*5 + j][col_idx] = val_u
                # We need to fill this for all j in 0..3.
                # But val_u depends on i only?
                # No, in the Kronecker product, the factor is U_44[i, j, c] = S_U[i1, c1] S_U[i2, c2] * S_U[j1, c1] S_U[j2, c2]?
                # No, the tensor product is (S_U ⊗ S_U).
                # The entry is S_U[i1, c1] * S_U[i2, c2] for the first factor?
                # The factor matrix for the tensor product T ⊗ T is U ⊗ U.
                # (U ⊗ U)[i, c] = U[i1, c1] * U[i2, c2] where i = 2*i1 + i2, c = 7*c1 + c2.
                # This U is 4x7. So U ⊗ U is 16x49.
                # The rows of U ⊗ U correspond to the 4x4 matrix indices (flattened).
                # So for a fixed col_idx, we fill U_global[i*5 + j] for i,j in 0..3.
                # But the factor is only for the "A" part.
                # The "A" part is 4x4.
                # So we fill U[i*5 + j] for i,j in 0..3.
                # The value is (U_44)[i*4+j][col_idx].
                
                # Let's compute U_44[i*4+j][col_idx]
                # i1 = i//2, i2 = i%2
                # j1 = j//2, j2 = j%2
                # Val = S_U[i1][c1] * S_U[i2][c2] * S_U[j1][c1] * S_U[j2][c2]?
                # No, the factor is U ⊗ U.
                # (U ⊗ U)[idx, col] = U[idx1, col1] * U[idx2, col2].
                # Here idx is the flattened index of the 4x4 matrix.
                # So for a given i (row of A) and j (col of A), the value is:
                # U_44[i, j, col] = S_U[i1, c1] * S_U[i2, c2] * S_U[j1, c1] * S_U[j2, c2]?
                # No, the Kronecker product of matrices U (4x7) and U (4x7) is (16x49).
                # The entry at (i*4+j, c1*7+c2) is U[i, c1] * U[j, c2]?
                # No, U is 4x7. U ⊗ U is (4*4) x (7*7).
                # Entry (i*4+j, c1*7+c2) = U[i, c1] * U[j, c2].
                # This corresponds to A[i, j] being multiplied by S_U[i, c1] S_U[j, c2].
                # This is correct for the A factor.
                
                for j in range(4):
                    j1 = j // 2
                    j2 = j % 2
                    val = S_U[i1][c1] * S_U[i2][c2] * S_U[j1][c1] * S_U[j2][c2]
                    # Wait, U[i, c1] is S_U[i1][c1] * S_U[i2][c2]?
                    # No, S_U is 4x7.
                    # In the 2x2 case, S_U is 4x7.
                    # In the 4x4 case, the factor is S_U ⊗ S_U?
                    # No, the 2x2 Strassen factor S_U is 4x7.
                    # The 4x4 Strassen factor is S_U ⊗ S_U?
                    # The 2x2 tensor is T_2.
                    # The 4x4 tensor is T_2 ⊗ T_2.
                    # The factor for T_2 ⊗ T_2 is S_U ⊗ S_U.
                    # S_U is 4x7. S_U ⊗ S_U is 16x49.
                    # Entry (i*4+j, c1*7+c2) = S_U[i, c1] * S_U[j, c2].
                    # Here i, j are 0..3.
                    # So val = S_U[i][c1] * S_U[j][c2].
                    
                    # My previous calculation was wrong.
                    # Let's use val = S_U[i][c1] * S_U[j][c2].
                    
                    U[i * m + j][col_idx] += S_U[i][c1] * S_U[j][c2]
                    # Wait, V and W also need to be filled.
                    
            # Fill V and W for this col_idx
            for j in range(4):
                for k in range(4):
                    # V factor: S_V[j][c1] * S_V[k][c2]?
                    # V is for B (4x4).
                    # V_44[j*4+k, col] = S_V[j, c1] * S_V[k, c2].
                    val_v = S_V[j][c1] * S_V[k][c2]
                    V[j * p + k][col_idx] += val_v
                    
            for k in range(4):
                for i in range(4):
                    # W factor: S_W[k][c1] * S_W[i][c2]?
                    # W is for C (4x4).
                    # W_44[k*4+i, col] = S_W[k, c1] * S_W[i, c2].
                    val_w = S_W[k][c1] * S_W[i][c2]
                    W[k * n + i][col_idx] += val_w
                    
            col += 1
            
    # Now add the naive blocks
    # Block 2: 4x4x1 (j' in 0..3, k=4)
    # i in 0..3, j in 0..3, k=4
    for i in range(4):
        for j in range(4):
            U[i * m + j][col] = 1
            V[j * p + 4][col] = 1
            W[4 * n + i][col] = 1
            col += 1
            
    # Block 3: 4x1x4 (j=4, k' in 0..3)
    # i in 0..3, j=4, k in 0..3
    for i in range(4):
        for k in range(4):
            U[i * m + 4][col] = 1
            V[4 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            
    # Block 4: 4x1x1 (j=4, k=4)
    # i in 0..3, j=4, k=4
    for i in range(4):
        U[i * m + 4][col] = 1
        V[4 * p + 4][col] = 1
        W[4 * n + i][col] = 1
        col += 1
        
    if col != rank:
        raise ValueError(f"Column count {col} does not match rank {rank}")
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
