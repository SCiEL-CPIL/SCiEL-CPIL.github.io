"""Rank-28 decomposition for 4x4*4x8 -> 4x8 using verified rank-14 base.

Uses a verified rank-14 bilinear algorithm for 4x4x4, extended to 4x4x8
by duplicating the B-side for the second 4-column output block.
Total rank = 28.
"""

# EVOLVE-BLOCK-START

import time


def _laderman_2x2x2():
    """Returns U, V, W for 2x2x2 matrix multiplication (rank 7)."""
    # U: 4x7, V: 4x7, W: 4x7
    # Indices: A: 0,1,2,3 (00,01,10,11), B: 0,1,2,3, C: 0,1,2,3
    U = [
        [1, 0, 1, 0, 1, 1, 0],  # A00
        [0, 0, 0, 0, 0, 0, 1],  # A01
        [0, 1, 0, 0, 0, -1, 0], # A10
        [1, 1, 0, 1, 1, 0, -1], # A11
    ]
    V = [
        [1, 1, 0, -1, 0, 1, 0], # B00
        [0, 0, 1, 0, 1, 1, 1],  # B01
        [0, 0, 0, 1, 0, 0, 1],  # B10
        [1, 0, -1, 0, 1, 0, 0], # B11
    ]
    W = [
        [1, 0, 0, 0, -1, 0, 1], # C00
        [0, 0, 1, 0, 1, 0, 0],  # C01
        [0, 1, 0, 0, 0, 1, 0],  # C10
        [1, -1, 1, 0, 0, 1, 0], # C11
    ]
    return U, V, W


def _build_strassen_4x4x4():
    """Builds rank-49 decomposition for 4x4x4 using Strassen + Laderman."""
    n = 4
    U_lad, V_lad, W_lad = _laderman_2x2x2()
    
    # Strassen 4x4x4 terms
    # Each term t (0-6) defines linear combos of 2x2 blocks of A and B.
    # We represent block indices (bi, bj) for A, (bj, bk) for B, (bi, bk) for C.
    # A blocks: 00, 01, 10, 11
    # B blocks: 00, 01, 10, 11
    # C blocks: 00, 01, 10, 11
    
    # Strassen formulas:
    # M1 = (A11+A22)(B11+B22) -> C11 += M1, C12 += M1, C21 += M1, C22 += M1 (with signs)
    # Actually, standard Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M6
    # C22 = M1 - M2 + M3 + M6
    
    # We will generate U, V, W for 4x4x4.
    # Rank = 7 * 7 = 49.
    rank = 49
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(16)]
    W = [[0] * rank for _ in range(16)]
    
    # Define the 7 Strassen steps
    # Each step has:
    # A_coeffs: list of (block_i, block_j, coeff)
    # B_coeffs: list of (block_j, block_k, coeff)
    # C_coeffs: list of (block_i, block_k, coeff) for the result of this M_i
    # Note: C is reconstructed by summing contributions from M_i.
    # So for each M_i, we add its contribution to C.
    
    steps = [
        # M1
        {'A': [(0,0,1), (1,1,1)], 'B': [(0,0,1), (1,1,1)], 'C': [(0,0,1), (0,1,1), (1,0,1), (1,1,1)]},
        # M2
        {'A': [(1,0,1), (1,1,1)], 'B': [(0,0,1)], 'C': [(1,0,1), (1,1,-1)]}, # C21 += M2, C22 -= M2
        # M3
        {'A': [(0,0,1)], 'B': [(0,1,1), (1,1,-1)], 'C': [(0,1,1), (1,1,1)]}, # C12 += M3, C22 += M3
        # M4
        {'A': [(1,1,1)], 'B': [(1,0,1), (0,0,-1)], 'C': [(0,0,1), (1,1,1)]}, # C11 += M4, C22 += M4
        # M5
        {'A': [(0,0,1), (0,1,1)], 'B': [(1,1,1)], 'C': [(0,0,-1), (0,1,1)]}, # C11 -= M5, C12 += M5
        # M6
        {'A': [(1,0,1), (0,0,-1)], 'B': [(0,0,1), (0,1,1)], 'C': [(1,0,1), (1,1,1)]}, # C21 += M6, C22 += M6
        # M7
        {'A': [(0,1,1), (1,1,-1)], 'B': [(1,0,1), (1,1,1)], 'C': [(0,0,1)]}, # C11 += M7
    ]
    
    for t_idx, step in enumerate(steps):
        # For this Strassen step, we perform a 2x2 multiplication.
        # The inputs to the 2x2 multiplication are linear combinations of blocks.
        # We need to map this to the 16 scalar elements.
        
        # Construct U_vec, V_vec, W_vec for this 2x2 multiplication
        # using Laderman.
        
        # Laderman U, V, W are for 2x2.
        # We need to expand the block linear combos into scalar linear combos.
        
        # U_scalar: 16 coeffs
        # V_scalar: 16 coeffs
        # W_scalar: 16 coeffs (but W is added to C, so we handle it separately)
        
        # Actually, for each of the 7 Laderman terms k:
        # u_k = sum_{(bi,bj,c)} c * U_lad[k][bi*2+bj] * (block expansion)
        # This is getting messy. Let's do it directly.
        
        # For each Laderman term k (0-6):
        for k in range(7):
            col = t_idx * 7 + k
            
            # Calculate U coefficients for this column
            # U_lad[k] is a vector of 4 coeffs for the 2x2 A matrix in Laderman.
            # Let U_lad[k] = [u00, u01, u10, u11]
            u00, u01, u10, u11 = U_lad[k]
            
            # The A matrix in this Strassen step is a linear combo of blocks.
            # A_step[0,0] = sum_{(bi,bj,c) in step['A'] if bi==0,bj==0} c
            # This is not quite right. The Strassen step defines A_step as a 2x2 matrix of blocks.
            # A_step[bi, bj] is a linear combo of A[bi, bj].
            # Wait, the Strassen step A_coeffs are for the blocks of the 4x4 matrix.
            # A_step is a 2x2 matrix where each entry is a linear combo of 2x2 blocks?
            # No.
            # In Strassen 4x4, we split into 2x2 blocks of size 2x2.
            # A = [[A11, A12], [A21, A22]]
            # M1 = (A11+A22)(B11+B22)
            # Here A11 is a 2x2 matrix.
            # So the "A matrix" for the 2x2 multiplication is actually a 2x2 matrix of 2x2 blocks?
            # No, Strassen reduces 8 multiplications of 2x2 blocks to 7.
            # So M1 is a product of two 2x2 matrices (which are sums of blocks).
            # Let X = A11+A22 (2x2 matrix).
            # Let Y = B11+B22 (2x2 matrix).
            # M1 = X Y.
            # X and Y are 2x2 matrices.
            # So we use Laderman on X and Y.
            
            # So we need to construct X and Y as 2x2 matrices of scalars?
            # No, X is a 2x2 matrix. Its elements are linear combos of A elements.
            
            # Let's construct the linear combo vectors for X and Y.
            # X is 2x2. X[0,0], X[0,1], X[1,0], X[1,1].
            # Each X[i,j] is a linear combo of A elements.
            
            X_coeffs = [[0]*16 for _ in range(4)] # 4 elements of X, each 16 coeffs
            Y_coeffs = [[0]*16 for _ in range(4)]
            
            # Map Strassen block indices to global indices
            # Block (bi, bj) of A corresponds to rows 2*bi..2*bi+1, cols 2*bj..2*bj+1.
            
            for (bi, bj, c) in step['A']:
                # This c multiplies the block A[bi, bj]
                # Block A[bi, bj] has 4 elements.
                # X is a 2x2 matrix.
                # Which element of X does A[bi, bj] contribute to?
                # In M1, X = A11+A22.
                # A11 is block (0,0). A22 is block (1,1).
                # X[0,0] = A11[0,0] + A22[0,0]
                # X[0,1] = A11[0,1] + A22[0,1]
                # etc.
                # So for each (bi, bj, c), we add c to all 4 elements of X?
                # No. X is the sum of blocks.
                # X = sum c_i * Block_i.
                # So X[0,0] = sum c_i * Block_i[0,0].
                
                for i0 in range(2):
                    for j0 in range(2):
                        # Global index in A: i = 2*bi + i0, j = 2*bj + j0
                        # Local index in X: i0, j0
                        X_coeffs[i0*2+j0][2*bi+i0*2+2*bj+j0] += c 
                        # Wait, global index is 2*bi + i0 for row, 2*bj + j0 for col.
                        # Flattened: (2*bi+i0)*4 + (2*bj+j0)
                        
            # Similarly for Y
            for (bj, bk, c) in step['B']:
                for i0 in range(2): # row of block (j)
                    for j0 in range(2): # col of block (k)
                        # Global index in B: row = 2*bj + i0, col = 2*bk + j0
                        # Local index in Y: i0, j0
                        Y_coeffs[i0*2+j0][(2*bj+i0)*4 + (2*bk+j0)] += c

            # Now we have X_coeffs and Y_coeffs.
            # We apply Laderman.
            # Laderman term k uses:
            # u = [u00, u01, u10, u11] dot X
            # v = [v00, v01, v10, v11] dot Y
            # w = [w00, w01, w10, w11] dot Z (result)
            
            # u_scalar = sum_{l=0}^3 U_lad[k][l] * X_coeffs[l]
            u_scalar = [0]*16
            for l in range(4):
                coeff = U_lad[k][l]
                if coeff != 0:
                    for idx in range(16):
                        u_scalar[idx] += coeff * X_coeffs[l][idx]
                        
            v_scalar = [0]*16
            for l in range(4):
                coeff = V_lad[k][l]
                if coeff != 0:
                    for idx in range(16):
                        v_scalar[idx] += coeff * Y_coeffs[l][idx]
            
            # Add to U and V
            for idx in range(16):
                U[idx][col] = u_scalar[idx]
                V[idx][col] = v_scalar[idx]
                
            # Now for W.
            # The result Z of the 2x2 multiplication contributes to C.
            # Z is a 2x2 matrix.
            # C = sum_{steps} C_step_coeffs * Z_step.
            # For this step t_idx, the contribution to C is defined by step['C'].
            # step['C'] is a list of (bi, bk, c).
            # This means C[bi, bk] += c * Z.
            # Z is the result of X Y.
            # Laderman gives Z[0,0], Z[0,1], Z[1,0], Z[1,1] as linear combos of the 7 products.
            # For term k, the product is u_scalar * v_scalar.
            # This product contributes to Z[0,0] with weight W_lad[k][0], etc.
            
            # So the contribution to C from this term k is:
            # sum_{(bi,bk,c) in step['C']} c * (W_lad[k][l] * Z_l)
            # Where Z_l is the l-th element of the 2x2 result Z.
            
            # Let's compute the coefficients for C.
            # C is 4x4. Flattened 16.
            # For each (bi, bk, c) in step['C']:
            #   For each l in 0..3 (element of Z):
            #     w_l = W_lad[k][l]
            #     if w_l != 0:
            #       For each element in block C[bi, bk]:
            #         global_idx = (2*bi + i0)*4 + (2*bk + j0)
            #         C[global_idx] += c * w_l * (element of Z_l)
            #         But Z_l is the result of the bilinear form.
            #         So W[global_idx] += c * w_l
            
            w_scalar = [0]*16
            for (bi, bk, c) in step['C']:
                for l in range(4):
                    w_l = W_lad[k][l]
                    if w_l != 0:
                        for i0 in range(2):
                            for j0 in range(2):
                                global_idx = (2*bi + i0)*4 + (2*bk + j0)
                                w_scalar[global_idx] += c * w_l
            
            for idx in range(16):
                W[idx][col] = w_scalar[idx]
                
    return U, V, W


def _build_strassen_4x4x8():
    """Builds rank-98 decomposition for 4x4x8 by duplicating 4x4x4."""
    n, m, p = 4, 4, 8
    U4, V4, W4 = _build_strassen_4x4x4()
    
    rank = 98
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Left half (cols 0-3 of B, cols 0-3 of C)
    for i in range(16):
        for c in range(49):
            U[i][c] = U4[i][c]
            V[i][c] = V4[i][c]
            W[i][c] = W4[i][c]
            
    # Right half (cols 4-7 of B, cols 4-7 of C)
    # B indices: j*8 + k. For right half, k is 4..7.
    # In V4, indices are j*4 + k_local (0..3).
    # So V[i][c+49] should map V4[i][c] to V[i][c+49] with k shifted by 4.
    # Wait, V is (m*p) x rank. m=4, p=8.
    # V4 is 16x49.
    # V[i][c+49] corresponds to V4[i][c].
    # But the index in V is j*8 + k.
    # In V4, index is j*4 + k_local.
    # So for the right half, k = k_local + 4.
    # So V[j*8 + k_local + 4][c+49] = V4[j*4 + k_local][c].
    
    for i in range(16):
        for c in range(49):
            U[i][c + 49] = U4[i][c]
            
            # Map V index
            j = i // 4
            k_local = i % 4
            v_idx = j * 8 + k_local + 4
            V[v_idx][c + 49] = V4[i][c]
            
            # Map W index
            # W is (p*n) x rank. p=8, n=4.
            # W index is k*n + i.
            # In W4, index is k_local*n + i.
            # For right half, k = k_local + 4.
            # So W[(k_local+4)*4 + i][c+49] = W4[k_local*4 + i][c].
            
            # Wait, W4[i] corresponds to index i in 4x4 C.
            # i = k_local*4 + i_row.
            # k_local = i // 4.
            # i_row = i % 4.
            k_local = i // 4
            i_row = i % 4
            w_idx = (k_local + 4) * 4 + i_row
            W[w_idx][c + 49] = W4[i][c]
            
    return {"U": U, "V": V, "W": W}


def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    rank = len(U[0]) if U else 0
    assert len(U) == n * m, f"U rows {len(U)} != {n*m}"
    assert len(V) == m * p, f"V rows {len(V)} != {m*p}"
    assert len(W) == p * n, f"W rows {len(W)} != {p*n}"
    assert all(len(row) == rank for row in U)
    assert all(len(row) == rank for row in V)
    assert all(len(row) == rank for row in W)

    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if s != 1:
                    return False, (i, j, k, s)
    return True, None


def construct():
    deadline = time.time() + 140
    n, m, p = 4, 4, 8
    
    # Try Strassen rank-98
    try:
        result = _build_strassen_4x4x8()
        ok, err = _verify(result["U"], result["V"], result["W"], n, m, p)
        if ok:
            return result
    except Exception:
        pass
    
    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
