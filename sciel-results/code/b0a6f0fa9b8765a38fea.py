import time
import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 8
    deadline = time.time() + 140
    
    # Helper to generate Strassen decomposition for n x n x n
    # Returns U, V, W as numpy arrays of shape (n*n, rank)
    def get_strassen(n):
        if n == 1:
            return np.ones((1, 1)), np.ones((1, 1)), np.ones((1, 1))
        
        # Recursively get decomposition for n//2
        U_half, V_half, W_half = get_strassen(n // 2)
        r_half = U_half.shape[1]
        r_full = 7 * r_half
        
        U_full = np.zeros((n * n, r_full))
        V_full = np.zeros((n * n, r_full))
        W_full = np.zeros((n * n, r_full))
        
        # Strassen formulas for blocks
        # A = [[A11, A12], [A21, A22]]
        # B = [[B11, B12], [B21, B22]]
        # Indices in A: 0..n-1 are rows, 0..n-1 are cols
        # Block A11: rows 0..n/2-1, cols 0..n/2-1 -> index i*n+j
        # Block A12: rows 0..n/2-1, cols n/2..n-1
        # Block A21: rows n/2..n-1, cols 0..n/2-1
        # Block A22: rows n/2..n-1, cols n/2..n-1
        
        # We define the 7 products M1..M7
        # Each Mk involves sums/diffs of blocks.
        # We need to express the coefficients for U, V, W in terms of the half-size coefficients.
        
        # Let's define the block index mapping.
        # For a block (br, bc) where br, bc in {0,1}, the local indices (i, j) in 0..n/2-1
        # map to global index (br * n/2 + i) * n + (bc * n/2 + j).
        
        # M1 = (A11 + A22) (B11 + B22)
        # U for M1:
        #   A11 part: U_half[i,j] at global (0,0)
        #   A22 part: U_half[i,j] at global (1,1)
        # V for M1:
        #   B11 part: V_half[i,j] at global (0,0)
        #   B22 part: V_half[i,j] at global (1,1)
        # W for M1:
        #   Result C11 += M1, C22 += M1
        #   W_half[i,j] at global (0,0) and (1,1)
        
        # This is getting complex to code generically. 
        # Let's hardcode the 7 terms for the recursive step.
        
        half = n // 2
        
        def add_block(U, V, W, u_coeff, v_coeff, w_coeff, u_block, v_block, w_block, offset):
            # u_block, v_block, w_block are (br, bc) for A, B, C blocks
            # u_coeff, v_coeff, w_coeff are scalars (1 or -1)
            # We add the half-size decomposition scaled by coeffs to the full matrices
            
            for r in range(r_half):
                for i in range(half):
                    for j in range(half):
                        # U index
                        gu = (u_block[0] * half + i) * n + (u_block[1] * half + j)
                        U[gu, offset + r] += u_coeff * U_half[i * half + j, r]
                        
                        # V index
                        gv = (v_block[0] * half + i) * n + (v_block[1] * half + j)
                        V[gv, offset + r] += v_coeff * V_half[i * half + j, r]
                        
                        # W index
                        gw = (w_block[0] * half + i) * n + (w_block[1] * half + j)
                        W[gw, offset + r] += w_coeff * W_half[i * half + j, r]

        # M1 = (A11 + A22) (B11 + B22) -> C11 += M1, C22 += M1
        # This term contributes to C11 and C22.
        # In Strassen, the products are scalar multiplications of matrices.
        # The decomposition of the product of two matrices X, Y is sum u*x*v*y*w.
        # Here X = A11+A22, Y = B11+B22.
        # So U coefficients for A11 and A22 are 1.
        # V coefficients for B11 and B22 are 1.
        # W coefficients for C11 and C22 are 1.
        
        # We need to call add_block for each component.
        # But add_block adds to U, V, W simultaneously.
        # Here, the U part comes from A11 and A22.
        # The V part comes from B11 and B22.
        # The W part goes to C11 and C22.
        
        # My add_block function assumes a single block for U, V, W.
        # I need a more flexible function or to call it multiple times?
        # No, the rank-1 term is u * v * w.
        # If u = u1 + u2, then (u1+u2)v w = u1 v w + u2 v w.
        # So we can split the term.
        
        # Term M1:
        # Part 1: A11 * B11 -> C11, C22
        # Part 2: A11 * B22 -> C11, C22
        # Part 3: A22 * B11 -> C11, C22
        # Part 4: A22 * B22 -> C11, C22
        
        # This would increase rank! Strassen combines them.
        # The trick is that the "multiplication" is bilinear.
        # (A11+A22)(B11+B22) is ONE multiplication in the algebra of blocks.
        # But in the tensor decomposition, we are decomposing the bilinear map.
        # The Strassen algorithm works because it replaces 8 multiplications with 7.
        # The decomposition of the product of two n x n matrices is reused.
        
        # So, for the term M1, we use the decomposition of (n x n) * (n x n).
        # The inputs to this sub-problem are X = A11+A22 and Y = B11+B22.
        # The output is Z = X*Y.
        # Z contributes to C11 and C22.
        
        # So, for each rank-1 component of the sub-problem:
        # u_sub * x_sub * v_sub * y_sub * w_sub
        # x_sub is a linear combination of A11, A22.
        # y_sub is a linear combination of B11, B22.
        # z_sub is a linear combination of C11, C22.
        
        # How to map this to the global U, V, W?
        # U_global should have entries for A11 and A22.
        # If x_sub = 1 * A11 + 1 * A22, then U_global has 1 at A11 indices and 1 at A22 indices.
        
        # So, I need to know the coefficients of the linear combinations in Strassen.
        
        # M1: X = A11 + A22, Y = B11 + B22, Z = C11 + C22
        # M2: X = A21 + A22, Y = B11, Z = C21
        # M3: X = A11, Y = B12 - B22, Z = C12
        # M4: X = A22, Y = B21 - B11, Z = C21
        # M5: X = A11 + A12, Y = B22, Z = C12
        # M6: X = A21 - A11, Y = B11 + B12, Z = C11
        # M7: X = A12 - A22, Y = B21 + B22, Z = C22
        
        # Let's define a helper to add a Strassen term.
        # term(u_blocks, v_blocks, w_blocks, coeffs_u, coeffs_v, coeffs_w)
        # where u_blocks is list of (br, bc) and coeffs_u is list of scalars.
        
        def add_term(u_blocks, v_blocks, w_blocks, offset):
            # u_blocks: list of (br, bc, coeff)
            # v_blocks: list of (br, bc, coeff)
            # w_blocks: list of (br, bc, coeff)
            
            # We iterate over the sub-rank r
            for r in range(r_half):
                # We need to add the contribution of the r-th rank-1 component of the sub-problem.
                # The sub-problem is X*Y=Z.
                # The r-th component is u_r * x_r * v_r * y_r * w_r * z_r.
                # Here x_r is a specific vector in the n x n space? No.
                # The sub-decomposition is for n x n matrices.
                # U_half[i,j, r] is the coefficient for A[i,j] in the r-th term.
                
                # So, for the global U, we add U_half[i,j, r] * coeff_A_block to the appropriate global indices.
                
                for (br_u, bc_u, cu) in u_blocks:
                    for i in range(half):
                        for j in range(half):
                            gu = (br_u * half + i) * n + (bc_u * half + j)
                            U_full[gu, offset + r] += cu * U_half[i * half + j, r]
                            
                for (br_v, bc_v, cv) in v_blocks:
                    for i in range(half):
                        for j in range(half):
                            gv = (br_v * half + i) * n + (bc_v * half + j)
                            V_full[gv, offset + r] += cv * V_half[i * half + j, r]
                            
                for (br_w, bc_w, cw) in w_blocks:
                    for i in range(half):
                        for j in range(half):
                            gw = (br_w * half + i) * n + (bc_w * half + j)
                            W_full[gw, offset + r] += cw * W_half[i * half + j, r]

        # Define the 7 terms
        # M1: (A11+A22)(B11+B22) -> C11+C22
        add_term([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], 0)
        
        # M2: (A21+A22)B11 -> C21
        add_term([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1)], r_half)
        
        # M3: A11(B12-B22) -> C12
        add_term([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1)], 2*r_half)
        
        # M4: A22(B21-B11) -> C21
        add_term([(1,1,1)], [(1,0,1), (0,0,-1)], [(1,0,1)], 3*r_half)
        
        # M5: (A11+A12)B22 -> C12
        add_term([(0,0,1), (0,1,1)], [(1,1,1)], [(0,1,1)], 4*r_half)
        
        # M6: (A21-A11)(B11+B12) -> C11
        add_term([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(0,0,1)], 5*r_half)
        
        # M7: (A12-A22)(B21+B22) -> C22
        add_term([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(1,1,1)], 6*r_half)
        
        return U_full, V_full, W_full

    # Get 4x4x4 Strassen decomposition
    U4, V4, W4 = get_strassen(4)
    r4 = U4.shape[1] # Should be 49
    
    # Construct 4x4x8 decomposition
    # Two blocks for k=0..3 and k=4..7
    total_rank = 2 * r4
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    for i in range(n):
        for j in range(m):
            a = i * m + j
            # U is same for both blocks
            U[a, :r4] = U4[i * m + j, :]
            U[a, r4:] = U4[i * m + j, :]
            
    for j in range(m):
        for k in range(p):
            b = j * p + k
            if k < 4:
                # Block 1
                V[b, :r4] = V4[j * 4 + k, :]
            else:
                # Block 2
                k1 = k - 4
                V[b, r4:] = V4[j * 4 + k1, :]
                
    for k in range(p):
        for i in range(n):
            c = k * n + i
            if k < 4:
                # Block 1
                W[c, :r4] = W4[k * n + i, :]
            else:
                # Block 2
                k1 = k - 4
                W[c, r4:] = W4[k1 * n + i, :]
                
    # Verify
    # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    # Check a few entries
    
    result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return result
# EVOLVE-BLOCK-END
