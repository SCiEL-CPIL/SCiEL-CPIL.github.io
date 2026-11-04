"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # A is n×m (4×4), B is m×p (4×7), C is n×p (4×7)
    
    start_time = time.time()
    DEADLINE = 145.0  # Leave 5s margin before 150s kill
    
    # Schoolbook fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    try:
        # Strassen rank-7 decomposition for 2×2×2 matrix multiplication
        # The tensor T[a,b,c] = delta_{a+c, b} where a,b,c in {0,1}
        # with indices (a0,a1,b0,b1) for input and (c0,c1) for output
        # Rank-1 terms: (u, v, w) where u in R^4 (for A), v in R^4 (for B), w in R^2 (for C)
        strassen_2x2 = [
            # (u for A: [A00, A01, A10, A11], v for B: [B00, B01, B10, B11], w for C: [C00, C01])
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0]),      # M1: (A00+A11)(B00+B11) -> C00, C11
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0]),       # M2: (A10+A11)B00 -> C10
            ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0]),      # M3: A00(B01-B11) -> C01
            ([0, 0, 0, 1], [0, 0, 1, -1], [0, 0]),      # M4: A11(B10-B00) -> C10 correction
            ([1, 1, 0, 0], [0, 0, 0, 1], [0, 0]),       # M5: (A00+A01)B11 -> C01 correction
            ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0]),      # M6: (A10-A00)(B00+B01) -> C10 correction
            ([0, 1, 0, -1], [0, 0, 1, 1], [0, 0]),      # M7: (A01-A11)(B10+B11) -> C11 correction
        ]
        
        # Actually, let me use the standard Strassen decomposition properly.
        # For 2×2 matrix multiplication: C = A * B
        # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
        # C = [[c00, c01], [c10, c11]]
        # 
        # Strassen:
        # M1 = (a00+a11)(b00+b11)
        # M2 = (a10+a11)b00
        # M3 = a00(b01-b11)
        # M4 = a11(b10-b00)
        # M5 = (a00+a01)b11
        # M6 = (a10-a00)(b00+b01)
        # M7 = (a01-a11)(b10+b11)
        #
        # c00 = M1 + M4 - M5 + M7
        # c01 = M3 + M5
        # c10 = M2 + M4
        # c11 = M1 - M2 + M3 + M6
        #
        # Each Mk is a product of a linear combination of A entries and B entries.
        # For the tensor decomposition, each rank-1 term corresponds to one Mk.
        # The rank-1 term has:
        #   u: coefficient vector for A entries (a00, a01, a10, a11)
        #   v: coefficient vector for B entries (b00, b01, b10, b11)
        #   w: coefficient vector for C entries (c00, c01, c10, c11)
        #
        # But wait - in our problem, the "output" W has shape (p*n, rank) = (7*4, rank) = (28, rank)
        # So W[k*n+i] is the coefficient for C[i,k].
        # 
        # For the 2×2×2 case embedded in our problem, we need to be careful about
        # how the output indices map.
        
        # Let me think differently. The tensor for matrix multiplication M_{n,m,p} has
        # entries T[i,j,k] = delta_{i+k, j} where i in [0,n), j in [0,m), k in [0,p).
        # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
        # The decomposition is: T = sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r]
        # where (U[:,r] ⊗ V[:,r] ⊗ W[:,r])[i*m+j, j'*p+k, k'*n+i'] 
        #   = U[i*m+j, r] * V[j'*p+k, r] * W[k'*n+i', r]
        # And this should equal delta_{i+k', j} * delta_{j+j', k} ... 
        # 
        # Actually, the standard tensor for matrix multiplication C = A*B where
        # A is n×m, B is m×p, C is n×p:
        # T[A_index, B_index, C_index] = 1 if A_index and B_index are compatible
        # and C_index is the corresponding output.
        #
        # A_index = i*m + j (i in [0,n), j in [0,m))
        # B_index = j'*p + k (j' in [0,m), k in [0,p))
        # C_index = k'*n + i' (k' in [0,p), i' in [0,n))
        #
        # T[A_idx, B_idx, C_idx] = 1 if j == j' and i == i' and k == k'
        #                          (i.e., the (i,j) element of A, (j,k) element of B
        #                           contribute to (i,k) element of C)
        #
        # So T[i*m+j, j*p+k, k*n+i] = 1 for valid i,j,k, and 0 otherwise.
        
        # For the schoolbook: each term r corresponds to a triple (i,j,k)
        # U[i*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+i, r] = 1
        
        # Now for Strassen: we group the 4×4×7 problem into blocks.
        # A is 4×4, which we can view as a 2×2 block matrix with 2×2 blocks.
        # B is 4×7, which we can view as... hmm, 7 is not a multiple of 2.
        # 
        # Let me think of it as:
        # A = [[A00, A01], [A10, A11]] where each Akℓ is 2×2
        # B is 4×7. We can split B into B = [B_left, B_right] where
        #   B_left is 4×4 (first 4 columns) and B_right is 4×3 (last 3 columns)
        # 
        # For C = A * B:
        # C = [[A00*B00 + A01*B10, A00*B01 + A01*B10],  (2×7)
        #      [A10*B00 + A11*B10, A10*B01 + A11*B10]]
        # 
        # Where B00, B10 are 2×2 blocks (rows 0-1 and 2-3 of first 4 cols of B)
        # B01, B11 are 2×2 blocks (rows 0-1 and 2-3 of first 4 cols... wait)
        
        # Let me re-index. B is 4×7. Let me write B as:
        # B = [[B00, B01, B02, B03],   (rows 0-1, cols 0-6)
        #      [B10, B11, B12, B13]]   (rows 2-3, cols 0-6)
        # where B00, B10 are 2×2 (cols 0-1), B01, B11 are 2×2 (cols 2-3),
        # B02, B12 are 2×1 (col 4), B03, B13 are 2×1 (cols 5-6)? 
        # No, that's awkward.
        
        # Better: Let me just handle the full 4×4×7 directly using a combination
        # of Strassen for the 4×4 part and schoolbook for the remaining columns.
        
        # Alternative approach: Use the known optimal rank for M_{4,4,7}.
        # The rank of M_{4,4,4} is known to be at most 49 (via Strassen: 7 * 7 = 49)
        # Actually, M_{4,4,4} has rank at most 49 using Strassen (4×4 = (2×2)^2,
        # each 2×2×2 has rank 7, so 4 blocks * 7 = 28? No...)
        
        # Let me just use Strassen properly for the 4×4×4 part and schoolbook for 4×4×3.
        
        # For 4×4×4: 
        # A is 4×4, B is 4×4, C is 4×4.
        # Using Strassen: 
        # A = [[A00, A01], [A10, A11]] (2×2 blocks of 2×2)
        # B = [[B00, B01], [B10, B11]] (2×2 blocks of 2×2)
        # 
        # The 7 Strassen products are 2×2 matrix multiplications:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        #
        # Each Mk is a 2×2 matrix. To compute each 2×2 multiplication, we need rank 7.
        # So total rank for 4×4×4 is 7*7 = 49.
        
        # For 4×4×3:
        # A is 4×4, B is 4×3, C is 4×3.
        # Schoolbook: 4*4*3 = 48.
        
        # Total: 49 + 48 = 97. But we can do better.
        
        # Actually, let me just implement the 4×4×4 Strassen decomposition and 
        # schoolbook for the rest.
        
        # First, let me define the Strassen decomposition for 2×2×2 properly.
        # The rank-1 terms for 2×2 matrix multiplication (C = A*B, A,B,C all 2×2):
        # We need (u, v, w) where u is 4-dim (coefficients for a00,a01,a10,a11),
        # v is 4-dim (coefficients for b00,b01,b10,b11), w is 4-dim (coefficients for c00,c01,c10,c11)
        
        # Strassen's 7 products:
        # M1 = (a00+a11)(b00+b11) -> contributes to c00 and c11
        # M2 = (a10+a11)b00 -> contributes to c10
        # M3 = a00(b01-b11) -> contributes to c01
        # M4 = a11(b10-b00) -> contributes to c10
        # M5 = (a00+a01)b11 -> contributes to c01
        # M6 = (a10-a00)(b00+b01) -> contributes to c10
        # M7 = (a01-a11)(b10+b11) -> contributes to c11
        
        # For the tensor decomposition, each product Mk corresponds to a rank-1 term.
        # The u vector for Mk gives coefficients for A entries.
        # The v vector for Mk gives coefficients for B entries.
        # The w vector for Mk gives coefficients for C entries.
        
        # M1: u = [1, 0, 0, 1] (a00 + a11), v = [1, 0, 0, 1] (b00 + b11)
        #     This product (a00+a11)(b00+b11) appears in:
        #     c00 = M1 + M4 - M5 + M7 -> coefficient of M1 in c00 is +1
        #     c11 = M1 - M2 + M3 + M6 -> coefficient of M1 in c11 is +1
        #     So w = [1, 0, 0, 1] (c00 + c11)
        
        # M2: u = [0, 0, 1, 1] (a10 + a11), v = [1, 0, 0, 0] (b00)
        #     c10 = M2 + M4 -> coefficient of M2 in c10 is +1
        #     c11 = M1 - M2 + M3 + M6 -> coefficient of M2 in c11 is -1
        #     So w = [0, 0, 1, -1]
        
        # M3: u = [1, 0, 0, 0] (a00), v = [0, 1, 0, -1] (b01 - b11)
        #     c01 = M3 + M5 -> coefficient of M3 in c01 is +1
        #     c11 = M1 - M2 + M3 + M6 -> coefficient of M3 in c11 is +1
        #     So w = [0, 1, 0, 1]
        
        # M4: u = [0, 0, 0, 1] (a11), v = [0, 0, 1, -1] (b10 - b00)
        #     c00 = M1 + M4 - M5 + M7 -> coefficient of M4 in c00 is +1
        #     c10 = M2 + M4 -> coefficient of M4 in c10 is +1
        #     So w = [1, 0, 1, 0]
        
        # M5: u = [1, 1, 0, 0] (a00 + a01), v = [0, 0, 0, 1] (b11)
        #     c00 = M1 + M4 - M5 + M7 -> coefficient of M5 in c00 is -1
        #     c01 = M3 + M5 -> coefficient of M5 in c01 is +1
        #     So w = [-1, 1, 0, 0]
        
        # M6: u = [-1, 0, 1, 0] (a10 - a00), v = [1, 1, 0, 0] (b00 + b01)
        #     c11 = M1 - M2 + M3 + M6 -> coefficient of M6 in c11 is +1
        #     So w = [0, 0, 0, 1]
        
        # M7: u = [0, 1, 0, -1] (a01 - a11), v = [0, 0, 1, 1] (b10 + b11)
        #     c00 = M1 + M4 - M5 + M7 -> coefficient of M7 in c00 is +1
        #     c11 = M1 - M2 + M3 + M6 -> wait, let me recheck.
        #     Actually: c11 = M1 - M2 + M3 + M6? Let me verify:
        #     c11 = a10*b01 + a11*b11
        #     M1 - M2 + M3 + M6 = (a00+a11)(b00+b11) - (a10+a11)b00 + a00(b01-b11) + (a10-a00)(b00+b01)
        #     = a00*b00 + a00*b11 + a11*b00 + a11*b11 - a10*b00 - a11*b00 + a00*b01 - a00*b11 + a10*b00 + a10*b01 - a00*b00 - a00*b01
        #     = (a00*b00 - a00*b00) + (a00*b11 - a00*b11) + (a11*b00 - a11*b00) + a11*b11 + (-a10*b00 + a10*b00) + (a10*b01) + (a00*b01 - a00*b01)
        #     = a11*b11 + a10*b01
        #     Hmm, that's not c11 = a10*b01 + a11*b11. Wait, it is! 
        #     So c11 = M1 - M2 + M3 + M6. But where does M7 go?
        #     Let me recheck the standard Strassen formulas:
        #     c00 = M1 + M4 - M5 + M7
        #     c01 = M3 + M5
        #     c10 = M2 + M4
        #     c11 = M1 - M2 + M3 + M6
        #     
        #     Wait, that doesn't use M7 in c11. Let me verify c00:
        #     c00 = a00*b00 + a01*b10
        #     M1 + M4 - M5 + M7 = (a00+a11)(b00+b11) + a11(b10-b00) - (a00+a01)b11 + (a01-a11)(b10+b11)
        #     = a00*b00 + a00*b11 + a11*b00 + a11*b11 + a11*b10 - a11*b00 - a00*b11 - a01*b11 + a01*b10 + a01*b11 - a11*b10 - a11*b11
        #     = a00*b00 + (a00*b11 - a00*b11) + (a11*b00 - a11*b00) + (a11*b11 - a11*b11) + (a11*b10 - a11*b10) + (-a01*b11 + a01*b11) + a01*b10
        #     = a00*b00 + a01*b10 = c00 ✓
        #     
        #     And c11 = a10*b01 + a11*b11
        #     M1 - M2 + M3 + M6 = (a00+a11)(b00+b11) - (a10+a11)b00 + a00(b01-b11) + (a10-a00)(b00+b01)
        #     = a00*b00 + a00*b11 + a11*b00 + a11*b11 - a10*b00 - a11*b00 + a00*b01 - a00*b11 + a10*b00 + a10*b01 - a00*b00 - a00*b01
        #     = a11*b11 + a10*b01 = c11 ✓
        #     
        #     So M7 is only used in c00. That's correct.
        #     
        #     So for M7: w = [1, 0, 0, 0] (only contributes to c00)
        
        strassen_2x2_terms = [
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),   # M1
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),  # M2
            ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),  # M3
            ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]),  # M4
            ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),  # M5
            ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),  # M6
            ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),  # M7
        ]
        
        # Now build the 4×4×4 decomposition using Strassen.
        # A is 4×4, indexed as A[i,j] for i,j in [0,4).
        # We divide A into 2×2 blocks:
        # A00 = A[0:2, 0:2], A01 = A[0:2, 2:4]
        # A10 = A[2:4, 0:2], A11 = A[2:4, 2:4]
        #
        # B is 4×4, similarly divided into B00, B01, B10, B11.
        # C is 4×4, similarly divided into C00, C01, C10, C11.
        #
        # The 7 Strassen products M1..M7 are each 2×2 matrix multiplications.
        # Each Mk = Xk * Yk where Xk is a 2×2 matrix (linear combo of A blocks)
        # and Yk is a 2×2 matrix (linear combo of B blocks).
        #
        # To compute Mk = Xk * Yk (both 2×2), we use the rank-7 Strassen decomposition
        # for 2×2×2. This gives 7 rank-1 terms per Mk, for 7*7 = 49 total terms.
        
        # For each Strassen product Mk, we need:
        # 1. The linear combination coefficients for A blocks to form Xk.
        # 2. The linear combination coefficients for B blocks to form Yk.
        # 3. Apply the 2×2×2 Strassen decomposition to compute Xk * Yk.
        # 4. Map the resulting 2×2 output C block to the full 4×4 output.
        
        # Let's define the A and B block combinations for each Strassen product:
        # Each Xk and Yk is a 2×2 matrix. We represent them as 4-vectors [x00, x01, x10, x11].
        
        strassen_block_combos = [
            # (Xk as 4-vector of coefficients for A blocks [A00, A01, A10, A11],
            #  Yk as 4-vector of coefficients for B blocks [B00, B01, B10, B11])
            ([1, 0, 0, 1], [1, 0, 0, 1]),  # M1: X=A00+A11, Y=B00+B11
            ([0, 0, 1, 1], [1, 0, 0, 0]),  # M2: X=A10+A11, Y=B00
            ([1, 0, 0, 0], [0, 1, 0, -1]), # M3: X=A00, Y=B01-B11
            ([0, 0, 0, 1], [0, 0, 1, -1]), # M4: X=A11, Y=B10-B00
            ([1, 1, 0, 0], [0, 0, 0, 1]),  # M5: X=A00+A01, Y=B11
            ([-1, 0, 1, 0], [1, 1, 0, 0]), # M6: X=A10-A00, Y=B00+B01
            ([0, 1, 0, -1], [0, 0, 1, 1]), # M7: X=A01-A11, Y=B10+B11
        ]
        
        # Output mapping: each Mk contributes to specific C blocks.
        # c00 = M1 + M4 - M5 + M7
        # c01 = M3 + M5
        # c10 = M2 + M4
        # c11 = M1 - M2 + M3 + M6
        #
        # So for each Mk, we have a 4-vector of coefficients for C blocks [C00, C01, C10, C11]:
        strassen_output_coeffs = [
            [1, 0, 0, 1],   # M1: C00 + C11
            [0, 0, 1, -1],  # M2: C10 - C11
            [0, 1, 0, 1],   # M3: C01 + C11
            [1, 0, 1, 0],   # M4: C00 + C10
            [-1, 1, 0, 0],  # M5: -C00 + C01
            [0, 0, 0, 1],   # M6: C11
            [1, 0, 0, 0],   # M7: C00
        ]
        
        U_list = []
        V_list = []
        W_list = []
        
        # Build 4×4×4 terms using Strassen
        for k in range(7):
            a_combo = strassen_block_combos[k][0]  # [coeff for A00, A01, A10, A11]
            b_combo = strassen_block_combos[k][1]  # [coeff for B00, B01, B10, B11]
            c_combo = strassen_output_coeffs[k]    # [coeff for C00, C01, C10, C11]
            
            # For each of the 7 rank-1 terms in the 2×2×2 Strassen decomposition,
            # we need to expand to the 4×4×4 context.
            for (u2, v2, w2) in strassen_2x2_terms:
                # u2 is 4-vector for the 2×2 X matrix: [x00, x01, x10, x11]
                # v2 is 4-vector for the 2×2 Y matrix: [y00, y01, y10, y11]
                # w2 is 4-vector for the 2×2 Z matrix: [z00, z01, z10, z11]
                
                # Build the full U vector (16 entries for 4×4 A matrix).
                # A[i*m+j] where i,j in [0,4).
                # A00 = A[0,0], A[0,1], A[1,0], A[1,1] -> indices 0,1,4,5
                # A01 = A[0,2], A[0,3], A[1,2], A[1,3] -> indices 2,3,6,7
                # A10 = A[2,0], A[2,1], A[3,0], A[3,1] -> indices 8,9,12,13
                # A11 = A[2,2], A[2,3], A[3,2], A[3,3] -> indices 10,11,14,15
                
                # For each element of the 2×2 X matrix, we have a coefficient from u2
                # and a coefficient from a_combo for which A block it comes from.
                # X is a 2×2 matrix where X[row, col] = sum over A blocks of 
                #   a_combo[block] * X_block[row, col]
                # where X_block is the 2×2 submatrix of the A block.
                
                # So for A index (i,j):
                # Determine which A block it belongs to and the position within the block.
                # Block 0 (A00): i in [0,2), j in [0,2) -> local (i, j)
                # Block 1 (A01): i in [0,2), j in [2,4) -> local (i, j-2)
                # Block 2 (A10): i in [2,4), j in [0,2) -> local (i-2, j)
                # Block 3 (A11): i in [2,4), j in [2,4) -> local (i-2, j-2)
                
                U_vec = np.zeros(16, dtype=float)
                for i in range(4):
                    for j in range(4):
                        idx = i * 4 + j
                        if i < 2 and j < 2:
                            block = 0
                            li, lj = i, j
                        elif i < 2 and j >= 2:
                            block = 1
                            li, lj = i, j - 2
                        elif i >= 2 and j < 2:
                            block = 2
                            li, lj = i - 2, j
                        else:
                            block = 3
                            li, lj = i - 2, j - 2
                        # u2[li*2 + lj] is the coefficient for X[li, lj]
                        # a_combo[block] is the coefficient for A_block
                        U_vec[idx] = u2[li * 2 + lj] * a_combo[block]
                
                # Build the full V vector (28 entries for 4×7 B matrix).
                # For the 4×4×4 part, we only use the first 4 columns of B.
                # B is 4×7, so B[j*p + k] for j in [0,4), k in [0,7).
                # For the 4×4 part, k in [0,4).
                # B00 = B[0,0], B[0,1], B[1,0], B[1,1] -> indices 0,1,4,5
                # B01 = B[0,2], B[0,3], B[1,2], B[1,3] -> indices 2,3,6,7
                # B10 = B[2,0], B[2,1], B[3,0], B[3,1] -> indices 8,9,12,13
                # B11 = B[2,2], B[2,3], B[3,2], B[3,3] -> indices 10,11,14,15
                
                V_vec = np.zeros(28, dtype=float)
                for j in range(4):
                    for k in range(4):
                        idx = j * 7 + k
                        if j < 2 and k < 2:
                            block = 0
                            lj, lk = j, k
                        elif j < 2 and k >= 2:
                            block = 1
                            lj, lk = j, k - 2
                        elif j >= 2 and k < 2:
                            block = 2
                            lj, lk = j - 2, k
                        else:
                            block = 3
                            lj, lk = j - 2, k - 2
                        # v2[lj*2 + lk] is the coefficient for Y[lj, lk]
                        # b_combo[block] is the coefficient for B_block
                        V_vec[idx] = v2[lj * 2 + lk] * b_combo[block]
                
                # Build the full W vector (28 entries for 4×7 C matrix, but only 4×4 part).
                # C[i*p + k] where i in [0,4), k in [0,7).
                # For the 4×4 part, k in [0,4).
                # C00 = C[0,0], C[0,1], C[1,0], C[1,1] -> indices 0,1,4,5 (in W: k*n+i)
                # Wait, W has shape (p*n, rank) = (7*4, rank) = (28, rank).
                # W[k*n + i] is the coefficient for C[i,k].
                # So for C[i,k] with i,k in [0,4):
                # C00 (i in [0,2), k in [0,2)): W[k*4 + i]
                # C01 (i in [0,2), k in [2,4)): W[k*4 + i]
                # C10 (i in [2,4), k in [0,2)): W[k*4 + i]
                # C11 (i in [2,4), k in [2,4)): W[k*4 + i]
                
                W_vec = np.zeros(28, dtype=float)
                for i in range(4):
                    for k in range(4):
                        w_idx = k * 4 + i
                        if i < 2 and k < 2:
                            cblock = 0
                            li, lk = i, k
                        elif i < 2 and k >= 2:
                            cblock = 1
                            li, lk = i, k - 2
                        elif i >= 2 and k < 2:
                            cblock = 2
                            li, lk = i - 2, k
                        else:
                            cblock = 3
                            li, lk = i - 2, k - 2
                        # w2[li*2 + lk] is the coefficient for Z[li, lk]
                        # c_combo[cblock] is the coefficient for C_block
                        W_vec[w_idx] = w2[li * 2 + lk] * c_combo[cblock]
                
                U_list.append(U_vec)
                V_list.append(V_vec)
                W_list.append(W_vec)
        
        # Now add schoolbook terms for the remaining 3 columns (columns 4,5,6 of B).
        # A is 4×4, B columns 4,5,6 are 4×3, C columns 4,5,6 are 4×3.
        for i in range(4):
            for j in range(4):
                for k in range(3):
                    global_k = k + 4  # Column index in the full 7-column B
                    U_vec = np.zeros(16, dtype=float)
                    V_vec = np.zeros(28, dtype=float)
                    W_vec = np.zeros(28, dtype=float)
                    U_vec[i * 4 + j] = 1.0
                    V_vec[j * 7 + global_k] = 1.0
                    W_vec[global_k * 4 + i] = 1.0
                    U_list.append(U_vec)
                    V_list.append(V_vec)
                    W_list.append(W_vec)
        
        rank = len(U_list)
        U = np.array(U_list).T
        V = np.array(V_list).T
        W = np.array(W_list).T
        
        # Verify the decomposition
        # Check that U @ V^T gives the correct pattern... actually, let's just check
        # that the tensor reconstruction works for a random input.
        A_test = np.random.randn(4, 4)
        B_test = np.random.randn(4, 7)
        C_expected = A_test @ B_test
        
        # Reconstruct C from the decomposition
        C_reconstructed = np.zeros((4, 7))
        for r in range(rank):
            A_part = np.zeros((4, 4))
            for i in range(4):
                for j in range(4):
                    A_part[i, j] = U[i * 4 + j, r] * A_test[i, j]
            B_part = np.zeros((4, 7))
            for j in range(4):
                for k in range(7):
                    B_part[j, k] = V[j * 7 + k, r] * B_test[j, k]
            C_contrib = np.zeros((4, 7))
            for i in range(4):
                for k in range(7):
                    C_contrib[i, k] = W[k * 4 + i, r]
            # The contribution is A_part * B_part scaled by... 
            # Actually, the tensor decomposition means:
            # C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] * A[i,j] * B[j,k]
            # Wait, that's not right either. Let me think again.
            
            # The standard tensor decomposition for matrix multiplication:
            # C[i,k] = sum_j A[i,j] * B[j,k]
            # = sum_j sum_r (U[i*m+j, r] * A[i,j]) * (V[j*p+k, r] * B[j,k]) * W[k*n+i, r]
            # So we need: sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j] * V[j*p+k, r] * B[j,k])
            # = sum_r W[k*n+i, r] * (A @ (V_r * B))[i,k] where (V_r * B)[j,k] = V[j*p+k, r] * B[j,k]
            
            # Let me compute it properly:
            A_scaled = np.zeros((4, 4))
            for i in range(4):
                for j in range(4):
                    A_scaled[i, j] = A_test[i, j] * U[i * 4 + j, r]
            B_scaled = np.zeros((4, 7))
            for j in range(4):
                for k in range(7):
                    B_scaled[j, k] = B_test[j, k] * V[j * 7 + k, r]
            C_contrib = A_scaled @ B_scaled * W[k * 4 + i, r]  # This is wrong, W varies with i,k
            
            # Correct approach:
            C_contrib = np.zeros((4, 7))
            for ii in range(4):
                for kk in range(7):
                    C_contrib[ii, kk] = W[kk * 4 + ii, r] * (A_scaled @ B_scaled)[ii, kk]
            C_reconstructed += C_contrib
        
        error = np.max(np.abs(C_reconstructed - C_expected))
        
        if error > 1e-10:
            # Fall back to schoolbook
            return schoolbook()
        
        return {"U": U, "V": V, "W": W}
        
    except Exception:
        if time.time() - start_time > DEADLINE:
            return schoolbook()
        return schoolbook()
# EVOLVE-BLOCK-END

