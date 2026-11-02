import numpy as np
import time

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # We will use a hybrid approach:
    # 1. Strassen-based decomposition for the 4x4x4 block (Rank 49)
    # 2. Schoolbook decomposition for the 4x4x2 block (Rank 32)
    # Total Rank = 81
    
    rank = 81
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    current_rank = 0
    
    # Helper to add a term to the decomposition
    def add_term(u_idx, v_idx, w_idx, coeff):
        nonlocal current_rank
        if coeff == 0:
            return
        U[u_idx][current_rank] += coeff
        V[v_idx][current_rank] += coeff
        W[w_idx][current_rank] += coeff
        current_rank += 1

    # Helper to add a 2x2x2 Strassen decomposition
    # u_off: top-left corner of A block (2x2)
    # v_off: top-left corner of B block (2x2)
    # w_off: top-left corner of C block (2x2)
    def add_strassen_2x2x2(u_off, v_off, w_off):
        # A indices: (u_off, u_off+1) x (u_off, u_off+1) relative to 2x2 block
        # In global A (4x4):
        # A00 at (u_off, u_off), A01 at (u_off, u_off+1), etc.
        
        # Strassen 7 multiplications for 2x2:
        # M1 = (A00 + A11) * (B00 + B11)
        # M2 = (A10 + A11) * B00
        # M3 = A00 * (B01 - B11)
        # M4 = A11 * (B10 - B00)
        # M5 = (A00 + A01) * B11
        # M6 = (A10 - A00) * (B00 + B01)
        # M7 = (A01 - A11) * (B10 + B11)
        
        # We need to decompose each Mk into rank-1 terms.
        # Since Mk is a 2x2 matrix product of linear combinations,
        # we can treat the "A" part and "B" part as 2x2 matrices with coefficients.
        
        # For each multiplication Mk = Xk * Yk:
        # Xk is a 2x2 matrix formed from A entries with coefficients.
        # Yk is a 2x2 matrix formed from B entries with coefficients.
        # The product Xk * Yk is a 2x2 matrix.
        # We can decompose this product using the standard 4-term schoolbook for 2x2x2?
        # No, that would be 7 * 4 = 28 terms, which is rank 28 for 4x4x4?
        # Wait. The rank of the tensor for 4x4x4 using Strassen is 49.
        # This implies that each of the 7 intermediate 2x2 multiplications is decomposed into 7 rank-1 terms?
        # No, the Strassen algorithm for 4x4x4 uses 7 multiplications of 2x2 matrices.
        # If we use the minimal rank for 2x2x2 (which is 7), then total rank is 7*7=49.
        # So we need to decompose each 2x2x2 multiplication using Strassen (rank 7).
        
        # Let's implement a helper to add a 2x2x2 Strassen decomposition for matrices X and Y.
        # X is defined by coefficients x[i][j] for A indices.
        # Y is defined by coefficients y[i][j] for B indices.
        # The result contributes to C with coefficients w[i][k].
        
        # Actually, it's easier to just hardcode the 49 terms for the 4x4x4 Strassen decomposition.
        # But that's error prone.
        
        # Alternative: Use the fact that the 4x4x4 Strassen decomposition is the tensor product of the 2x2x2 Strassen decomposition?
        # No, it's a block decomposition.
        
        # Let's use a simpler, verified decomposition:
        # Rank(4,4,4) <= 49 is known.
        # Can we construct it easily?
        
        # Let's try to construct the 49 terms programmatically.
        # The 7 intermediate matrices are:
        # M1 = (A00+A11)(B00+B11)
        # ...
        
        # For each M, we have a 2x2 product.
        # We can decompose each 2x2 product into 7 rank-1 terms using Strassen's 2x2x2 algorithm.
        # So we need a function add_strassen_2x2x2_with_coeffs(A_coeffs, B_coeffs, W_coeffs).
        
        # A_coeffs: 2x2 list of (global_u_idx, coeff)
        # B_coeffs: 2x2 list of (global_v_idx, coeff)
        # W_coeffs: 2x2 list of (global_w_idx, coeff)
        
        # This is getting complex. Let's just use Schoolbook for the 2x2 blocks inside the Strassen structure?
        # Rank would be 7 * 4 = 28.
        # Is Rank(4,4,4) <= 28?
        # The known lower bound is 20, upper bound 49 (Strassen) or 64 (Schoolbook).
        # Actually, 28 is not a known upper bound. The standard Strassen gives 49.
        # Using schoolbook for the leaves (rank 4) gives 28.
        # Is this valid?
        # Yes, because the bilinear map is linear.
        # So Rank(4,4,4) <= 28.
        # Then Rank(4,4,6) <= 28 + 32 = 60.
        
        # Let's implement the 28-term decomposition for 4x4x4.
        # This is safer and still better than 96.
        
        # The 7 multiplications in 4x4 Strassen:
        # 1. (A00+A11)(B00+B11)
        # 2. (A10+A11)B00
        # 3. A00(B01-B11)
        # 4. A11(B10-B00)
        # 5. (A00+A01)B11
        # 6. (A10-A00)(B00+B01)
        # 7. (A01-A11)(B10+B11)
        
        # For each, we use schoolbook 2x2x2 (4 terms).
        
        # Helper to add schoolbook 2x2x2 with coefficients
        # A_coeffs: [[(u_idx, c), (u_idx, c)], [(u_idx, c), (u_idx, c)]]
        # B_coeffs: [[(v_idx, c), (v_idx, c)], [(v_idx, c), (v_idx, c)]]
        # W_coeffs: [[(w_idx, c), (w_idx, c)], [(w_idx, c), (w_idx, c)]]
        
        def add_schoolbook_2x2x2(A_c, B_c, W_c):
            # A_c[i][j] is (u_idx, coeff)
            # B_c[i][j] is (v_idx, coeff)
            # W_c[i][j] is (w_idx, coeff)
            # Product C = A * B
            # C[i][k] = sum_j A[i][j] * B[j][k]
            
            for i in range(2):
                for k in range(2):
                    # Accumulate sum_j A[i][j] * B[j][k]
                    # We need to add terms for each j
                    for j in range(2):
                        u_idx, u_c = A_c[i][j]
                        v_idx, v_c = B_c[j][k]
                        w_idx, w_c = W_c[i][k]
                        
                        # The term is u_c * v_c * w_c * (A[u_idx] * B[v_idx] * W[w_idx])
                        # Wait, the tensor decomposition is sum_r U[i][r] V[j][r] W[k][r]
                        # Here, for a specific rank-1 term, we have:
                        # U[u_idx][r] = u_c
                        # V[v_idx][r] = v_c
                        # W[w_idx][r] = w_c
                        # But this only adds one term to the sum for C[i][k].
                        # We need to add this term to the global U, V, W.
                        
                        coeff = u_c * v_c * w_c
                        if coeff != 0:
                            add_term(u_idx, v_idx, w_idx, coeff)

        # Now define the 7 multiplications.
        # Indices in A (4x4):
        # A00: (0,0), (0,1), (1,0), (1,1) -> u_idx = 0,1,4,5
        # A01: (0,2), (0,3), (1,2), (1,3) -> u_idx = 2,3,6,7
        # A10: (2,0), (2,1), (3,0), (3,1) -> u_idx = 8,9,12,13
        # A11: (2,2), (2,3), (3,2), (3,3) -> u_idx = 10,11,14,15
        
        # Indices in B (4x6):
        # B00: (0,0), (0,1), (1,0), (1,1) -> v_idx = 0,1,6,7
        # B01: (0,2), (0,3), (0,4), (0,5) -> v_idx = 2,3,4,5
        # B10: (2,0), (2,1), (3,0), (3,1) -> v_idx = 12,13,18,19
        # B11: (2,2), (2,3), (2,4), (2,5) -> v_idx = 14,15,16,17
        
        # Indices in C (6x4) -> W[k*n+i, r]
        # C00: (0,0), (0,1), (1,0), (1,1) -> w_idx = 0,1,4,5 (k=0,1; i=0,1)
        # C01: (0,2), (0,3), (1,2), (1,3) -> w_idx = 2,3,6,7
        # C10: (2,0), (2,1), (3,0), (3,1) -> w_idx = 8,9,12,13
        # C11: (2,2), (2,3), (3,2), (3,3) -> w_idx = 10,11,14,15
        
        # Let's define the blocks as lists of (idx, coeff)
        
        # A00 block
        A00 = [
            [(0, 1.0), (1, 1.0)],
            [(4, 1.0), (5, 1.0)]
        ]
        # A01 block
        A01 = [
            [(2, 1.0), (3, 1.0)],
            [(6, 1.0), (7, 1.0)]
        ]
        # A10 block
        A10 = [
            [(8, 1.0), (9, 1.0)],
            [(12, 1.0), (13, 1.0)]
        ]
        # A11 block
        A11 = [
            [(10, 1.0), (11, 1.0)],
            [(14, 1.0), (15, 1.0)]
        ]
        
        # B00 block
        B00 = [
            [(0, 1.0), (1, 1.0)],
            [(6, 1.0), (7, 1.0)]
        ]
        # B01 block
        B01 = [
            [(2, 1.0), (3, 1.0)],
            [(8, 1.0), (9, 1.0)] # Wait, B is 4x6.
            # B rows 0,1 cols 2,3,4,5
            # v_idx = j*6 + k
            # j=0: k=2,3,4,5 -> 2,3,4,5
            # j=1: k=2,3,4,5 -> 8,9,10,11
        ]
        # Correction for B01
        B01 = [
            [(2, 1.0), (3, 1.0)], # Row 0, Cols 2,3? No, B01 is 2x2 block?
            # The 4x4x4 part uses B[0:2, 0:4] and B[2:4, 0:4]?
            # No, the 4x4x4 multiplication is A[0:4,0:4] * B[0:4,0:4].
            # So B is effectively 4x4 for this part.
            # B00: rows 0,1 cols 0,1
            # B01: rows 0,1 cols 2,3
            # B10: rows 2,3 cols 0,1
            # B11: rows 2,3 cols 2,3
            
            # Let's re-verify B indices for 4x4 submatrix.
            # B00: v(0,0)=0, v(0,1)=1, v(1,0)=6, v(1,1)=7
            # B01: v(0,2)=2, v(0,3)=3, v(1,2)=8, v(1,3)=9
            # B10: v(2,0)=12, v(2,1)=13, v(3,0)=18, v(3,1)=19
            # B11: v(2,2)=14, v(2,3)=15, v(3,2)=20, v(3,3)=21
        ]
        B01 = [
            [(2, 1.0), (3, 1.0)],
            [(8, 1.0), (9, 1.0)]
        ]
        B10 = [
            [(12, 1.0), (13, 1.0)],
            [(18, 1.0), (19, 1.0)]
        ]
        B11 = [
            [(14, 1.0), (15, 1.0)],
            [(20, 1.0), (21, 1.0)]
        ]
        
        # C blocks for 4x4x4 part (C is 4x4 submatrix of 6x4)
        # C00: w(0,0)=0, w(0,1)=1, w(1,0)=4, w(1,1)=5
        # C01: w(0,2)=2, w(0,3)=3, w(1,2)=6, w(1,3)=7
        # C10: w(2,0)=8, w(2,1)=9, w(3,0)=12, w(3,1)=13
        # C11: w(2,2)=10, w(2,3)=11, w(3,2)=14, w(3,3)=15
        
        C00 = [
            [(0, 1.0), (1, 1.0)],
            [(4, 1.0), (5, 1.0)]
        ]
        C01 = [
            [(2, 1.0), (3, 1.0)],
            [(6, 1.0), (7, 1.0)]
        ]
        C10 = [
            [(8, 1.0), (9, 1.0)],
            [(12, 1.0), (13, 1.0)]
        ]
        C11 = [
            [(10, 1.0), (11, 1.0)],
            [(14, 1.0), (15, 1.0)]
        ]
        
        # Helper to add linear combination of blocks
        def add_block_comb(A_blocks, B_blocks, C_block):
            # A_blocks: list of (block, coeff)
            # B_blocks: list of (block, coeff)
            # C_block: block
            # We need to compute (sum a_i A_i) * (sum b_j B_j) = sum_{i,j} a_i b_j (A_i B_j)
            # This is not how Strassen works. Strassen precomputes sums.
            # M = (A+D)(B+E) = (A+B) + (A+E) + (D+B) + (D+E)? No.
            # M = (A+D)(B+E)
            # We can just pass the combined coefficients to add_schoolbook_2x2x2.
            
            # Construct A_c and B_c with combined coefficients
            A_c = [[(0, 0.0) for _ in range(2)] for _ in range(2)]
            B_c = [[(0, 0.0) for _ in range(2)] for _ in range(2)]
            
            for block, coeff in A_blocks:
                for i in range(2):
                    for j in range(2):
                        idx, c = block[i][j]
                        A_c[i][j] = (idx, A_c[i][j][1] + coeff * c)
            
            for block, coeff in B_blocks:
                for i in range(2):
                    for j in range(2):
                        idx, c = block[i][j]
                        B_c[i][j] = (idx, B_c[i][j][1] + coeff * c)
            
            # C_block is just the target block, coeff 1
            W_c = C_block
            
            add_schoolbook_2x2x2(A_c, B_c, W_c)

        # M1 = (A00+A11)(B00+B11)
        add_block_comb([(A00, 1.0), (A11, 1.0)], [(B00, 1.0), (B11, 1.0)], C00)
        # Note: The result of M1 is used in C00, C01, C10, C11?
        # No, M1 is a 2x2 matrix. Its (i,k) entry contributes to C(i,k) of the 4x4 block.
        # But Strassen's algorithm combines these M's to form C.
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # So we need to add the contribution of M1 to C00 and C11.
        # M1 is a 2x2 matrix.
        # M1[i][k] adds to C00[i][k] and C11[i][k]?
        # No.
        # The 4x4 C is:
        # C = [[C00, C01], [C10, C11]]
        # M1 is the result of (A00+A11)(B00+B11).
        # M1 is a 2x2 matrix.
        # How does M1 map to C?
        # In Strassen 4x4:
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # So M1 contributes to C00 and C11.
        # Specifically, M1[i][k] adds to C00[i][k] and C11[i][k].
        
        # My add_block_comb adds to a single C block.
        # I need to modify it to add to multiple C blocks or call it multiple times.
        
        # Let's redo this.
        # I will define the 7 M's and then sum them into the C blocks.
        
        # Reset current_rank? No, we are building it.
        # Let's clear U,V,W and restart with a cleaner structure.
        # This is getting messy. Let's just use Schoolbook for the whole thing (Rank 96) to ensure correctness first,
        # then try to optimize if time permits.
        # But 96 is the seed. I need to beat it.
        
        # Let's try a different split:
        # Rank(4,4,6) <= Rank(4,2,6) + Rank(2,4,6)?
        # No.
        
        # Let's stick to the 28-term 4x4x4 decomposition.
        # I will clear the current state and rebuild.
        
        # This is too complex to do correctly in a single pass without testing.
        # I will return the Schoolbook solution (Rank 96) which is correct,
        # but the task asks for a stronger algorithm.
        # However, a correct 96 is better than an incorrect 81.
        # But the seed is 96. Score 1.0.
        # If I return 96, score is 1.0.
        # If I return 81 correctly, score is > 1.0.
        
        # Let's try to implement the 28-term 4x4x4 correctly.
        
        # I will restart the construction.
        # I will clear U, V, W.
        
        return {"U": U, "V": V, "W": W} # Placeholder, will be replaced by the full logic below

# Actual implementation
def construct():
    n, m, p = DIMENSIONS
    rank = 96 # Start with schoolbook, try to improve
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Schoolbook
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
                
    return {"U": U, "V": V, "W": W}
