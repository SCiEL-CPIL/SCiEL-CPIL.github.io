import time

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 8
    deadline = time.time() + 140
    
    # We use the decomposition:
    # M_{2,4,8} <= 2 * M_{2,4,4} <= 2 * (2 * M_{2,2,2}) = 4 * 7 = 28
    #
    # Structure:
    # A is 2x4: A = [A_left (2x2) | A_right (2x2)]
    # B is 4x8: B = [B_left (4x4) | B_right (4x4)]
    #   B_left = [[B11 (2x2), B12 (2x2)], [B21 (2x2), B22 (2x2)]]
    #   B_right = [[B31 (2x2), B32 (2x2)], [B41 (2x2), B42 (2x2)]]
    # C is 2x8: C = [C_left (2x4) | C_right (2x4)]
    #   C_left = [C11 (2x2), C12 (2x2)] where C11 = A_left*B11 + A_right*B21, C12 = A_left*B12 + A_right*B22
    #   C_right = [C21 (2x2), C22 (2x2)] where C21 = A_left*B31 + A_right*B41, C22 = A_left*B32 + A_right*B42
    #
    # Each 2x2x2 multiplication uses Strassen with 7 rank-1 terms.
    # Total rank = 4 * 7 = 28.
    
    rank = 28
    nm = n * m  # 8
    mp = m * p  # 32
    pn = p * n  # 16
    
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    
    # Strassen 2x2x2: given A (2x2), B (2x2), compute C = A*B
    # A indices: a00, a01, a10, a11 (row-major)
    # B indices: b00, b01, b10, b11 (row-major)
    # C indices: c00, c01, c10, c11 (row-major)
    #
    # M1 = (a00+a11)*(b00+b11)
    # M2 = (a10+a11)*b00
    # M3 = a00*(b10-b11)
    # M4 = a11*(b00-b10)
    # M5 = (a00+a01)*b11
    # M6 = (a10-a00)*(b00+b01)
    # M7 = (a01-a11)*(b10+b11)
    #
    # c00 = m1+m4-m5+m7
    # c01 = m3+m5
    # c10 = m2+m4
    # c11 = m1-m2+m3+m6
    
    def add_strassen_block(a_base, b_base, c_base, col_offset):
        """
        Add a Strassen 2x2x2 block to the factor matrices.
        
        a_base: (i0, j0) - top-left corner of the 2x2 A submatrix
        b_base: (j0, k0) - top-left corner of the 2x2 B submatrix  
        c_base: (i0, k0) - top-left corner of the 2x2 C submatrix
        
        In our indexing:
        U is indexed by [i*m + j] where i is row of A, j is col of A
        V is indexed by [j*p + k] where j is row of B, k is col of B
        W is indexed by [k*n + i] where k is col of C, i is row of C
        """
        # The 7 Strassen multiplications
        # Each multiplication is: (linear combo of A) * (linear combo of B)
        # which contributes to linear combos of C
        
        # M1 = (a00+a11)*(b00+b11)
        # M2 = (a10+a11)*b00
        # M3 = a00*(b10-b11)
        # M4 = a11*(b00-b10)
        # M5 = (a00+a01)*b11
        # M6 = (a10-a00)*(b00+b01)
        # M7 = (a01-a11)*(b10+b11)
        
        # For each Strassen term, we have:
        # U contributions: coefficients on A entries
        # V contributions: coefficients on B entries  
        # W contributions: coefficients on C entries (from the output formulas)
        
        # Let's define the 7 terms as (U_coeff, V_coeff, W_coeff)
        # where U_coeff is a dict mapping (i,j) -> coeff, etc.
        
        # A entries: (0,0)=a00, (0,1)=a01, (1,0)=a10, (1,1)=a11
        # B entries: (0,0)=b00, (0,1)=b01, (1,0)=b10, (1,1)=b11
        # C entries: (0,0)=c00, (0,1)=c01, (1,0)=c10, (1,1)=c11
        
        # M1: A: {(0,0):1, (1,1):1}, B: {(0,0):1, (1,1):1}, C: {(0,0):1, (1,1):1}
        # M2: A: {(1,0):1, (1,1):1}, B: {(0,0):1}, C: {(1,0):1, (1,1):-1}
        # M3: A: {(0,0):1}, B: {(1,0):1, (1,1):-1}, C: {(0,1):1, (1,1):1}
        # M4: A: {(1,1):1}, B: {(0,0):1, (1,0):-1}, C: {(0,0):1, (1,0):1}
        # M5: A: {(0,0):1, (0,1):1}, B: {(1,1):1}, C: {(0,0):-1, (0,1):1}
        # M6: A: {(1,0):1, (0,0):-1}, B: {(0,0):1, (0,1):1}, C: {(1,1):1}
        # M7: A: {(0,1):1, (1,1):-1}, B: {(1,0):1, (1,1):1}, C: {(0,0):1}
        
        strassen_terms = [
            # (A_coeffs, B_coeffs, C_coeffs)
            ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),      # M1
            ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),                   # M2
            ({(0,0):1}, {(1,0):1, (1,1):-1}, {(0,1):1, (1,1):1}),                   # M3
            ({(1,1):1}, {(0,0):1, (1,0):-1}, {(0,0):1, (1,0):1}),                   # M4
            ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),                   # M5
            ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),                   # M6
            ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),                   # M7
        ]
        
        for term_idx, (a_coeff, b_coeff, c_coeff) in enumerate(strassen_terms):
            col = col_offset + term_idx
            for (ai, aj), val in a_coeff.items():
                U[a_base[0] + ai][col] += val
                # U index: (a_base[0]+ai) * m + (a_base[1]+aj)
                # Wait, I need to be more careful.
                # a_base is (i0, j0), so the A entry at local (ai, aj) is at global (i0+ai, j0+aj)
                # U index = (i0+ai) * m + (j0+aj)
            for (bi, bj), val in b_coeff.items():
                # V index: (b_base[0]+bi) * p + (b_base[1]+bj)
                V[b_base[0] + bi][col] += val
            for (ci, cj), val in c_coeff.items():
                # W index: (c_base[1]+cj) * n + (c_base[0]+ci)
                W[c_base[1] + cj][col] += val
    
    # Wait, I made an error in the indexing. Let me redo this more carefully.
    
    # U has shape (n*m, rank) = (8, 28)
    # U[i*m + j, col] corresponds to A[i, j]
    # 
    # V has shape (m*p, rank) = (32, 28)
    # V[j*p + k, col] corresponds to B[j, k]
    #
    # W has shape (p*n, rank) = (16, 28)
    # W[k*n + i, col] corresponds to C[i, k] (with the trace convention W[k*n+i, r])
    
    # So for a Strassen block where:
    # A submatrix is at rows [i0, i0+1], cols [j0, j0+1]
    # B submatrix is at rows [j0', j0'+1], cols [k0, k0+1]
    # C submatrix is at rows [i0, i0+1], cols [k0, k0+1]
    
    # For A entry at local (ai, aj): global A[i0+ai, j0+aj], U index = (i0+ai)*m + (j0+aj)
    # For B entry at local (bi, bj): global B[j0'+bi, k0+bj], V index = (j0'+bi)*p + (k0+bj)
    # For C entry at local (ci, cj): global C[i0+ci, k0+cj], W index = (k0+cj)*n + (i0+ci)
    
    def add_strassen_block(i0, j0, j0_b, k0, col_offset):
        """
        Add a Strassen 2x2x2 block.
        i0: row offset for A and C
        j0: col offset for A
        j0_b: row offset for B
        k0: col offset for B and C
        """
        strassen_terms = [
            ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
            ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
            ({(0,0):1}, {(1,0):1, (1,1):-1}, {(0,1):1, (1,1):1}),
            ({(1,1):1}, {(0,0):1, (1,0):-1}, {(0,0):1, (1,0):1}),
            ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
            ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
            ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
        ]
        
        for term_idx, (a_coeff, b_coeff, c_coeff) in enumerate(strassen_terms):
            col = col_offset + term_idx
            for (ai, aj), val in a_coeff.items():
                U[(i0+ai)*m + (j0+aj)][col] += val
            for (bi, bj), val in b_coeff.items():
                V[(j0_b+bi)*p + (k0+bj)][col] += val
            for (ci, cj), val in c_coeff.items():
                W[(k0+cj)*n + (i0+ci)][col] += val
    
    # Now, the 4 Strassen blocks:
    # Block 0: C11 = A_left * B11 + A_right * B21
    #   This is NOT a single 2x2x2 multiplication! It's a SUM of two 2x2x2 multiplications.
    #   
    # Wait, I need to reconsider. Strassen computes C = A*B where both are 2x2.
    # But here C11 = A_left*B11 + A_right*B21, which is a sum of two products.
    #
    # So I need to use Strassen for each product separately, then add the results.
    # That means I need 2 Strassen blocks for C11, 2 for C12, 2 for C21, 2 for C22.
    # Total: 8 Strassen blocks * 7 = 56 rank. That's worse than schoolbook!
    
    # Hmm, that's not right. Let me reconsider the decomposition.
    
    # Actually, the issue is that M_{2,4,4} is NOT simply 2 * M_{2,2,2}.
    # The correct decomposition is:
    # C = A_left * B_top + A_right * B_bottom
    # where A_left, A_right are 2x2, B_top, B_bottom are 2x2 (top and bottom halves of B).
    # C is 2x2.
    #
    # So M_{2,4,4} = M_{2,2,2} + M_{2,2,2} in the sense that we need to compute two 2x2x2 products and add them.
    # But in tensor decomposition terms, this means the rank is at most 2 * rank(M_{2,2,2}) = 2 * 7 = 14.
    #
    # For M_{2,4,8}:
    # C = A * B where B is 4x8.
    # Split B into B_left (4x4) and B_right (4x4).
    # C = A*B_left + A*B_right, each is 2x4.
    # Each 2x4 product has rank <= 14.
    # Total rank <= 28.
    
    # But wait, for each 2x4 product (say A*B_left where B_left is 4x4):
    # C_block = A_left*B11 + A_right*B21 (for the first 2 columns) 
    #         + A_left*B12 + A_right*B22 (for the next 2 columns)
    # No wait, that's not right either.
    
    # Let me be very explicit.
    # A is 2x4: A = [[a00, a01, a02, a03], [a10, a11, a12, a13]]
    # B is 4x4: B = [[b00, b01, b02, b03], [b10, b11, b12, b13], [b20, b21, b22, b23], [b30, b31, b32, b33]]
    # C is 2x4: C = [[c00, c01, c02, c03], [c10, c11, c12, c13]]
    #
    # C[0,0] = a00*b00 + a01*b10 + a02*b20 + a03*b30
    # C[0,1] = a00*b01 + a01*b11 + a02*b21 + a03*b31
    # C[0,2] = a00*b02 + a01*b12 + a02*b22 + a03*b32
    # C[0,3] = a00*b03 + a01*b13 + a02*b23 + a03*b33
    # ... etc.
    #
    # Split A into A_left = [[a00, a01], [a10, a11]] and A_right = [[a02, a03], [a12, a13]]
    # Split B into B_top = [[b00, b01, b02, b03], [b10, b11, b12, b13]] (2x4)
    #              B_bot = [[b20, b21, b22, b23], [b30, b31, b32, b33]] (2x4)
    #
    # C = A_left * B_top + A_right * B_bot
    #
    # Now A_left * B_top is 2x2 * 2x4 = 2x4.
    # A_right * B_bot is 2x2 * 2x4 = 2x4.
    #
    # For A_left * B_top:
    # Split B_top into B_top_left (2x2) and B_top_right (2x2).
    # A_left * B_top = A_left * [B_top_left | B_top_right]
    #                  = [A_left*B_top_left | A_left*B_top_right]
    # Each is 2x2 * 2x2 = 2x2.
    #
    # So A_left * B_top requires 2 Strassen blocks (one for each 2x2 output).
    # Similarly, A_right * B_bot requires 2 Strassen blocks.
    # Total for the 2x4x4 product: 4 Strassen blocks = 28 rank.
    #
    # That's worse than I hoped. Let me reconsider.
    
    # Actually, I think the key insight is different. Let me use the fact that:
    # M_{2,4,4} can be done with rank 8 (not 14).
    #
    # Here's why: The 2x4 * 4x4 product can be viewed as:
    # C[i,k] = sum_j A[i,j] * B[j,k] for i=0,1 and k=0,1,2,3
    #
    # This is just 8 independent scalar multiplications if we do schoolbook.
    # But we can share computations.
    #
    # Actually, I recall that M_{2,n,n} has rank at most 2n for any n (using a simple sharing argument).
    # For n=4, that's rank 8. For n=8, that's rank 16.
    #
    # Let me verify: For M_{2,4,4}, rank <= 8?
    # C[0,k] = a00*B[0,k] + a01*B[1,k] + a02*B[2,k] + a03*B[3,k]
    # C[1,k] = a10*B[0,k] + a11*B[1,k] + a12*B[2,k] + a13*B[3,k]
    #
    # We can compute:
    # t1 = a00*B[0,k] for all k (but B[0,k] varies with k)
    # Hmm, this doesn't directly give sharing.
    
    # Let me try a different approach. I'll just implement the rank-28 solution using Strassen
    # on the 2x2x2 subproblems, even if it's not optimal.
    
    # Reset and rebuild
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    
    # For M_{2,4,8}:
    # A is 2x4, B is 4x8, C is 2x8
    #
    # Split into 4 blocks of 2x2x2:
    # Block (r, s) where r=0,1 (row of A/C) and s=0,1,2,3 (group of 2 cols of B/C)
    # 
    # Actually, let me just use the fact that M_{2,4,8} <= 4 * M_{2,2,2} = 4*7 = 28
    # by splitting the 4 rows of B into 4 groups of 1... no that doesn't work.
    
    # OK let me just do this carefully. The cleanest decomposition:
    # 
    # M_{2,4,8}: A(2x4) * B(4x8) = C(2x8)
    # 
    # Write A = [A1 A2] where A1, A2 are 2x2
    # Write B = [[B11 B12 B13 B14], [B21 B22 B23 B24]] where each Bij is 2x2
    # 
    # C = [C1 C2 C3 C4] where Ci is 2x2
    # C1 = A1*B11 + A2*B21
    # C2 = A1*B12 + A2*B22
    # C3 = A1*B13 + A2*B23
    # C4 = A1*B14 + A2*B24
    #
    # Each Ci requires computing A1*Bij + A2*B2j, which is a sum of two 2x2x2 products.
    # Using Strassen for each: 7 + 7 = 14 rank per Ci.
    # Total: 4 * 14 = 56. Worse than schoolbook!
    
    # So Strassen on 2x2x2 subblocks doesn't help here because the intermediate sums
    # don't allow sharing between the two products being added.
    
    # Let me go back to schoolbook but check if there's any simple sharing.
    # Actually, the current score is 1.0 which corresponds to rank 64 (schoolbook).
    # Let me just verify the schoolbook is correct and return it.
    
    # Reset to schoolbook
    rank = n * m * p  # 64
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
