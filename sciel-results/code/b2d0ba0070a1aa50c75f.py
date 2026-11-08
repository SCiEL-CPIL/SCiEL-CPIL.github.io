import time
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 130.0  # Safety margin before 150s limit
    
    n, m, p = 4, 5, 5  # A is n×m, B is m×p, C is n×p
    
    # Strategy: Pad A (4×5) to A_pad (5×5) by adding a zero row at the bottom.
    # Then C_pad = A_pad * B is 5×5, and C = C_pad[:4, :] (first 4 rows).
    # Use the known rank-50 decomposition for 5×5×5 matrix multiplication.
    
    # Build the rank-50 decomposition for 5×5×5 using Kronecker products of Strassen 2×2
    # Actually, let's use a simpler approach: decompose 5×5×5 directly.
    # The standard approach: 5×5 can be seen as blocks, but let's use the known result.
    
    # For 5×5×5, one known construction uses rank 50 via the following:
    # Split into 2×2 blocks with an extra row/col, but a cleaner way:
    # Use the fact that 5×5 multiplication can be done with rank 50 by
    # padding to 8×8 (rank 37 via Strassen) doesn't help directly.
    
    # Actually, let me construct it more carefully.
    # 5×5×5: We can view this as:
    # A is 5×5, B is 5×5, C is 5×5
    # Split A = [A1 A2] where A1 is 5×3, A2 is 5×2
    # Split B = [B1; B2] where B1 is 3×5, B2 is 2×5
    # C = A1*B1 + A2*B2
    # But this doesn't directly give a clean rank bound.
    
    # Let me use a different approach: known that r(5,5,5) ≤ 50
    # Construction: use the 2×2 Strassen (rank 7) and pad.
    # 5×5 = 4×4 + extra row/col
    # 4×4×4 uses Strassen with rank 7 (for 2×2 blocks of 2×2)
    # Actually 4×4×4 can be done with rank 7 by applying Strassen to 2×2 blocks where each block is 2×2.
    # Wait, that's not right. 4×4×4: split into 2×2 blocks of size 2×2.
    # Each "scalar" is a 2×2 matrix, so we need 7 multiplications of 2×2×2 matrices.
    # Each 2×2×2 has rank 7, so total rank = 7*7 = 49. That matches the current approach.
    
    # For 5×5×5: 
    # Approach 1: Pad to 8×8, use rank 37 (Strassen 3-level), but 8×8 is too big.
    # Approach 2: Direct construction.
    
    # Let me think differently. The current approach gives 85.
    # The padding approach: embed 4×5×5 into 5×5×5.
    # If I can get 5×5×5 at rank 50, then I'm done (just project).
    
    # Known: r(5,5,5) ≤ 50. One construction:
    # View 5×5 as a 2×2 block matrix with blocks of sizes:
    # A = [[A11, A12], [A21, A22]] where A11 is 2×2, A12 is 2×3, A21 is 3×2, A22 is 3×3
    # This is getting complex. Let me use a simpler known construction.
    
    # Alternative: Use the fact that 5×5×5 can be decomposed using
    # the 2×2 Strassen on a 4×4 subblock plus handling the 5th row/col.
    # 5×5×5 = 4×4×4 (rank 49) + corrections for the 5th row and column.
    # This might not beat 50.
    
    # Let me try a direct tensor decomposition approach using numpy.
    # Actually, for the purpose of this task, let me construct the 5×5×5 
    # decomposition at rank 50 using a known method.
    
    # Known construction for r(5,5,5) ≤ 50:
    # Use the following block structure. Split indices into {0,1,2} and {3,4}.
    # This is a 3+2 split.
    # A: 5×5, B: 5×5, C: 5×5
    # A = [A11 A12; A21 A22] where A11 is 3×3, A12 is 3×2, A21 is 2×3, A22 is 2×2
    # B = [B11 B12; B21 B22] where B11 is 3×3, B12 is 3×2, B21 is 2×3, B22 is 2×2
    # C = A*B, so:
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    
    # This is 4 sub-problems. Not obviously better.
    
    # Let me just use a straightforward construction that I know works.
    # For 5×5×5, one can achieve rank 50 using the following:
    # 
    # Actually, let me reconsider. The simplest correct approach:
    # Use the standard 5×5×5 decomposition at rank 50.
    # 
    # Here's a concrete construction:
    # We can write 5×5×5 as a sum of 50 rank-1 tensors.
    # 
    # Let me use a different strategy: since we just need to multiply 4×5 by 5×5,
    # and the output is 4×5, let me think about what's minimal.
    #
    # The bilinear complexity of (4,5,5) multiplication:
    # Lower bound: max(4*5, 5*5, 4*5) = 25 (trivially)
    # Upper bound: 4*5*5 = 100 (naive)
    # Current: 85
    # 
    # Can we do better? 
    # One approach: C = A*B where A is 4×5, B is 5×5.
    # C[i][j] = sum_k A[i][k]*B[k][j]
    # 
    # Let me try: pad A to 5×5 (add zero row), compute 5×5×5 product, take first 4 rows.
    # If 5×5×5 has rank R, then 4×5×5 has rank ≤ R.
    # 
    # Known: r(5,5,5) ≤ 50. Let me construct this.
    
    # Construction of 5×5×5 at rank 50:
    # Use the following: decompose 5×5×5 using a 2×2 Strassen-like approach
    # on a partition of the indices.
    #
    # Partition {0,1,2,3,4} into {0,1,2} and {3,4}
    # A = [[A11, A12], [A21, A22]] with A11:3×3, A12:3×2, A21:2×3, A22:2×2
    # B = [[B11, B12], [B21, B22]] with B11:3×3, B12:3×2, B21:2×3, B22:2×2
    #
    # C11 = A11*B11 + A12*B21  (3×3)
    # C12 = A11*B12 + A12*B22  (3×2)
    # C21 = A21*B11 + A22*B21  (2×3)
    # C22 = A21*B12 + A22*B22  (2×2)
    #
    # This doesn't obviously decompose. Let me use a different known construction.
    #
    # Actually, I recall that r(5,5,5) ≤ 50 can be achieved by:
    # r(5,5,5) ≤ r(4,4,4) + 5*5 - 4*4 + ... no, that's not right.
    #
    # Let me just use the naive 5×5×5 at rank 25... no, that's the number of terms, not rank.
    # Naive is rank 25 (one term per (i,j,k) triple... no, rank 25 means 25 rank-1 tensors).
    # Wait, naive matrix multiplication of 5×5×5 has rank 25? No!
    # The tensor T[i,j,k] = delta_{i,k} for matrix multiplication has rank 25?
    # No, the rank of the matrix multiplication tensor <n,m,p> is the bilinear complexity.
    # For <5,5,5>, naive is 25 multiplications, so rank ≤ 25? No, that's wrong.
    # Naive: C[i][j] = sum_k A[i][k]*B[k][j]. This uses 25 products, each is a rank-1 tensor
    # in the bilinear sense? No. Each product A[i][k]*B[k][j] contributes to C[i][j].
    # The tensor is T = sum_{i,j,k} e_i ⊗ e_k ⊗ e_j * delta... 
    # Actually the matrix multiplication tensor <m,n,p> has T[i,j,k] = 1 if i=k (for the appropriate indexing).
    # The rank of <5,5,5> is at most 25 (naive) and it's known to be at most 50... 
    # Wait, I'm confusing myself. The naive algorithm uses 25 scalar multiplications,
    # which corresponds to a tensor decomposition of rank 25. So r(5,5,5) ≤ 25?
    # No! The naive algorithm for 5×5×5 uses 5*5*5 = 125 scalar multiplications,
    # but the bilinear complexity (tensor rank) is different.
    # 
    # Actually, the matrix multiplication tensor <n,m,p> is defined as:
    # T = sum_{i=1}^n sum_{j=1}^m sum_{k=1}^p e_i ⊗ f_k ⊗ g_j
    # where the output is sum_k A[i,k]*B[k,j] for each (i,j).
    # This tensor has rank at most n*m*p (trivially) but the bilinear complexity
    # (minimum number of scalar multiplications) is what we're after.
    # For <5,5,5>, the naive bilinear complexity is 25 (since C[i,j] = sum_k A[i,k]B[k,j],
    # and each product A[i,k]*B[k,j] is one multiplication, total 5*5*5=125... no).
    # Wait: for each output element C[i,j], we need sum over k of A[i,k]*B[k,j].
    # That's 5 multiplications per output, 25 outputs, total 125 multiplications naively.
    # But we can share: the bilinear complexity is the tensor rank.
    # r(5,5,5) is known to be between 25 and 50 or so.
    #
    # I think I'm overcomplicating this. Let me just verify: the current program
    # achieves rank 85 for 4×5×5. The naive would be 4*5*5 = 100. So 85 < 100, good.
    #
    # For the padding approach to work, I need r(5,5,5) ≤ 50, which would give
    # rank 50 for 4×5×5, a big improvement over 85.
    #
    # Let me construct r(5,5,5) ≤ 50 explicitly.
    #
    # One known construction: use the 2×2 Strassen (rank 7) and embed.
    # 5 = 4 + 1. 
    # 4×4×4 has rank 49 (as used in current program).
    # 5×5×5 = 4×4×4 + corrections.
    # This likely gives rank > 50.
    #
    # Alternative known construction for r(5,5,5) ≤ 50:
    # I'll use a direct construction. The tensor <5,5,5> can be decomposed as:
    #
    # Actually, let me look at this from a different angle.
    # The bilinear complexity of <4,5,5> is what we want.
    # 
    # Known results: r(4,5,5) ≤ 50? Let me check if there's a direct construction.
    #
    # For now, let me try a simpler improvement: optimize the current block decomposition.
    # Current: 49 + 16 + 16 + 4 = 85
    # 
    # Can I do better on the sub-problems?
    # Part 1: 4×4×4 at rank 49 (Strassen). This is optimal for 4×4×4? 
    # r(4,4,4) = 49 is the best known. So that's fixed.
    # Part 2: 4×1×4 at rank 16. This is just sum_k A[i,k]*B[k,j] for i in 0..3, j in 0..3.
    # This is a 4×4 matrix product where one factor is a column vector.
    # C[i,j] = sum_k A[i,k]*B[k,j] where A is 4×4, B is 4×4... 
    # Wait, Part 2 is A_R (4×1) times B_BL (4×4) -> 4×4.
    # C[i,j] = A_R[i] * B_BL[i,j]... no.
    # A_R is 4×1 (column vector), B_BL is 4×4.
    # C = A_R * B_BL is 4×4? No: (4×1)*(4×4) is not defined!
    # 
    # I think there's an error in the current program. Let me re-examine.
    # A is 4×5, B is 5×5.
    # A = [A_L | A_R] where A_L is 4×4, A_R is 4×1.
    # B = [B_TL B_TR; B_BL B_BR] where B_TL is 4×4, B_TR is 4×1, B_BL is 1×4, B_BR is 1×1.
    # C = A*B = [A_L B_L] * [B_T; B_B] = A_L*B_T + A_R*B_B
    # where B_T = [B_TL B_TR] (4×5) and B_B = [B_BL B_BR] (1×5).
    # C_L (4×4) = A_L*B_TL + A_R*B_BL
    # C_R (4×1) = A_L*B_TR + A_R*B_BR
    #
    # A_R is 4×1, B_BL is 1×4. So A_R*B_BL is (4×1)*(1×4) = 4×4. Correct!
    # Part 2: 4×1×4 means A is 4×1, B is 1×4, C is 4×4.
    # C[i,j] = A_R[i]*B_BL[j]. This is an outer product! Rank 1!
    # Wait, that's not right. C[i,j] = sum_k A_R[i,k]*B_BL[k,j].
    # A_R is 4×1, so A_R[i,k] is only defined for k=0. So C[i,j] = A_R[i]*B_BL[j].
    # This is indeed an outer product, rank 1!
    #
    # But the current program uses rank 16 for this. That's way too high!
    # Similarly, Part 3: A_L (4×4) * B_TR (4×1) -> 4×1.
    # C[i] = sum_j A_L[i,j]*B_TR[j]. This is a standard matrix-vector product.
    # The tensor for this is <4,4,1>. Rank? 
    # C[i] = sum_j A[i,j]*B[j]. This has rank 4 (one per output i, since B[j] is shared).
    # Actually, the bilinear complexity: we need 4 outputs, each is a dot product of length 4.
    # Can we share computations? The tensor is sum_i sum_j e_i ⊗ e_j ⊗ e_i (for the i-th output).
    # This is the identity tensor of size 4, which has rank 4.
    # So Part 3 should be rank 4, not 16!
    #
    # Part 4: A_R (4×1) * B_BR (1×1) -> 4×1.
    # C[i] = A_R[i]*B_BR. This is scaling, rank 1? 
    # Actually C[i] = A_R[i]*B_BR[0]. The tensor is sum_i e_i ⊗ e_0 ⊗ e_i.
    # This has rank 4 (one per i). But wait, it's just scaling each element.
    # The bilinear complexity is 4 (one multiplication per output).
    #
    # So the corrected ranks:
    # Part 1: 4×4×4, rank 49
    # Part 2: 4×1×4 (outer product), rank 1
    # Part 3: 4×4×1 (matrix-vector), rank 4
    # Part 4: 4×1×1 (scaling), rank 4
    # Total: 49 + 1 + 4 + 4 = 58
    #
    # That's a big improvement from 85 to 58!
    
    rank = 58
    U = [[0] * rank for _ in range(n * m)]  # 20 rows
    V = [[0] * rank for _ in range(m * p)]  # 25 rows
    W = [[0] * rank for _ in range(p * n)]  # 20 rows
    
    # Strassen 2x2 factors
    def get_strassen_2x2_factors():
        u = [[0]*7 for _ in range(4)]
        v = [[0]*7 for _ in range(4)]
        w = [[0]*7 for _ in range(4)]
        
        # Term 0: (a00+a11)(b00+b11)
        u[0][0]=1; u[3][0]=1
        v[0][0]=1; v[3][0]=1
        w[0][0]=1; w[3][0]=1
        
        # Term 1: (a10+a11)b00
        u[2][1]=1; u[3][1]=1
        v[0][1]=1
        w[2][1]=1; w[3][1]=-1
        
        # Term 2: a00(b01-b11)
        u[0][2]=1
        v[1][2]=1; v[3][2]=-1
        w[1][2]=1; w[3][2]=1
        
        # Term 3: a11(b10-b00)
        u[3][3]=1
        v[2][3]=1; v[0][3]=-1
        w[0][3]=1; w[2][3]=1
        
        # Term 4: (a00+a01)b11
        u[0][4]=1; u[1][4]=1
        v[3][4]=1
        w[0][4]=-1; w[1][4]=1
        
        # Term 5: (a10-a00)(b00+b01)
        u[0][5]=-1; u[2][5]=1
        v[0][5]=1; v[1][5]=1
        w[3][5]=1
        
        # Term 6: (a01-a11)(b10+b11)
        u[1][6]=1; u[3][6]=-1
        v[2][6]=1; v[3][6]=1
        w[0][6]=1
        
        return u, v, w
    
    s_u, s_v, s_w = get_strassen_2x2_factors()
    
    # Part 1: 4×4×4 using Strassen (rank 49)
    def add_4x4_term(a_block_coeffs, b_block_coeffs, c_block_coeffs, start_col):
        for inner_k in range(7):
            u_vec = [s_u[r][inner_k] for r in range(4)]
            v_vec = [s_v[r][inner_k] for r in range(4)]
            w_vec = [s_w[r][inner_k] for r in range(4)]
            col_idx = start_col + inner_k
            
            for (a_br, a_bc), a_coeff in a_block_coeffs.items():
                for lr in range(2):
                    for lc in range(2):
                        local_idx = lr * 2 + lc
                        u_val = u_vec[local_idx]
                        if u_val != 0:
                            global_i = 2 * a_br + lr
                            global_j = 2 * a_bc + lc
                            global_u_row = global_i * 5 + global_j
                            U[global_u_row][col_idx] += a_coeff * u_val
            
            for (b_br, b_bc), b_coeff in b_block_coeffs.items():
                for lr in range(2):
                    for lc in range(2):
                        local_idx = lr * 2 + lc
                        v_val = v_vec[local_idx]
                        if v_val != 0:
                            global_j = 2 * b_br + lr
                            global_k = 2 * b_bc + lc
                            global_v_row = global_j * 5 + global_k
                            V[global_v_row][col_idx] += b_coeff * v_val
            
            for (c_br, c_bc), c_coeff in c_block_coeffs.items():
                for lr in range(2):
                    for lc in range(2):
                        local_idx = lr * 2 + lc
                        w_val = w_vec[local_idx]
                        if w_val != 0:
                            global_k = 2 * c_bc + lc
                            global_i = 2 * c_br + lr
                            global_w_row = global_k * 4 + global_i
                            W[global_w_row][col_idx] += c_coeff * w_val
    
    # Strassen 4x4 terms
    terms = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),  # M1
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),  # M2
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),  # M3
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),  # M4
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),  # M5
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),  # M6
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),  # M7
    ]
    
    for t_idx, (a_coeff, b_coeff, c_coeff) in enumerate(terms):
        start_col = t_idx * 7
        add_4x4_term(a_coeff, b_coeff, c_coeff, start_col)
    
    col = 49
    
    # Part 2: A_R * B_BL -> 4×4 (outer product), rank 1
    # C[i,j] = A_R[i] * B_BL[j]
    # U[i*5+4] = A_R[i] for i in 0..3
    # V[4*5+j] = B_BL[j] for j in 0..3
    # W[j*4+i] = C[i,j] for i in 0..3, j in 0..3
    for i in range(4):
        U[i * 5 + 4][col] = 1
    for j in range(4):
        V[4 * 5 + j][col] = 1
    for i in range(4):
        for j in range(4):
            W[j * 4 + i][col] = 1
    col += 1  # col = 50
    
    # Part 3: A_L * B_TR -> 4×1 (matrix-vector), rank 4
    # C[i] = sum_j A_L[i,j] * B_TR[j]
    # For each i, we need a separate term (since the sum is over j but i is fixed)
    # Actually, the tensor is: for each i, C[i] = dot(A_L[i,:], B_TR)
    # This can be done with 4 rank-1 terms:
    # Term i: U[i*5+j] = A_L[i,j] for all j, V[j*5+4] = B_TR[j] for all j, W[4*4+i] = C[i]
    # Wait, that's 4 terms, each with 4 entries in U, 4 in V, 1 in W.
    for i in range(4):
        for j in range(4):
            U[i * 5 + j][col] = 1
            V[j * 5 + 4][col] = 1
        W[4 * 4 + i][col] = 1
        col += 1  # col = 54
    
    # Part 4: A_R * B_BR -> 4×1 (scaling), rank 4
    # C[i] = A_R[i] * B_BR
    for i in range(4):
        U[i * 5 + 4][col] = 1
        V[4 * 5 + 4][col] = 1
        W[4 * 4 + i][col] = 1
        col += 1  # col = 58
    
    return {"U": U, "V": V, "W": W}
