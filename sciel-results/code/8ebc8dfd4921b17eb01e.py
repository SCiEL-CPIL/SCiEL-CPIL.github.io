import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # n=4, m=4, p=5
    
    # Part 1: 4x4 * 4x4 (k=0..3)
    # Rank 49
    # Part 2: 4x4 * 4x1 (k=4)
    # Rank 16
    
    rank1 = 49
    rank2 = 16
    total_rank = rank1 + rank2
    
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    # --- Part 1: Strassen 4x4x4 ---
    # We need to map the 49 scalar multiplications to the 16x16x16 tensor.
    # Let's define the Strassen 2x2x2 decomposition first.
    # s in 0..6
    
    # Coefficients for A, B, C for each of the 7 terms in 2x2x2
    # A indices: 0,1,2,3 (00, 01, 10, 11)
    # B indices: 0,1,2,3
    # C indices: 0,1,2,3
    
    # Term 0: M1 = (A00+A11)(B00+B11)
    # Term 1: M2 = A11(B01-B11)
    # Term 2: M3 = (A00+A01)B11
    # Term 3: M4 = A00(B10-B00)
    # Term 4: M5 = (A10+A11)B00
    # Term 5: M6 = (A00-A10)(B10+B11)
    # Term 6: M7 = (A01-A11)(B00+B10)
    
    # We can generate the 2x2x2 U, V, W for each term l.
    # Then for 4x4x4, we substitute the 2x2 blocks with their 2x2 scalar components.
    
    # Let's hardcode the 49 multiplications.
    # This is tedious but safe.
    # Alternatively, use the recursive structure.
    
    # Let's try to build it programmatically.
    
    # 2x2x2 Strassen Tensor
    # U2: 4x7, V2: 4x7, W2: 4x7
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # Term 0: (A0+A3)(B0+B3) -> C0 += 1, C1 += 0, C2 += 0, C3 += 0 ? No.
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Indices for 2x2:
    # A: 00, 01, 10, 11 -> 0, 1, 2, 3
    # B: 00, 01, 10, 11 -> 0, 1, 2, 3
    # C: 00, 01, 10, 11 -> 0, 1, 2, 3
    
    # Term 0 (M1):
    # A: 00(1), 11(1) -> U2[0,0]=1, U2[3,0]=1
    # B: 00(1), 11(1) -> V2[0,0]=1, V2[3,0]=1
    # C: 00(1), 11(1) -> W2[0,0]=1, W2[3,0]=1
    
    U2[0,0] = 1; U2[3,0] = 1
    V2[0,0] = 1; V2[3,0] = 1
    W2[0,0] = 1; W2[3,0] = 1
    
    # Term 1 (M2): A11 * (B01 - B11)
    U2[3,1] = 1
    V2[1,1] = 1; V2[3,1] = -1
    # C contribution: C10 += 1, C11 += -1
    W2[2,1] = 1; W2[3,1] = -1
    
    # Term 2 (M3): (A00 + A01) * B11
    U2[0,2] = 1; U2[1,2] = 1
    V2[3,2] = 1
    # C contribution: C01 += 1, C11 += 1
    W2[1,2] = 1; W2[3,2] = 1
    
    # Term 3 (M4): A00 * (B10 - B00)
    U2[0,3] = 1
    V2[2,3] = 1; V2[0,3] = -1
    # C contribution: C00 += 1, C10 += 1
    W2[0,3] = 1; W2[2,3] = 1
    
    # Term 4 (M5): (A10 + A11) * B00
    U2[2,4] = 1; U2[3,4] = 1
    V2[0,4] = 1
    # C contribution: C00 += -1, C01 += 1
    W2[0,4] = -1; W2[1,4] = 1
    
    # Term 5 (M6): (A00 - A10) * (B10 + B11)
    U2[0,5] = 1; U2[2,5] = -1
    V2[2,5] = 1; V2[3,5] = 1
    # C contribution: C11 += 1
    W2[3,5] = 1
    
    # Term 6 (M7): (A01 - A11) * (B00 + B10)
    U2[1,6] = 1; U2[3,6] = -1
    V2[0,6] = 1; V2[2,6] = 1
    # C contribution: C00 += 1
    W2[0,6] = 1
    
    # Now build 4x4x4 from 2x2x2.
    # The 4x4 matrix is composed of 2x2 blocks.
    # A_4x4 indices (i,j) where i,j in 0..3.
    # Block A_ab (a,b in 0..1) contains A_4x4[2a+x, 2b+y] for x,y in 0..1.
    # So A_4x4 index is 2a+x, 2b+y.
    # In the 2x2 tensor, the index is a*2+b.
    
    # For each of the 7 terms in 4x4 Strassen:
    # We need to substitute the 2x2 blocks with the actual 4x4 entries.
    
    # Let's iterate through the 7 top-level terms (l=0..6).
    # For each term, we have a 2x2x2 tensor (U2, V2, W2).
    # We need to expand this into 4x4x4.
    
    # Wait, the 4x4 Strassen algorithm uses 7 products of 2x2 matrices.
    # Each 2x2 product is computed using the 2x2 Strassen algorithm (7 scalar mults).
    # So total 49 scalar mults.
    
    # Let's generate U, V, W for the 49 terms.
    # col = 0
    for l in range(7):
        # Get the 2x2x2 tensor for this term from U2, V2, W2 column l
        u2 = U2[:, l]
        v2 = V2[:, l]
        w2 = W2[:, l]
        
        # Now, for each non-zero entry in u2, v2, w2, we need to expand it to 4x4.
        # u2[idx] is the coefficient for A_block[idx].
        # A_block[idx] corresponds to A_4x4[2*(idx//2) + x, 2*(idx%2) + y].
        # So for a fixed l and s (scalar mult within the 2x2 product),
        # we are effectively doing:
        # Product = (sum u2[a] A_a) * (sum v2[b] B_b)
        # = sum_a sum_b u2[a] v2[b] A_a B_b
        
        # Wait, this is not how Strassen works.
        # Strassen computes 7 specific linear combinations.
        # M_l = L_A(l) * L_B(l)
        # Where L_A(l) is a 2x2 matrix (linear combo of A blocks)
        # and L_B(l) is a 2x2 matrix.
        
        # The 2x2 product M_l is then computed using 7 scalar multiplications.
        # Let's call the scalar multiplications s=0..6.
        # For each s, we have:
        # ScalarProd_s = (sum_{a} u2[a] A_a)_{i,j} * (sum_{b} v2[b] B_b)_{k,m} ?
        # No.
        
        # Let's be precise.
        # The 4x4 Strassen algorithm computes 7 matrices M_l (each 2x2).
        # M_l = A_l * B_l, where A_l and B_l are 2x2 matrices formed from linear combos of A and B blocks.
        # Then each M_l is computed using the 2x2 Strassen algorithm, which uses 7 scalar multiplications.
        
        # So for each l (0..6), we have a 2x2x2 tensor.
        # Let's call the 2x2x2 tensor for the l-th product T_l.
        # T_l has rank 7.
        # The entries of T_l are determined by the coefficients of A_l and B_l.
        
        # How to get T_l?
        # A_l is a 2x2 matrix. Its entries are linear combos of A_00, A_01, A_10, A_11.
        # B_l is a 2x2 matrix. Its entries are linear combos of B_00, B_01, B_10, B_11.
        
        # Let A_l = [[a00, a01], [a10, a11]]
        # Let B_l = [[b00, b01], [b10, b11]]
        
        # The product M_l = A_l B_l.
        # We decompose M_l using the 2x2 Strassen tensor (U2, V2, W2).
        # The 2x2 Strassen tensor computes:
        # C_2x2 = A_2x2 * B_2x2
        # Using 7 scalar multiplications.
        
        # So for each l, we substitute A_2x2 = A_l and B_2x2 = B_l into the 2x2 Strassen formula.
        
        # The 2x2 Strassen formula (from U2, V2, W2) is:
        # For s in 0..6:
        # Term_s = (sum_{i} U2[i, s] A_l[i]) * (sum_{j} V2[j, s] B_l[j])
        # And C_l[k] += W2[k, s] Term_s
        
        # Here A_l[i] is the i-th entry of A_l (00, 01, 10, 11).
        # A_l[i] is itself a linear combo of the 4x4 A blocks.
        
        # Let's define the coefficients of A_l and B_l for each l.
        
        # l=0: M1 = (A00+A11)(B00+B11)
        # A_l = [[1, 0], [0, 1]] (coeffs for A00, A01, A10, A11) -> A00=1, A11=1
        # B_l = [[1, 0], [0, 1]]
        
        # l=1: M2 = A11(B01-B11)
        # A_l = [[0, 0], [0, 1]]
        # B_l = [[0, 1], [0, -1]]
        
        # l=2: M3 = (A00+A01)B11
        # A_l = [[1, 1], [0, 0]]
        # B_l = [[0, 0], [0, 1]]
        
        # l=3: M4 = A00(B10-B00)
        # A_l = [[1, 0], [0, 0]]
        # B_l = [[-1, 0], [1, 0]]
        
        # l=4: M5 = (A10+A11)B00
        # A_l = [[0, 0], [1, 1]]
        # B_l = [[1, 0], [0, 0]]
        
        # l=5: M6 = (A00-A10)(B10+B11)
        # A_l = [[1, 0], [-1, 0]]
        # B_l = [[0, 0], [1, 1]]
        
        # l=6: M7 = (A01-A11)(B00+B10)
        # A_l = [[0, 1], [0, -1]]
        # B_l = [[1, 0], [1, 0]]
        
        A_l_coeffs = [
            [[1, 0], [0, 1]],
            [[0, 0], [0, 1]],
            [[1, 1], [0, 0]],
            [[1, 0], [0, 0]],
            [[0, 0], [1, 1]],
            [[1, 0], [-1, 0]],
            [[0, 1], [0, -1]]
        ]
        B_l_coeffs = [
            [[1, 0], [0, 1]],
            [[0, 1], [0, -1]],
            [[0, 0], [0, 1]],
            [[-1, 0], [1, 0]],
            [[1, 0], [0, 0]],
            [[0, 0], [1, 1]],
            [[1, 0], [1, 0]]
        ]
        
        for s in range(7):
            col = l * 7 + s
            
            # Get the 2x2 Strassen coefficients for this scalar multiplication s
            u2_s = U2[:, s]
            v2_s = V2[:, s]
            w2_s = W2[:, s]
            
            # u2_s has 4 entries (for A00, A01, A10, A11 of A_l)
            # v2_s has 4 entries (for B00, B01, B10, B11 of B_l)
            
            # We need to fill U[col], V[col], W[col]
            
            # U[col] has 16 entries (for A_4x4)
            # V[col] has 16 entries (for B_4x4, k=0..3)
            # W[col] has 16 entries (for C_4x4, k=0..3, i=0..3)
            
            # For each a in 0..3 (index in A_l):
            # if u2_s[a] != 0:
            #   A_l[a] is a 2x2 block? No, A_l is a 2x2 matrix of coefficients.
            #   A_l[a] corresponds to the coefficient of A_block[a] in the linear combo.
            #   Wait, A_l is a 2x2 matrix.
            #   A_l = [[c00, c01], [c10, c11]]
            #   where c00 is coeff of A00, c01 of A01, etc.
            #   So A_l[a] is just a scalar coefficient.
            
            # Let's re-read the Strassen step.
            # M_l = A_l * B_l
            # A_l is a 2x2 matrix.
            # The 2x2 Strassen algorithm takes a 2x2 matrix A_l and B_l.
            # It computes 7 scalar multiplications.
            # The input to the 2x2 Strassen algorithm is the 4 entries of A_l and B_l.
            # So u2_s[a] is the coefficient of the a-th entry of A_l in the s-th scalar multiplication.
            
            # So, for the s-th scalar multiplication of the l-th term:
            # Term = (sum_{a=0}^3 u2_s[a] * A_l[a]) * (sum_{b=0}^3 v2_s[b] * B_l[b])
            
            # A_l[a] is the coefficient of A_block[a] in the linear combination for A_l.
            # A_block[a] is a 2x2 submatrix of A_4x4.
            # A_block[a] has indices (2*(a//2)+x, 2*(a%2)+y).
            
            # So the coefficient for A_4x4[i, j] in this term is:
            # sum_{a: A_l[a] != 0} u2_s[a] * A_l[a] * (1 if (i,j) is in A_block[a] else 0)
            
            # Similarly for B.
            
            # And for C:
            # The result of the 2x2 product M_l is added to C_l.
            # C_l is a 2x2 matrix of coefficients for C_4x4 blocks.
            # The 2x2 Strassen algorithm outputs C_2x2.
            # w2_s[k] is the coefficient of the k-th entry of C_2x2.
            # C_2x2[k] is the coefficient of C_block[k] in the final sum.
            
            # So the coefficient for C_4x4[i, j] is:
            # sum_{k: w2_s[k] != 0} w2_s[k] * (1 if (i,j) is in C_block[k] else 0)
            
            # Let's implement this.
            
            for a in range(4):
                if u2_s[a] != 0:
                    # A_l[a] is the coefficient
                    coeff_a = A_l_coeffs[l][a//2][a%2]
                    if coeff_a != 0:
                        # A_block[a] is at (2*(a//2), 2*(a%2))
                        base_i = 2 * (a // 2)
                        base_j = 2 * (a % 2)
                        for x in range(2):
                            for y in range(2):
                                i = base_i + x
                                j = base_j + y
                                idx = i * m + j
                                U[idx, col] += u2_s[a] * coeff_a
            
            for b in range(4):
                if v2_s[b] != 0:
                    coeff_b = B_l_coeffs[l][b//2][b%2]
                    if coeff_b != 0:
                        base_j = 2 * (b // 2)
                        base_k = 2 * (b % 2)
                        for x in range(2):
                            for y in range(2):
                                j = base_j + x
                                k = base_k + y
                                idx = j * p + k  # p=5, but k<4
                                V[idx, col] += v2_s[b] * coeff_b
            
            for k in range(4):
                if w2_s[k] != 0:
                    # C_block[k] is at (2*(k//2), 2*(k%2))
                    base_k = 2 * (k // 2)
                    base_i = 2 * (k % 2)
                    for x in range(2):
                        for y in range(2):
                            k_out = base_k + x
                            i_out = base_i + y
                            idx = k_out * n + i_out
                            W[idx, col] += w2_s[k]
    
    # --- Part 2: Schoolbook 4x4x1 ---
    # k=4
    # U2: 16x16
    # V2: 4x16 (only k=4)
    # W2: 4x16 (only k=4)
    
    col_offset = 49
    for i in range(4):
        for j in range(4):
            col = col_offset + i * 4 + j
            U[i * m + j, col] = 1
            V[j * p + 4, col] = 1
            W[4 * n + i, col] = 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
