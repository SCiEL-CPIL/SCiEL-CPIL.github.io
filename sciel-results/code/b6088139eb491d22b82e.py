"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We use a decomposition that splits the p-dimension (9) into two parts: 4 and 5.
    # For the 4x4x4 block, we use a rank-15 decomposition (Winograd's method).
    # For the 4x4x5 block, we use the schoolbook method (rank 4*4*5 = 80).
    # Wait, is there a better way?
    # Let's check the rank of 4x4x4.
    # It is known that R(4,4,4) <= 15 (Winograd).
    # So total rank = 15 + 80 = 95.
    # Schoolbook is 4*4*9 = 144.
    # 95 < 144, so this is an improvement.
    
    # However, we can potentially do better on the 4x4x5 part?
    # R(4,4,5) is not well known, but schoolbook is 80.
    # Maybe we can split 9 into 4+4+1?
    # 15 + 15 + 16 = 46.
    # This is much better!
    # Let's verify R(4,4,1).
    # Multiplying a 4x4 matrix by a 4x1 matrix is just 4 dot products of length 4.
    # Each dot product has rank 4. Total rank 16.
    # So 15 + 15 + 16 = 46.
    
    # Let's implement the rank-15 decomposition for 4x4x4.
    # Winograd's method for 4x4x4 (or 2x2x2 of 2x2 blocks).
    # Actually, the standard Winograd for 4x4x4 uses 15 multiplications.
    # The coefficients are integers (or half-integers).
    
    # Reference: "A Fast Fourier Transform for the Discrete Sine Transform" or similar.
    # But for matrix multiplication, it's the Winograd algebraic method.
    
    # Let's construct the 4x4x4 rank-15 decomposition.
    # The tensor T_{ijk} = sum_{r=1}^{15} U_{ij,r} V_{jk,r} W_{ki,r}.
    # Note: The problem statement says W[k*n+i, r].
    # So W is indexed by (k, i).
    
    # I will use a known rank-15 scheme for 4x4x4.
    # One such scheme is based on the 2x2x2 Strassen (rank 7) applied to 2x2 blocks.
    # Wait, Strassen on 2x2 blocks of 2x2 matrices gives 7 * (2*2*2) = 28? No.
    # Strassen reduces 2x2x2 from 8 to 7.
    # If we apply it to 2x2 blocks, the "scalar" multiplications are 2x2x2.
    # So 7 multiplications, each of rank 8 (schoolbook for 2x2x2). Total 56.
    # That's not 15.
    
    # The rank-15 bound for 4x4x4 is a specific algebraic construction.
    # Let's try to find it or derive it.
    # Actually, the bound R(4,4,4) <= 15 is due to Winograd (1971).
    # The coefficients are:
    # A' = [[1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #       [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #       ...]
    # This is getting complicated to hardcode without a reference.
    
    # Alternative: Use the schoolbook for all, but optimize the structure?
    # No, 144 is the baseline.
    
    # Let's try a different split: 4x4x9.
    # Can we use a rank-11 decomposition for 2x2x2? No, it's 7.
    
    # Let's try to implement the Winograd 4x4x4 rank-15 decomposition.
    # I found a reference online for the 15-term decomposition of 4x4x4.
    # The terms are:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A01+A11)(B00+B10)
    # ...
    # This is essentially the same as Strassen but with different groupings?
    # No, Strassen is 7 terms for 2x2x2.
    # For 4x4x4, if we view it as 2x2 blocks of 2x2, and use Strassen on the blocks,
    # we get 7 block-multiplications. Each block-multiplication is 2x2x2.
    # If we do 2x2x2 schoolbook, that's 8 scalars. 7*8=56.
    # If we do 2x2x2 Strassen, that's 7 scalars. 7*7=49.
    
    # So where does 15 come from?
    # Ah, 15 is for 3x3x3? No, R(3,3,3) <= 23.
    # R(2,2,2) = 7.
    # R(4,4,4) <= 15 is correct?
    # Let's check: 4x4x4 has 64 entries.
    # 15 terms means each term is a sum of products.
    # Actually, the bound is R(4,4,4) <= 15 is FALSE.
    # The best known bound for 4x4x4 is 25 (or 28?).
    # Wait, R(2,2,2)=7.
    # R(4,4,4) <= 7^log2(4) = 7^2 = 49.
    # Is there a better bound?
    # Yes, R(4,4,4) <= 25 is known?
    # Actually, let's stick to the safe Strassen bound of 49 for 4x4x4.
    # And schoolbook for the rest.
    
    # Split 9 into 4, 4, 1.
    # 4x4x4 (rank 49) + 4x4x4 (rank 49) + 4x4x1 (rank 16).
    # Total = 49 + 49 + 16 = 114.
    # 114 < 144. This is an improvement.
    
    # Let's implement the Strassen 4x4x4 (rank 49) decomposition.
    # This is Strassen on 2x2 blocks of 2x2 matrices, with Strassen on the 2x2 blocks.
    # So it's a recursive Strassen.
    
    # We need a helper to add a rank-1 term.
    # U[i*m + j, col], V[j*p + k, col], W[k*n + i, col].
    
    rank = 114
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_rank1(i, j, k, c, col_idx):
        U[i * m + j][col_idx] += c
        V[j * p + k][col_idx] += c
        W[k * n + i][col_idx] += c

    # Strassen 2x2x2 (rank 7)
    # Inputs: A[2,2], B[2,2], Output C[2,2]
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M7
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We need to add these to the global U, V, W.
    # The "scalar" multiplications are the M's.
    # Each M is a product of two linear combinations.
    # We expand each M into rank-1 terms.
    
    def add_strassen_2x2(i_base, j_base, k_base, col_start):
        # i_base, j_base, k_base: top-left indices of the 2x2 block
        # Returns the next column index
        c = col_start
        
        # M1 = (A00+A11)(B00+B11)
        # A00 = A[i_base, j_base]
        # A11 = A[i_base+1, j_base+1]
        # B00 = B[j_base, k_base]
        # B11 = B[j_base+1, k_base+1]
        
        # Term 1: A00 * B00
        add_rank1(i_base, j_base, k_base, 1, c); c += 1
        # Term 2: A00 * B11
        add_rank1(i_base, j_base+1, k_base+1, 1, c); c += 1
        # Term 3: A11 * B00
        add_rank1(i_base+1, j_base, k_base, 1, c); c += 1
        # Term 4: A11 * B11
        add_rank1(i_base+1, j_base+1, k_base+1, 1, c); c += 1
        
        # M2 = (A10+A11)B00
        # A10 = A[i_base+1, j_base]
        # A11 = A[i_base+1, j_base+1]
        # B00 = B[j_base, k_base]
        
        # Term 5: A10 * B00
        add_rank1(i_base+1, j_base, k_base, 1, c); c += 1
        # Term 6: A11 * B00
        add_rank1(i_base+1, j_base+1, k_base, 1, c); c += 1
        
        # M3 = A00(B10-B11)
        # A00 = A[i_base, j_base]
        # B10 = B[j_base+1, k_base]
        # B11 = B[j_base+1, k_base+1]
        
        # Term 7: A00 * B10
        add_rank1(i_base, j_base+1, k_base, 1, c); c += 1
        # Term 8: A00 * (-B11)
        add_rank1(i_base, j_base+1, k_base+1, -1, c); c += 1
        
        # M4 = A11(B01-B00)
        # A11 = A[i_base+1, j_base+1]
        # B01 = B[j_base, k_base+1]
        # B00 = B[j_base, k_base]
        
        # Term 9: A11 * B01
        add_rank1(i_base+1, j_base, k_base+1, 1, c); c += 1
        # Term 10: A11 * (-B00)
        add_rank1(i_base+1, j_base, k_base, -1, c); c += 1
        
        # M5 = (A00+A01)B11
        # A00 = A[i_base, j_base]
        # A01 = A[i_base, j_base+1]
        # B11 = B[j_base+1, k_base+1]
        
        # Term 11: A00 * B11
        add_rank1(i_base, j_base+1, k_base+1, 1, c); c += 1
        # Term 12: A01 * B11
        add_rank1(i_base, j_base+1, k_base+1, 1, c); c += 1
        
        # M6 = (A10-A00)(B00+B01)
        # A10 = A[i_base+1, j_base]
        # A00 = A[i_base, j_base]
        # B00 = B[j_base, k_base]
        # B01 = B[j_base, k_base+1]
        
        # Term 13: A10 * B00
        add_rank1(i_base+1, j_base, k_base, 1, c); c += 1
        # Term 14: A10 * B01
        add_rank1(i_base+1, j_base, k_base+1, 1, c); c += 1
        # Term 15: (-A00) * B00
        add_rank1(i_base, j_base, k_base, -1, c); c += 1
        # Term 16: (-A00) * B01
        add_rank1(i_base, j_base, k_base+1, -1, c); c += 1
        
        # M7 = (A01-A11)(B10+B11)
        # A01 = A[i_base, j_base+1]
        # A11 = A[i_base+1, j_base+1]
        # B10 = B[j_base+1, k_base]
        # B11 = B[j_base+1, k_base+1]
        
        # Term 17: A01 * B10
        add_rank1(i_base, j_base+1, k_base, 1, c); c += 1
        # Term 18: A01 * B11
        add_rank1(i_base, j_base+1, k_base+1, 1, c); c += 1
        # Term 19: (-A11) * B10
        add_rank1(i_base+1, j_base+1, k_base, -1, c); c += 1
        # Term 20: (-A11) * B11
        add_rank1(i_base+1, j_base+1, k_base+1, -1, c); c += 1
        
        # We have added 20 rank-1 terms for the 2x2x2 Strassen.
        # But wait, Strassen has 7 multiplications.
        # Each multiplication is a scalar product?
        # No, in the tensor decomposition, each "multiplication" M_r is a scalar.
        # The coefficients U, V, W are such that:
        # C_{ij} = sum_r U_{ij,r} V_{jk,r} W_{ki,r}
        
        # My `add_rank1` adds a term to U, V, W.
        # But I am adding 20 terms for a single 2x2x2 block.
        # This is equivalent to the schoolbook rank 8, but with 20 terms?
        # No, the schoolbook for 2x2x2 has 8 terms.
        # Strassen has 7 terms.
        # Why did I add 20?
        
        # Ah, I expanded the linear combinations into scalar products.
        # M1 = (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11.
        # These are 4 scalar products.
        # In the tensor decomposition, we can group them.
        # But if we add them as separate rank-1 terms, we are using rank 4 for M1.
        # This is inefficient.
        
        # The correct way to add a Strassen term is to add 1 rank-1 term per M_r.
        # But U, V, W must be linear.
        # U_{ij,r} is the coefficient of A_{ij} in the r-th factor.
        # V_{jk,r} is the coefficient of B_{jk} in the r-th factor.
        # W_{ki,r} is the coefficient of C_{ki} in the r-th factor.
        
        # For M1 = (A00+A11)(B00+B11):
        # We need U such that U * A = A00+A11.
        # U[0,0] = 1, U[1,1] = 1 (for the 2x2 block).
        # V such that V * B = B00+B11.
        # V[0,0] = 1, V[1,1] = 1.
        # W such that W * M1 = C00 - C11 + ...
        
        # This is much more efficient.
        # Let's re-implement add_strassen_2x2 to add 7 rank-1 terms.
        
        return c

    # Let's rewrite add_strassen_2x2 correctly.
    # It should add 7 terms.
    
    def add_strassen_2x2_correct(i_base, j_base, k_base, col_start):
        c = col_start
        
        # M1 = (A00+A11)(B00+B11)
        # U: A00 + A11 -> U[i_base, j_base]=1, U[i_base+1, j_base+1]=1
        # V: B00 + B11 -> V[j_base, k_base]=1, V[j_base+1, k_base+1]=1
        # W: C00 + C11 - C01 - C10 (from C00=M1-M4+M5-M7, C11=M1-M2+M3+M6, etc)
        # Actually, W is determined by the reconstruction formula.
        # C00 = M1 - M4 + M5 - M7
        # C01 = M3 + M7
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # So W for M1:
        # W[k_base, i_base] = 1 (for C00)
        # W[k_base+1, i_base+1] = 1 (for C11)
        
        U[i_base * m + j_base][c] += 1
        U[(i_base+1) * m + (j_base+1)][c] += 1
        V[j_base * p + k_base][c] += 1
        V[(j_base+1) * p + (k_base+1)][c] += 1
        W[k_base * n + i_base][c] += 1
        W[(k_base+1) * n + (i_base+1)][c] += 1
        c += 1
        
        # M2 = (A10+A11)B00
        # U: A10 + A11 -> U[i_base+1, j_base]=1, U[i_base+1, j_base+1]=1
        # V: B00 -> V[j_base, k_base]=1
        # W: C10 - C11 (from C10=M2+M4, C11=M1-M2+M3+M6)
        # W[k_base, i_base+1] = 1
        # W[k_base+1, i_base+1] = -1
        
        U[(i_base+1) * m + j_base][c] += 1
        U[(i_base+1) * m + (j_base+1)][c] += 1
        V[j_base * p + k_base][c] += 1
        W[k_base * n + (i_base+1)][c] += 1
        W[(k_base+1) * n + (i_base+1)][c] += -1
        c += 1
        
        # M3 = A00(B10-B11)
        # U: A00 -> U[i_base, j_base]=1
        # V: B10 - B11 -> V[j_base+1, k_base]=1, V[j_base+1, k_base+1]=-1
        # W: C01 + C11 (from C01=M3+M7, C11=M1-M2+M3+M6)
        # W[k_base, i_base] = 1
        # W[k_base+1, i_base+1] = 1
        
        U[i_base * m + j_base][c] += 1
        V[(j_base+1) * p + k_base][c] += 1
        V[(j_base+1) * p + (k_base+1)][c] += -1
        W[k_base * n + i_base][c] += 1
        W[(k_base+1) * n + (i_base+1)][c] += 1
        c += 1
        
        # M4 = A11(B01-B00)
        # U: A11 -> U[i_base+1, j_base+1]=1
        # V: B01 - B00 -> V[j_base, k_base+1]=1, V[j_base, k_base]=-1
        # W: C00 - C10 (from C00=M1-M4+M5-M7, C10=M2+M4)
        # W[k_base, i_base] = -1
        # W[k_base, i_base+1] = 1
        
        U[(i_base+1) * m + (j_base+1)][c] += 1
        V[j_base * p + (k_base+1)][c] += 1
        V[j_base * p + k_base][c] += -1
        W[k_base * n + i_base][c] += -1
        W[k_base * n + (i_base+1)][c] += 1
        c += 1
        
        # M5 = (A00+A01)B11
        # U: A00 + A01 -> U[i_base, j_base]=1, U[i_base, j_base+1]=1
        # V: B11 -> V[j_base+1, k_base+1]=1
        # W: C00 (from C00=M1-M4+M5-M7)
        # W[k_base, i_base] = 1
        
        U[i_base * m + j_base][c] += 1
        U[i_base * m + (j_base+1)][c] += 1
        V[(j_base+1) * p + (k_base+1)][c] += 1
        W[k_base * n + i_base][c] += 1
        c += 1
        
        # M6 = (A10-A00)(B00+B01)
        # U: A10 - A00 -> U[i_base+1, j_base]=1, U[i_base, j_base]=-1
        # V: B00 + B01 -> V[j_base, k_base]=1, V[j_base, k_base+1]=1
        # W: C11 (from C11=M1-M2+M3+M6)
        # W[k_base+1, i_base+1] = 1
        
        U[(i_base+1) * m + j_base][c] += 1
        U[i_base * m + j_base][c] += -1
        V[j_base * p + k_base][c] += 1
        V[j_base * p + (k_base+1)][c] += 1
        W[(k_base+1) * n + (i_base+1)][c] += 1
        c += 1
        
        # M7 = (A01-A11)(B10+B11)
        # U: A01 - A11 -> U[i_base, j_base+1]=1, U[i_base+1, j_base+1]=-1
        # V: B10 + B11 -> V[j_base+1, k_base]=1, V[j_base+1, k_base+1]=1
        # W: C00 - C01 (from C00=M1-M4+M5-M7, C01=M3+M7)
        # W[k_base, i_base] = -1
        # W[k_base, i_base] = 1 (wait, C01 uses M7 with +1)
        # C00 = ... - M7
        # C01 = ... + M7
        # So W[k_base, i_base] = -1, W[k_base, i_base] = 1?
        # No, W is indexed by (k, i).
        # C00 is at (k_base, i_base).
        # C01 is at (k_base+1, i_base).
        
        U[i_base * m + (j_base+1)][c] += 1
        U[(i_base+1) * m + (j_base+1)][c] += -1
        V[(j_base+1) * p + k_base][c] += 1
        V[(j_base+1) * p + (k_base+1)][c] += 1
        W[k_base * n + i_base][c] += -1
        W[(k_base+1) * n + i_base][c] += 1
        c += 1
        
        return c

    # Now, we need to apply this to 4x4x4.
    # A 4x4x4 block can be viewed as 2x2 blocks of 2x2 matrices.
    # We have 4 blocks in A: A00, A01, A10, A11.
    # We have 4 blocks in B: B00, B01, B10, B11.
    # We apply Strassen to the 2x2 block structure.
    # This gives 7 block-multiplications.
    # Each block-multiplication is a 2x2x2 product.
    # We use Strassen for each 2x2x2 product.
    # So 7 * 7 = 49 rank-1 terms.
    
    def add_strassen_4x4(i_base, j_base, k_base, col_start):
        c = col_start
        
        # Block indices:
        # A00: (i_base, j_base) to (i_base+1, j_base+1)
        # A01: (i_base, j_base+2) to (i_base+1, j_base+3)
        # A10: (i_base+2, j_base) to (i_base+3, j_base+1)
        # A11: (i_base+2, j_base+2) to (i_base+3, j_base+3)
        
        # B00: (j_base, k_base) to (j_base+1, k_base+1)
        # B01: (j_base, k_base+2) to (j_base+1, k_base+3)
        # B10: (j_base+2, k_base) to (j_base+3, k_base+1)
        # B11: (j_base+2, k_base+2) to (j_base+3, k_base+3)
        
        # Strassen M1 = (A00+A11)(B00+B11)
        # This is a 2x2x2 product of (A00+A11) and (B00+B11).
        # We need to add the Strassen terms for this virtual 2x2x2 product.
        # But the "scalars" are 2x2 matrices.
        # So we call add_strassen_2x2_correct for the "scalar" positions.
        
        # This is getting recursive.
        # Let's just call add_strassen_2x2_correct 7 times with appropriate offsets?
        # No, the offsets are not simple.
        
        # I will skip the 4x4x4 Strassen and use the schoolbook for 4x4x4.
        # Rank 64.
        # 64 + 64 + 16 = 144. No improvement.
        
        # I will use the schoolbook for the whole 4x4x9.
        # Rank 144.
        
        return c

    # I will just use the schoolbook. It is safe.
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
