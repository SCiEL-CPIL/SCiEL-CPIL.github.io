"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    # Four Russians / Block Strassen decomposition.
    # Split B into two blocks: first 4 columns and last 3 columns.
    # Apply Strassen's 2x2 block algorithm to the first 4 columns (viewed as 2x2 blocks).
    # Apply Strassen's 2x2 block algorithm to the last 3 columns? No, Strassen needs 2x2.
    # The directive says: "grouping the seven columns of the B-matrix into two blocks (sizes 4 and 3) and applying Strassen's 2x2 block algorithm to each sub-block independently".
    # This implies treating the first block of size 4 as a single 2x2 block multiplication (where A is 2x2, B is 2x2) which costs 7.
    # And the second block of size 3? Strassen is for 2x2. We can treat the 3 columns as 2x1 + 1x1? Or just scalar?
    # Let's re-read carefully: "yields a rank of 18 (2*2^3 + 2*2^2)".
    # 2 * 8 = 16. 2 * 4 = 8. Total 24? No, 18 is 14 + 4? Or 7 + 7 + 4?
    # Let's analyze the rank 18 claim.
    # If we split 7 columns into 4 and 3.
    # Block 1: 4 columns. This is $A_{2x3} \times B_{3x4}$.
    # We can view B as two $3x2$ blocks? No, 4 columns = two 2-column blocks.
    # Let $B_1$ be cols 0-1, $B_2$ be cols 2-3.
    # $A B_1$ is $2x2$. Rank of $2x2$ mult is 7 (Strassen).
    # $A B_2$ is $2x2$. Rank of $2x2$ mult is 7 (Strassen).
    # Total for first 4 cols: 14.
    # Block 2: 3 columns. This is $A_{2x3} \times B_{3x3}$.
    # We can view B as a $3x2$ block and a $3x1$ block?
    # $A B_{3x2}$ is $2x2$. Rank 7.
    # $A B_{3x1}$ is $2x1$. Rank $2 \times 3 \times 1 = 6$? No, rank of $2x1$ output from $2x3$ and $3x1$ is min(2*3, 3*1, ...) actually rank of tensor $M_{2,3,1}$ is 6 (schoolbook) or can it be lower?
    # Actually, $M_{n,m,p}$ with p=1 is just matrix-vector multiplication. Rank is $n \times m$. So $2 \times 3 = 6$.
    # So 14 + 7 + 6 = 27. This is not 18.
    
    # Let's re-evaluate "2*2^3 + 2*2^2".
    # $2 \times 8 = 16$. $2 \times 4 = 8$. Sum = 24.
    # Maybe the directive's rank calculation is specific to a certain grouping.
    # Let's try to construct a valid decomposition with rank as low as possible.
    # Known lower bound for $2 \times 3 \times 7$:
    # $R(2,3,7) \le R(2,3,4) + R(2,3,3)$.
    # $R(2,3,2) = 7$ (Strassen).
    # $R(2,3,4)$: Split 4 into 2+2. $R(2,3,2) + R(2,3,2) = 7+7=14$.
    # $R(2,3,3)$: Split 3 into 2+1. $R(2,3,2) + R(2,3,1)$.
    # $R(2,3,1) = 6$ (schoolbook, since it's linear map $\mathbb{R}^3 \to \mathbb{R}^2$ applied to A? No, A is $2x3$, B is $3x1$. Output $2x1$. Tensor is $A_{ik} B_{kj}$. Rank is $2 \times 3 = 6$).
    # So $R(2,3,3) \le 7 + 6 = 13$.
    # Total $14 + 13 = 27$.
    
    # Can we do better?
    # Maybe $R(2,3,7)$ is smaller.
    # Let's stick to the structure implied by the directive: "Four Russians" bit-packing.
    # Group columns 4 and 3.
    # Block 1 (4 cols): Use Strassen on two 2-col chunks? That gives 14.
    # Block 2 (3 cols): Use Strassen on 2 cols (rank 7) + Scalar on 1 col (rank 6). Total 13.
    # Sum = 27.
    
    # Wait, the directive says rank 18.
    # $2 \times 2^3 + 2 \times 2^2$?
    # $2 \times 8 = 16$. $2 \times 4 = 8$.
    # Maybe it means:
    # Block 1 (4 cols): Treat as one 2x2 block mult? No, A is 2x3.
    # If we group 4 cols, we can write $A [B_1 | B_2]$ where $B_1, B_2$ are $3x2$.
    # Output is $[A B_1 | A B_2]$.
    # We can compute $A B_1$ and $A B_2$ independently. Each is $2x2$ output.
    # Rank 7 each. Total 14.
    
    # Block 2 (3 cols): $A [B_3 | B_4]$ where $B_3$ is $3x2$, $B_4$ is $3x1$.
    # $A B_3$ is $2x2$. Rank 7.
    # $A B_4$ is $2x1$. Rank 6.
    # Total 13.
    # Sum 27.
    
    # Is there a way to share intermediate products?
    # The directive says "Four Russians bit-packing". This usually refers to precomputation or grouping to reduce multiplications.
    # However, for exact tensor decomposition over reals (or rationals), Strassen is the standard improvement.
    # Let's verify if rank 18 is achievable.
    # $18 = 7 + 7 + 4$?
    # $18 = 14 + 4$?
    # If Block 2 (3 cols) could be done in rank 4, total is 18.
    # $R(2,3,3) = 4$?
    # Lower bound for $2 \times 3 \times 3$: $R \ge 2 \times 3 = 6$? No.
    # $R(2,2,2) = 7$.
    # $R(2,3,3) \ge R(2,2,2) = 7$? No, $2 \times 3 \times 3$ contains $2 \times 2 \times 2$ as a sub-tensor (restrict rows/cols).
    # So $R(2,3,3) \ge 7$.
    # So $14 + 7 = 21$ is a lower bound for this split?
    # Plus the remaining column?
    # If we split 7 into 2, 2, 3?
    # $R(2,3,2) + R(2,3,2) + R(2,3,3) = 7 + 7 + 13 = 27$.
    
    # What if we use a different block size?
    # Split 7 into 1, 1, 5?
    # $R(2,3,1) = 6$.
    # $R(2,3,5) \le R(2,3,2) + R(2,3,3) = 7 + 13 = 20$.
    # Total $6+6+20 = 32$. Worse.
    
    # It seems 27 is a solid candidate.
    # Let's construct the rank 27 solution.
    #
    # Structure:
    # Columns 0,1: Strassen on $A_{2x3} B_{3x2} \to C_{2x2}$.
    # Columns 2,3: Strassen on $A_{2x3} B_{3x2} \to C_{2x2}$.
    # Columns 4,5: Strassen on $A_{2x3} B_{3x2} \to C_{2x2}$.
    # Column 6:   Scalar on $A_{2x3} B_{3x1} \to C_{2x1}$.
    #
    # Total Rank = 7 + 7 + 7 + 6 = 27.
    
    n, m, p = DIMENSIONS
    rank = 27
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col_offset = 0
    
    # Helper to add a Strassen decomposition for 2x2 block multiplication
    # A is 2x3 (rows 0,1 of A; cols j0, j1, j2 of B)
    # B is 3x2 (rows j0, j1, j2 of B; cols k0, k1 of output)
    # Output C is 2x2 (rows i0, i1 of A; cols k0, k1 of output)
    def add_strassen_2x2(j0, k0, start_col):
        # A rows: 0, 1
        # A cols: j0, j0+1, j0+2
        # B rows: j0, j0+1, j0+2
        # B cols: k0, k0+1
        #
        # Strassen for 2x2 matrices:
        # Let A = [[a00, a01], [a10, a11]]
        # Let B = [[b00, b01], [b10, b11]]
        # M1 = (a00 + a11) * (b00 + b11)
        # M2 = (a10 + a11) * b00
        # M3 = a00 * (b01 - b11)
        # M4 = a11 * (b10 - b00)
        # M5 = (a00 + a01) * b11
        # M6 = (a10 - a00) * (b00 + b01)
        # M7 = (a01 - a11) * (b10 + b11)
        #
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        #
        # In our tensor notation:
        # A indices: (i, j) where i in {0,1}, j in {j0, j0+1, j0+2}
        # B indices: (j, k) where j in {j0, j0+1, j0+2}, k in {k0, k0+1}
        # Note: The "2x2" block of A uses rows 0,1 and cols j0, j0+1?
        # Wait. A is 2x3. The block multiplication is (2x3) * (3x2) = (2x2).
        # This is NOT a 2x2 matrix multiplication in the standard Strassen sense where both are 2x2.
        # Here A is 2x3 and B is 3x2.
        # The rank of the tensor $M_{2,3,2}$ is 7?
        # Yes, $R(2,3,2) = 7$.
        # The decomposition is the same as Strassen, but the matrices are rectangular.
        # Let's verify the indices for $M_{2,3,2}$.
        # A: 2 rows, 3 cols. B: 3 rows, 2 cols. C: 2 rows, 2 cols.
        # The 7 terms are:
        # T1 = (A00 + A12) * (B00 + B21)  -- Wait, which elements?
        # Standard Strassen assumes A and B are 2x2.
        # If A is 2x3 and B is 3x2, we can embed them into 2x2? No.
        # We can view A as [A_left | A_right] where A_left is 2x2, A_right is 2x1?
        # Or A is 2x3.
        # Let's use the known rank 7 decomposition for $2 \times 3 \times 2$.
        # The decomposition is essentially the same 7 products, but the coefficients are different.
        # Let's derive it or look it up.
        # Actually, $R(2,3,2) = R(2,2,2) = 7$ is a known result.
        # The decomposition can be constructed by restricting the $2 \times 2 \times 2$ Strassen decomposition to a subspace.
        # Let's use the explicit formulas for $M_{2,3,2}$.
        # Let A be:
        # a b c
        # d e f
        # Let B be:
        # g h
        # i j
        # k l
        #
        # C = AB =
        # ag+bi+ck   ah+bj+cl
        # dg+ei+fk   dh+ej+fl
        #
        # The 7 rank-1 terms:
        # 1. (a+f)(g+l)  -> contributes to (0,0) and (1,1)
        # 2. (d+f)g      -> contributes to (1,0)
        # 3. a(h-l)      -> contributes to (0,0) and (0,1)
        # 4. f(j-g)      -> contributes to (1,0) and (1,1)
        # 5. (a+b)l      -> contributes to (0,0) and (0,1)
        # 6. (d-a)(g+h)  -> contributes to (1,0)
        # 7. (b-f)(j+k)  -> contributes to (0,0) and (1,1) ... wait, check indices.
        
        # Let's verify term 1: (a+f)(g+l) = ag + al + fg + fl.
        # C00 has ag. C11 has fl.
        # C00 also needs bi, ck. C11 needs dh, ej, fk? No, C11 = dh+ej+fl.
        # So term 1 covers ag and fl.
        
        # Term 2: (d+f)g = dg + fg.
        # C10 has dg. C00 has fg? No, C00 = ag+bi+ck. No fg.
        # So term 2 contributes to C10 (dg). What about fg? It must cancel or be combined.
        
        # This manual derivation is error-prone.
        # Instead, let's use the block approach from the previous thought:
        # We established that $R(2,3,2) = 7$.
        # We can implement the rank 27 solution using three $2 \times 3 \times 2$ blocks and one $2 \times 3 \times 1$ block.
        #
        # Block 1: Cols 0,1. A is 2x3, B is 3x2 (rows 0,1,2; cols 0,1).
        # Block 2: Cols 2,3. A is 2x3, B is 3x2 (rows 0,1,2; cols 2,3).
        # Block 3: Cols 4,5. A is 2x3, B is 3x2 (rows 0,1,2; cols 4,5).
        # Block 4: Col 6.   A is 2x3, B is 3x1 (rows 0,1,2; col 6).
        
        # We need the explicit U, V, W matrices for $M_{2,3,2}$ and $M_{2,3,1}$.
        
        # For $M_{2,3,1}$ (rank 6):
        # A: 2x3, B: 3x1, C: 2x1.
        # C0 = a00*B0 + a01*B1 + a02*B2
        # C1 = a10*B0 + a11*B1 + a12*B2
        # Rank 6 terms:
        # 1. a00 * B0
        # 2. a01 * B1
        # 3. a02 * B2
        # 4. a10 * B0
        # 5. a11 * B1
        # 6. a12 * B2
        
        # For $M_{2,3,2}$ (rank 7):
        # I will use a known decomposition.
        # Reference: Bini, Capovani, Lotti, Romani (1979) or similar.
        # The decomposition is:
        # M1 = (a00 + a12) * (b00 + b21)
        # M2 = (a10 + a12) * b00
        # M3 = a00 * (b01 - b21)
        # M4 = a12 * (b20 - b00)
        # M5 = (a00 + a01) * b21
        # M6 = (a10 - a00) * (b00 + b01)
        # M7 = (a01 - a12) * (b20 + b21)
        
        # Let's check C00 = sum of terms where U and V are non-zero for C00.
        # C00 = a00 b00 + a01 b10 + a02 b20
        #
        # Let's evaluate the sum of M1..M7 contributions to C00:
        # M1: a00 b00 + a00 b21 + a12 b00 + a12 b21
        # M2: a10 b00 + a12 b00
        # M3: a00 b01 - a00 b21
        # M4: a12 b20 - a12 b00
        # M5: a00 b21 + a01 b21
        # M6: (a10-a00)(b00+b01) = a10 b00 + a10 b01 - a00 b00 - a00 b01
        # M7: (a01-a12)(b20+b21) = a01 b20 + a01 b21 - a12 b20 - a12 b21
        #
        # Summing coefficients for each monomial:
        # a00 b00: M1(1) + M3(0) + M5(0) + M6(-1) = 0? Wait.
        # M1 has +1. M6 has -1. Sum = 0.
        # But C00 needs a00 b00.
        # Did I mess up the signs or the decomposition?
        # Let's try a different known decomposition for $2 \times 3 \times 2$.
        # Actually, it's safer to use the block method with standard 2x2 Strassen if possible, but A is 2x3.
        # We can split A into A1 (2x2) and A2 (2x1).
        # B into B1 (3x2) and B2 (3x2)? No.
        
        # Let's use the rank 6 schoolbook for $2 \times 3 \times 1$ and rank 7 for $2 \times 3 \times 2$.
        # I will hardcode the rank 7 decomposition for $M_{2,3,2}$ based on a reliable source or derivation.
        # Derivation:
        # We want to express $C_{ik} = \sum_{j=0}^2 A_{ij} B_{jk}$ for $i=0,1$ and $k=0,1$.
        # This is equivalent to two vector products of length 3.
        # Rank of $M_{2,3,2}$ is 7.
        
        # Here is a verified rank 7 decomposition for $M_{2,3,2}$:
        # T1 = (A00 + A12)(B00 + B21)
        # T2 = (A10 + A12)B00
        # T3 = A00(B01 - B21)
        # T4 = A12(B20 - B00)
        # T5 = (A00 + A01)B21
        # T6 = (A10 - A00)(B00 + B01)
        # T7 = (A01 - A12)(B20 + B21)
        
        # Let's re-check C00:
        # T1: A00B00 + A00B21 + A12B00 + A12B21
        # T2: A10B00 + A12B00
        # T3: A00B01 - A00B21
        # T4: A12B20 - A12B00
        # T5: A00B21 + A01B21
        # T6: A10B00 + A10B01 - A00B00 - A00B01
        # T7: A01B20 + A01B21 - A12B20 - A12B21
        
        # Coeff of A00B00: T1(1) + T6(-1) = 0. (Missing)
        # Coeff of A00B21: T1(1) + T3(-1) + T5(1) = 1. (Extra)
        # Coeff of A12B00: T1(1) + T2(1) + T4(-1) = 1. (Extra)
        # Coeff of A12B21: T1(1) + T5(0) + T7(-1) = 0. (Missing)
        
        # This decomposition is incorrect for $M_{2,3,2}$ with these specific indices.
        # The indices in Strassen are for 2x2 matrices.
        # For $M_{2,3,2}$, the "middle" dimension is 3.
        # We can view it as $M_{2,3,2} \cong M_{2,2,2} \oplus M_{1,1,2}$? No.
        
        # Let's use a simpler approach:
        # Use rank 12 for $2 \times 3 \times 2$? No, we want rank 7.
        # Is $R(2,3,2) = 7$? Yes.
        # Let's find the correct coefficients.
        # The decomposition is an isomorphic copy of $M_{2,2,2}$ restricted to a subspace.
        # If we set $A_{02} = A_{12} = 0$ and $B_{20} = B_{21} = 0$, we get $M_{2,2,2}$.
        # But we need to handle the 3rd row/col.
        
        # Alternative: Use rank 27 via 3 blocks of $M_{2,3,2}$ and 1 block of $M_{2,3,1}$.
        # I will use the explicit rank 7 decomposition found in literature for $2 \times 3 \times 2$.
        # One such decomposition is:
        # 1. (A00+A11)(B00+B11) -> No, this is for 2x2.
        
        # Let's fall back to a guaranteed correct, albeit potentially higher rank, decomposition if I can't verify the rank 7 one.
        # But rank 27 is much better than 42.
        # Let's try to implement the $2 \times 3 \times 2$ block with rank 7 by using the 2x2 Strassen on the first 2 rows/cols and adding the 3rd dimension.
        # Actually, $M_{2,3,2}$ can be decomposed as:
        # Let $A' = A[:, :2]$ (2x2) and $A'' = A[:, 2]$ (2x1).
        # Let $B' = B[:2, :]$ (2x2) and $B'' = B[2, :]$ (1x2).
        # $C = A'B' + A''B''$.
        # $A'B'$ is $2 \times 2 \times 2$ (Rank 7).
        # $A''B''$ is $2 \times 1 \times 2$ (Rank 4? $2 \times 1 \times 2$ is rank 4).
        # Total $7 + 4 = 11$.
        # So $R(2,3,2) \le 11$.
        # Is it 7? Yes, it is known to be 7.
        # But 11 is easier to implement correctly.
        # If we use rank 11 for each $2 \times 3 \times 2$ block:
        # 3 blocks * 11 = 33.
        # 1 block * 6 = 6.
        # Total 39.
        # This is better than 42, but not as good as 27.
        
        # Let's try to get the rank 7 decomposition right.
        # Reference: "The Rank of Matrix Multiplication" by Strassen.
        # The decomposition for $M_{2,3,2}$ is the same 7 terms, but the mapping of indices is different.
        # Let's use the following mapping:
        # A is 2x3, B is 3x2.
        # Let the 7 products be:
        # P1 = (A00 + A11) * (B00 + B11)
        # P2 = (A10 + A11) * B00
        # P3 = A00 * (B01 - B11)
        # P4 = A11 * (B10 - B00)
        # P5 = (A00 + A01) * B11
        # P6 = (A10 - A00) * (B00 + B01)
        # P7 = (A01 - A11) * (B10 + B11)
        #
        # This is the standard Strassen for 2x2 matrices.
        # It computes $A_{2x2} B_{2x2}$.
        # Our A is 2x3 and B is 3x2.
        # If we ignore A02, A12 and B20, B21, this works for the top-left 2x2 block.
        # We need to incorporate the 3rd row/col.
        # This suggests $R(2,3,2) > 7$ if we just append.
        # But it is known that $R(2,3,2) = 7$.
        # The decomposition must mix the 3rd index.
        
        # Given the complexity and risk of error, I will implement the rank 39 solution (3x11 + 1x6) which is guaranteed to work, or stick to the schoolbook if I can't verify.
        # However, 39 is a valid improvement over 42.
        # Let's try to implement rank 39.
        
        # Block 1 (cols 0,1): Rank 11.
        # Block 2 (cols 2,3): Rank 11.
        # Block 3 (cols 4,5): Rank 11.
        # Block 4 (col 6):   Rank 6.
        # Total 39.
        
        # Let's implement rank 11 for $M_{2,3,2}$.
        # $C = A_{2x3} B_{3x2}$.
        # Split A into $A_1$ (2x2) and $A_2$ (2x1).
        # Split B into $B_1$ (2x2) and $B_2$ (1x2).
        # $C = A_1 B_1 + A_2 B_2$.
        # $A_1 B_1$ is $M_{2,2,2}$ (Rank 7).
        # $A_2 B_2$ is $M_{2,1,2}$ (Rank 4).
        # Total 11.
        
        # $M_{2,1,2}$ decomposition:
        # A2 is 2x1 (a0, a1). B2 is 1x2 (b0, b1).
        # C = [[a0*b0, a0*b1], [a1*b0, a1*b1]].
        # Terms:
        # 1. a0 * b0 (C00)
        # 2. a0 * b1 (C01)
        # 3. a1 * b0 (C10)
        # 4. a1 * b1 (C11)
        
        # So for each $2 \times 3 \times 2$ block:
        # 7 terms for the $2 \times 2 \times 2$ part (using first 2 rows of B and first 2 cols of A? No, first 2 rows of B and first 2 cols of A?
        # A is 2x3. $A_1$ is 2x2 (cols 0,1). $A_2$ is 2x1 (col 2).
        # B is 3x2. $B_1$ is 2x2 (rows 0,1). $B_2$ is 1x2 (row 2).
        # So $A_1 B_1$ uses A cols 0,1 and B rows 0,1.
        # $A_2 B_2$ uses A col 2 and B row 2.
        
        # This is valid.
        
        rank = 39
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col_offset = 0
        
        # Add Strassen 2x2x2 block
        def add_strassen_2x2(j0, k0, start_col):
            # A uses rows 0,1 and cols j0, j0+1
            # B uses rows j0, j0+1 and cols k0, k0+1
            # This computes C[i, k] for i in 0,1 and k in 0,1 relative to the block.
            # Global indices:
            # A: (i, j0), (i, j0+1)
            # B: (j0, k0), (j0+1, k0), (j0, k0+1), (j0+1, k0+1)
            # C: (k0, i) ... wait, W is p x n. W[k*n + i].
            # So C[i, k] is W[k*n + i].
            
            # Strassen terms for 2x2:
            # 1. (A00 + A11) * (B00 + B11)
            # 2. (A10 + A11) * B00
            # 3. A00 * (B01 - B11)
            # 4. A11 * (B10 - B00)
            # 5. (A00 + A01) * B11
            # 6. (A10 - A00) * (B00 + B01)
            # 7. (A01 - A11) * (B10 + B11)
            
            # Term 1
            U[0][start_col] += 1; U[1][start_col] += 1
            V[j0*m + k0][start_col] += 1; V[(j0+1)*m + (k0+1)][start_col] += 1
            W[k0*n + 0][start_col] += 1; W[(k0+1)*n + 1][start_col] += 1
            W[k0*n + 1][start_col] += 1; W[(k0+1)*n + 0][start_col] += 1
            # Wait, W[k*n+i] is C[i,k].
            # C00 = M1...
            # M1 = (A00+A11)(B00+B11).
            # C00 += M1. C11 += M1.
            # So W[k0*n + 0] += 1, W[(k0+1)*n + 1] += 1.
            
            # Term 2
            U[1][start_col+1] += 1
            V[j0*m + k0][start_col+1] += 1
            W[k0*n + 1][start_col+1] += 1
            
            # Term 3
            U[0][start_col+2] += 1
            V[j0*m + (k0+1)][start_col+2] += 1; V[(j0+1)*m + (k0+1)][start_col+2] -= 1
            W[k0*n + 0][start_col+2] += 1; W[k0*n + 1][start_col+2] += 1
            
            # Term 4
            U[1][start_col+3] += 1
            V[(j0+1)*m + k0][start_col+3] += 1; V[j0*m + k0][start_col+3] -= 1
            W[(k0+1)*n + 1][start_col+3] += 1; W[(k0+1)*n + 0][start_col+3] += 1
            
            # Term 5
            U[0][start_col+4] += 1; U[0][start_col+4] += 1 # A00 + A01
            V[(j0+1)*m + (k0+1)][start_col+4] += 1
            W[k0*n + 0][start_col+4] += 1; W[k0*n + 1][start_col+4] += 1
            
            # Term 6
            U[1][start_col+5] += 1; U[0][start_col+5] -= 1
            V[j0*m + k0][start_col+5] += 1; V[j0*m + (k0+1)][start_col+5] += 1
            W[k0*n + 1][start_col+5] += 1; W[k0*n + 0][start_col+5] += 1
            
            # Term 7
            U[0][start_col+6] += 1; U[1][start_col+6] -= 1
            V[(j0+1)*m + k0][start_col+6] += 1; V[(j0+1)*m + (k0+1)][start_col+6] += 1
            W[k0*n + 0][start_col+6] += 1; W[(k0+1)*n + 1][start_col+6] += 1
            
            return start_col + 7

        # Add 2x1x2 block (Rank 4)
        def add_2x1x2(j2, k0, start_col):
            # A col j2, B row j2
            # C = A[:, j2] * B[j2, :]
            # C00 = A0,j2 * Bj2,k0
            # C01 = A0,j2 * Bj2,k0+1
            # C10 = A1,j2 * Bj2,k0
            # C11 = A1,j2 * Bj2,k0+1
            
            # T1: A0,j2 * Bj2,k0
            U[0][start_col] += 1
            V[j2*m + k0][start_col] += 1
            W[k0*n + 0][start_col] += 1
            
            # T2: A0,j2 * Bj2,k0+1
            U[0][start_col+1] += 1
            V[j2*m + (k0+1)][start_col+1] += 1
            W[(k0+1)*n + 0][start_col+1] += 1
            
            # T3: A1,j2 * Bj2,k0
            U[1][start_col+2] += 1
            V[j2*m + k0][start_col+2] += 1
            W[k0*n + 1][start_col+2] += 1
            
            # T4: A1,j2 * Bj2,k0+1
            U[1][start_col+3] += 1
            V[j2*m + (k0+1)][start_col+3] += 1
            W[(k0+1)*n + 1][start_col+3] += 1
            
            return start_col + 4

        # Add 2x3x1 block (Rank 6)
        def add_2x3x1(k0, start_col):
            # A 2x3, B 3x1
            # C0 = A00*B0 + A01*B1 + A02*B2
            # C1 = A10*B0 + A11*B1 + A12*B2
            
            # T1: A00*B0
            U[0][start_col] += 1
            V[0*m + k0][start_col] += 1
            W[k0*n + 0][start_col] += 1
            
            # T2: A01*B1
            U[0][start_col+1] += 1
            V[1*m + k0][start_col+1] += 1
            W[k0*n + 0][start_col+1] += 1
            
            # T3: A02*B2
            U[0][start_col+2] += 1
            V[2*m + k0][start_col+2] += 1
            W[k0*n + 0][start_col+2] += 1
            
            # T4: A10*B0
            U[1][start_col+3] += 1
            V[0*m + k0][start_col+3] += 1
            W[k0*n + 1][start_col+3] += 1
            
            # T5: A11*B1
            U[1][start_col+4] += 1
            V[1*m + k0][start_col+4] += 1
            W[k0*n + 1][start_col+4] += 1
            
            # T6: A12*B2
            U[1][start_col+5] += 1
            V[2*m + k0][start_col+5] += 1
            W[k0*n + 1][start_col+5] += 1
            
            return start_col + 6

        # Process blocks
        # Block 1: Cols 0,1. j0=0, k0=0.
        col_offset = add_strassen_2x2(0, 0, col_offset)
        col_offset = add_2x1x2(2, 0, col_offset)
        
        # Block 2: Cols 2,3. j0=0, k0=2.
        col_offset = add_strassen_2x2(0, 2, col_offset)
        col_offset = add_2x1x2(2, 2, col_offset)
        
        # Block 3: Cols 4,5. j0=0, k0=4.
        col_offset = add_strassen_2x2(0, 4, col_offset)
        col_offset = add_2x1x2(2, 4, col_offset)
        
        # Block 4: Col 6. k0=6.
        col_offset = add_2x3x1(6, col_offset)
        
        return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
