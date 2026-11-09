import time

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    # Strassen decomposition of 4x4 * 4x4 -> 4x4
    # This gives rank 49 (7 outer steps * 7 inner Strassen multiplies)
    
    n, m, p = DIMENSIONS
    # For 4x4 matrix multiplication
    # We partition into 2x2 blocks of 2x2 matrices
    # Apply Strassen at the block level (7 multiplications of 2x2 blocks)
    # Then apply Strassen within each 2x2 multiplication (7 scalar multiplications each)
    # Total: 7 * 7 = 49
    
    rank = 49
    
    # U has shape (n*m, rank) = (16, 49)
    # V has shape (m*p, rank) = (16, 49)
    # W has shape (p*n, rank) = (16, 49)
    # W[k*n+i, r] convention
    
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(16)]
    W = [[0] * rank for _ in range(16)]
    
    # Strassen 4x4 via 2x2 blocks
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # where each block is 2x2
    
    # Strassen 2x2 block multiplication:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For each block multiplication Ms, we need to multiply two 2x2 matrices
    # Each 2x2 matrix multiply uses 7 scalar multiplications (Strassen)
    
    # Let's index the 4x4 matrices:
    # A[i,j] for i,j in 0..3
    # A00 = A[0:2, 0:2], A01 = A[0:2, 2:4], A10 = A[2:4, 0:2], A11 = A[2:4, 2:4]
    
    # B[k,l] for k,l in 0..3
    # B00 = B[0:2, 0:2], B01 = B[0:2, 2:4], B10 = B[2:4, 0:2], B11 = B[2:4, 2:4]
    
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # C00 = C[0:2, 0:2], C01 = C[0:2, 2:4], C10 = C[2:4, 0:2], C11 = C[2:4, 2:4]
    
    # For each of the 7 Strassen products M1..M7:
    # Each Ms is a 2x2 matrix = Ls * Rs where Ls and Rs are 2x2 matrices
    # The product Ls * Rs requires 7 scalar multiplications via Strassen
    
    # Strassen for 2x2:
    # Given X, Y (2x2), compute Z = X*Y
    # N1 = (X00 + X11)(Y00 + Y11)
    # N2 = (X10 + X11)Y00
    # N3 = X00(Y01 - Y11)
    # N4 = X11(Y10 - Y00)
    # N5 = (X00 + X01)Y11
    # N6 = (X10 - X00)(Y00 + Y01)
    # N7 = (X01 - X11)(Y10 + Y11)
    # Z00 = N1 + N4 - N5 + N7
    # Z01 = N3 + N5
    # Z10 = N2 + N4
    # Z11 = N1 - N2 + N3 + N6
    
    # For each Ms (s in 0..6), we have:
    # Ms = Ls * Rs
    # where Ls is a linear combination of A-blocks, Rs is a linear combination of B-blocks
    
    # The entries of Ls are linear combinations of A entries
    # The entries of Rs are linear combinations of B entries
    
    # For the inner Strassen, each scalar multiply is of form:
    # (linear combo of A entries) * (linear combo of B entries)
    # This contributes one rank-1 term
    
    # Let me define the Strassen outer coefficients:
    # For s=0 (M1): L1 = A00 + A11, R1 = B00 + B11
    # For s=1 (M2): L2 = A10 + A11, R2 = B00
    # For s=2 (M3): L3 = A00, R3 = B01 - B11
    # For s=3 (M4): L4 = A11, R4 = B10 - B00
    # For s=4 (M5): L5 = A00 + A01, R5 = B11
    # For s=5 (M6): L6 = A10 - A00, R6 = B00 + B01
    # For s=6 (M7): L7 = A01 - A11, R7 = B10 + B11
    
    # Inner Strassen for each Ms = Ls * Rs:
    # t=0: (L00+L11)(R00+R11)
    # t=1: (L10+L11)R00
    # t=2: L00(R01-R11)
    # t=3: L11(R10-R00)
    # t=4: (L00+L01)R11
    # t=5: (L10-L00)(R00+R01)
    # t=6: (L01-L11)(R10+R11)
    
    # Result for Ms:
    # Ms[0,0] = N1+N4-N5+N7
    # Ms[0,1] = N3+N5
    # Ms[1,0] = N2+N4
    # Ms[1,1] = N1-N2+N3+N6
    
    # C blocks:
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # So C00[0,0] = M1[0,0]+M4[0,0]-M5[0,0]+M7[0,0]
    # etc.
    
    # Let's build the U, V, W matrices
    
    # First, define which A entries go into each Ls block
    # A00 = [[A[0,0], A[0,1]], [A[1,0], A[1,1]]]
    # A01 = [[A[0,2], A[0,3]], [A[1,2], A[1,3]]]
    # A10 = [[A[2,0], A[2,1]], [A[3,0], A[3,1]]]
    # A11 = [[A[2,2], A[2,3]], [A[3,2], A[3,3]]]
    
    # For each Ls, it's a 2x2 matrix whose entries are linear combos of A entries
    # Ls[0,0], Ls[0,1], Ls[1,0], Ls[1,1]
    
    # Define Ls coefficients: for each s, for each position (a,b) in 2x2,
    # Ls[a,b] = sum over A entries with coefficients
    
    # Let me represent A entry (i,j) as index i*4+j in a flat vector of length 16
    # Similarly B entry (k,l) as index k*4+l
    
    # U[i*4+j, r] = coefficient of A[i,j] in the r-th multiply
    # V[k*4+l, r] = coefficient of B[k,l] in the r-th multiply
    # W[l*4+i, r] = coefficient of C[i,l] from the r-th multiply
    # Note: W[k*n+i, r] where k is the B row index and i is the A row index
    # C[i,l] = sum_j A[i,j]B[j,l], so W[j*4+i, r] for the term A[i,j]*B[j,l]
    # Wait, let me re-read: "W[k*n+i,r]" - k is from B (column of A / row of B), i is from A (row of A / row of C)
    # So W[j*4+i, r] corresponds to A[i,j]*B[j,l] contributing to C[i,l]
    # Actually C[i,l] = sum_j A[i,j] * B[j,l]
    # The tensor is T[i,j,k,l] = delta... no, T_{i,j,k,l} = 1 if we're multiplying A[i,j]*B[j,k] to get C[i,k]
    # So the term is A[i,j]*B[j,k] -> C[i,k]
    # U[i*4+j, r], V[j*4+k, r], W[k*4+i, r]
    # Yes: W[k*n+i, r] where n=4, so W[k*4+i, r]
    
    # For each (s, t) pair, we have one multiply:
    # U_part = linear combo of A entries (from Ls entries used in inner Strassen step t)
    # V_part = linear combo of B entries (from Rs entries used in inner Strassen step t)
    # W_part = linear combo of C entries (from Ms[a,b] where Ms appears in C blocks)
    
    # Let me define the contributions more carefully
    
    # For inner Strassen step t of product Ms = Ls * Rs:
    # The scalar multiply is:
    # t=0: X = Ls[0,0]+Ls[1,1], Y = Rs[0,0]+Rs[1,1]
    # t=1: X = Ls[1,0]+Ls[1,1], Y = Rs[0,0]
    # t=2: X = Ls[0,0], Y = Rs[0,1]-Rs[1,1]
    # t=3: X = Ls[1,1], Y = Rs[1,0]-Rs[0,0]
    # t=4: X = Ls[0,0]+Ls[0,1], Y = Rs[1,1]
    # t=5: X = Ls[1,0]-Ls[0,0], Y = Rs[0,0]+Rs[0,1]
    # t=6: X = Ls[0,1]-Ls[1,1], Y = Rs[1,0]+Rs[1,1]
    
    # Where Ls[a,b] is entry (a,b) of the 2x2 matrix Ls
    # Ls[a,b] is a linear combo of A entries
    # Rs[a,b] is a linear combo of B entries
    
    # For outer Strassen step s, Ls is:
    # s=0: Ls = A00+A11 -> Ls[a,b] = A00[a,b]+A11[a,b]
    # s=1: Ls = A10+A11 -> Ls[a,b] = A10[a,b]+A11[a,b]
    # s=2: Ls = A00 -> Ls[a,b] = A00[a,b]
    # s=3: Ls = A11 -> Ls[a,b] = A11[a,b]
    # s=4: Ls = A00+A01 -> Ls[a,b] = A00[a,b]+A01[a,b]
    # s=5: Ls = A10-A00 -> Ls[a,b] = A10[a,b]-A00[a,b]
    # s=6: Ls = A01-A11 -> Ls[a,b] = A01[a,b]-A11[a,b]
    
    # Similarly for Rs:
    # s=0: Rs = B00+B11
    # s=1: Rs = B00
    # s=2: Rs = B01-B11
    # s=3: Rs = B10-B00
    # s=4: Rs = B11
    # s=5: Rs = B00+B01
    # s=6: Rs = B10+B11
    
    # A00[a,b] = A[a, b] where a,b in {0,1}
    # A01[a,b] = A[a, b+2] where a,b in {0,1}
    # A10[a,b] = A[a+2, b] where a,b in {0,1}
    # A11[a,b] = A[a+2, b+2] where a,b in {0,1}
    
    # B00[a,b] = B[a, b] where a,b in {0,1}
    # B01[a,b] = B[a, b+2] where a,b in {0,1}
    # B10[a,b] = B[a+2, b] where a,b in {0,1}
    # B11[a,b] = B[a+2, b+2] where a,b in {0,1}
    
    # For the W part:
    # Ms[a,b] contributes to C blocks:
    # C00 = M0+M3-M4+M6 (using 0-indexed: M1=index0, M2=index1, etc.)
    # Wait, let me use 0-indexed: s=0..6 for M1..M7
    # C00 = M0+M3-M4+M6
    # C01 = M2+M4
    # C10 = M1+M3
    # C11 = M0-M1+M2+M5
    
    # So for Ms (s in 0..6), Ms[a,b] contributes to:
    # If s=0: C00[a,b] += Ms[a,b], C11[a,b] += Ms[a,b]
    # If s=1: C10[a,b] += Ms[a,b], C11[a,b] -= Ms[a,b]
    # If s=2: C01[a,b] += Ms[a,b], C11[a,b] += Ms[a,b]
    # If s=3: C00[a,b] += Ms[a,b], C10[a,b] += Ms[a,b]
    # If s=4: C00[a,b] -= Ms[a,b], C01[a,b] += Ms[a,b]
    # If s=5: C11[a,b] += Ms[a,b]
    # If s=6: C00[a,b] += Ms[a,b]
    
    # C00[a,b] = C[a, b] where a,b in {0,1}
    # C01[a,b] = C[a, b+2] where a,b in {0,1}
    # C10[a,b] = C[a+2, b] where a,b in {0,1}
    # C11[a,b] = C[a+2, b+2] where a,b in {0,1}
    
    # Now, for inner Strassen step t, the scalar multiply Nt contributes to Ms:
    # t=0 (N1): Ms[0,0] += N1, Ms[1,1] += N1
    # t=1 (N2): Ms[1,0] += N2, Ms[1,1] -= N2
    # t=2 (N3): Ms[0,1] += N3, Ms[1,1] += N3
    # t=3 (N4): Ms[0,0] += N4, Ms[1,0] += N4
    # t=4 (N5): Ms[0,0] -= N5, Ms[0,1] += N5
    # t=5 (N6): Ms[1,1] += N6
    # t=6 (N7): Ms[0,0] += N7
    
    # So the W contribution for (s,t) is the sum of contributions from Nt to Ms[a,b]
    # which then propagates to C blocks
    
    # Let me just compute this directly
    
    def get_A_block_entry(block, a, b):
        """Return (i, j) for A[block][a,b]"""
        if block == 0:  # A00
            return (a, b)
        elif block == 1:  # A01
            return (a, b + 2)
        elif block == 2:  # A10
            return (a + 2, b)
        elif block == 3:  # A11
            return (a + 2, b + 2)
    
    def get_B_block_entry(block, a, b):
        """Return (k, l) for B[block][a,b]"""
        if block == 0:  # B00
            return (a, b)
        elif block == 1:  # B01
            return (a, b + 2)
        elif block == 2:  # B10
            return (a + 2, b)
        elif block == 3:  # B11
            return (a + 2, b + 2)
    
    def get_C_block_entry(block, a, b):
        """Return (i, l) for C[block][a,b]"""
        if block == 0:  # C00
            return (a, b)
        elif block == 1:  # C01
            return (a, b + 2)
        elif block == 2:  # C10
            return (a + 2, b)
        elif block == 3:  # C11
            return (a + 2, b + 2)
    
    # Define Ls block compositions: for each s, list of (block, coefficient)
    Ls_composition = {
        0: [(0, 1), (3, 1)],   # A00 + A11
        1: [(2, 1), (3, 1)],   # A10 + A11
        2: [(0, 1)],           # A00
        3: [(3, 1)],           # A11
        4: [(0, 1), (1, 1)],   # A00 + A01
        5: [(2, 1), (0, -1)],  # A10 - A00
        6: [(1, 1), (3, -1)],  # A01 - A11
    }
    
    # Define Rs block compositions
    Rs_composition = {
        0: [(0, 1), (3, 1)],   # B00 + B11
        1: [(0, 1)],           # B00
        2: [(1, 1), (3, -1)],  # B01 - B11
        3: [(2, 1), (0, -1)],  # B10 - B00
        4: [(3, 1)],           # B11
        5: [(0, 1), (1, 1)],   # B00 + B01
        6: [(2, 1), (3, 1)],   # B10 + B11
    }
    
    # For inner Strassen, define which Ls entries are combined
    # Inner Strassen X compositions: for each t, list of (a, b, coeff) for Ls[a,b]
    inner_X = {
        0: [(0, 0, 1), (1, 1, 1)],           # L00 + L11
        1: [(1, 0, 1), (1, 1, 1)],           # L10 + L11
        2: [(0, 0, 1)],                       # L00
        3: [(1, 1, 1)],                       # L11
        4: [(0, 0, 1), (0, 1, 1)],           # L00 + L01
        5: [(1, 0, 1), (0, 0, -1)],          # L10 - L00
        6: [(0, 1, 1), (1, 1, -1)],          # L01 - L11
    }
    
    # Inner Strassen Y compositions: for each t, list of (a, b, coeff) for Rs[a,b]
    inner_Y = {
        0: [(0, 0, 1), (1, 1, 1)],           # R00 + R11
        1: [(0, 0, 1)],                       # R00
        2: [(0, 1, 1), (1, 1, -1)],          # R01 - R11
        3: [(1, 0, 1), (0, 0, -1)],          # R10 - R00
        4: [(1, 1, 1)],                       # R11
        5: [(0, 0, 1), (0, 1, 1)],           # R00 + R01
        6: [(1, 0, 1), (1, 1, 1)],           # R10 + R11
    }
    
    # For inner Strassen, define which Ms entries Nt contributes to
    # inner_W[t] = list of (a, b, coeff) for Ms[a,b]
    inner_W = {
        0: [(0, 0, 1), (1, 1, 1)],           # N1: Ms[0,0] += N1, Ms[1,1] += N1
        1: [(1, 0, 1), (1, 1, -1)],          # N2: Ms[1,0] += N2, Ms[1,1] -= N2
        2: [(0, 1, 1), (1, 1, 1)],           # N3: Ms[0,1] += N3, Ms[1,1] += N3
        3: [(0, 0, 1), (1, 0, 1)],           # N4: Ms[0,0] += N4, Ms[1,0] += N4
        4: [(0, 0, -1), (0, 1, 1)],          # N5: Ms[0,0] -= N5, Ms[0,1] += N5
        5: [(1, 1, 1)],                       # N6: Ms[1,1] += N6
        6: [(0, 0, 1)],                       # N7: Ms[0,0] += N7
    }
    
    # For outer Strassen, define which C blocks Ms contributes to
    # outer_W[s] = list of (c_block, a, b, coeff)
    outer_W = {
        0: [(0, None, None, 1), (3, None, None, 1)],   # C00 += M0, C11 += M0 (all entries)
        1: [(2, None, None, 1), (3, None, None, -1)],  # C10 += M1, C11 -= M1
        2: [(1, None, None, 1), (3, None, None, 1)],   # C01 += M2, C11 += M2
        3: [(0, None, None, 1), (2, None, None, 1)],   # C00 += M3, C10 += M3
        4: [(0, None, None, -1), (1, None, None, 1)],  # C00 -= M4, C01 += M4
        5: [(3, None, None, 1)],                        # C11 += M5
        6: [(0, None, None, 1)],                        # C00 += M6
    }
    
    # Now build U, V, W
    col = 0
    for s in range(7):
        for t in range(7):
            # Compute U part: linear combo of A entries
            # For each (a, b, coeff) in inner_X[t]:
            #   Ls[a,b] = sum over (block, bcoeff) in Ls_composition[s] of bcoeff * A[block][a,b]
            #   So A entry is at get_A_block_entry(block, a, b)
            for (a, b, xcoeff) in inner_X[t]:
                for (block, bcoeff) in Ls_composition[s]:
                    i, j = get_A_block_entry(block, a, b)
                    U[i * 4 + j][col] += xcoeff * bcoeff
            
            # Compute V part
            for (a, b, ycoeff) in inner_Y[t]:
                for (block, bcoeff) in Rs_composition[s]:
                    k, l = get_B_block_entry(block, a, b)
                    V[k * 4 + l][col] += ycoeff * bcoeff
            
            # Compute W part
            # For each (ms_a, ms_b, ncoeff) in inner_W[t]:
            #   Ms[ms_a, ms_b] gets ncoeff * Nt
            #   Ms[ms_a, ms_b] contributes to C blocks per outer_W[s]
            for (ms_a, ms_b, ncoeff) in inner_W[t]:
                for (c_block, _, _, outer_coeff) in outer_W[s]:
                    i, l = get_C_block_entry(c_block, ms_a, ms_b)
                    # W[k*4+i, r] where the term is A[i,j]*B[j,l] -> C[i,l]
                    # But here we just need the C entry index
                    # W is indexed by [k*n + i, r] where k is the B row (j in A[i,j]*B[j,k])
                    # and i is the A row
                    # Wait, the C entry is C[i, l] where i is row, l is column
                    # The tensor element is T[i, j, k, l] = delta_{j,k}... no
                    # T[i,j,k,l] contributes A[i,j]*B[j,k] to C[i,k]
                    # So for C[i,l], j can be anything, and k=l
                    # W[j*4+i, r] for each j
                    # But in our decomposition, the term is a product of a linear combo of A entries
                    # and a linear combo of B entries, and it contributes to a specific C[i,l]
                    # The W matrix should have W[j*4+i, r] = coefficient for the (i,j) row of A and (j,k) col of B
                    # Actually, I think W is just indexed by the C output position
                    # W[k*n+i, r] = coefficient for C[i,k] from the r-th term
                    # So W[l*4+i, r] where C[i,l] is the output
                    W[l * 4 + i][col] += ncoeff * outer_coeff
            
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
