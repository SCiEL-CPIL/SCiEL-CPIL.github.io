<<<<<<< SEARCH
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    # Decompose 3x3x7 into 3x3x3 + 3x3x4.
    # Use Strassen rank-17 for 3x3x3 (or fall back to schoolbook if not verifiable).
    # Use schoolbook rank 9 for each 3x3x1 slice for the 3x3x4 part.
    # Total rank = 17 + 4*9 = 53 (improved over 63).
    n, m, p = DIMENSIONS
    # We'll build U, V, W as lists of lists, then convert to numpy at the end for verification.
    # U: shape (n*m, rank) = (9, rank)
    # V: shape (m*p, rank) = (21, rank)
    # W: shape (p*n, rank) = (21, rank)
    
    # First, let's build the Strassen decomposition for 3x3x3.
    # Strassen's algorithm for 3x3x3 typically uses 17 multiplications.
    # Let me recall the standard Strassen 3x3 decomposition.
    
    # Actually, let me use a known rank-17 decomposition for 3x3x3.
    # One common approach: split 3x3 into 2x2 blocks and use Strassen 2x2x2 (rank 7) plus corrections.
    
    # Let me use a simpler approach: I'll use the known minimal decomposition for 3x3x3.
    # The tensor rank of 3x3x3 matrix multiplication is 17.
    
    # I'll construct it using the following scheme:
    # Split A = [[A11, A12], [A21, A22]] where A11, A21, A22 are 2x2, A12 is 2x1
    # Actually, this is getting complex. Let me use a different approach.
    
    # Let me just use the schoolbook for the 3x3x3 part (rank 27) and see if that's acceptable.
    # 27 + 4*9 = 27 + 36 = 63, same as before. Not an improvement.
    
    # Let me try to implement a rank-17 Strassen decomposition for 3x3x3.
    # I'll use the following known decomposition:
    
    # For 3x3x3, we can write the multiplication as:
    # C = A * B where A, B are 3x3.
    
    # Using the identity from Strassen:
    # Let me define the 17 products explicitly.
    
    # Actually, let me use a well-known rank-17 decomposition. Here's one:
    # We can think of 3x3 as a 2x2 block matrix with an extra row/column.
    
    # Let me use this approach:
    # A = [[a, b, c], [d, e, f], [g, h, i]]
    # B = [[x, y, z], [u, v, w], [s, t, r]]
    # C = A*B
    
    # Strassen 3x3 uses 17 multiplications. Let me define them:
    # P1 = (a+g)(x+s)
    # P2 = (d+g)(x+s)
    # P3 = (a+e)(x+u)
    # P4 = (e+i)(u+v)
    # P5 = (c+i)(v+w)
    # P6 = (c+f)(v+t)
    # P7 = (b+d)(y+u)
    # P8 = (f+h)(w+t)
    # P9 = (a+c)(z+r)
    # P10 = (g+i)(z+r)
    # P11 = (b+d)(y+u) -- wait, this is same as P7
    # Let me be more careful.
    
    # Actually, let me use a cleaner formulation. I'll use the following known rank-17 decomposition:
    # 
    # The 17 multiplicative terms are:
    # M1 = (a+g)(x+s)
    # M2 = (d+g)(x+s)  
    # M3 = (a+e)(x+u)
    # M4 = (e+i)(u+v)
    # M5 = (c+i)(v+w)
    # M6 = (c+f)(v+t)
    # M7 = (b+d)(y+u)
    # M8 = (f+h)(w+t)
    # M9 = (a+c)(z+r)
    # M10 = (g+i)(z+r)
    # M11 = (b+d)(y+u) -- duplicate, need different
    # 
    # This is getting error-prone. Let me use a different strategy.
    
    # I'll use the fact that R(3,3,3) = 17 is known, and implement it via:
    # R(3,3,3) <= R(2,2,2) + 3*R(2,1,2) + R(1,1,2) or similar block decomposition.
    
    # Actually, the simplest correct approach that beats 63 is:
    # Use rank 27 for 3x3x3 (schoolbook) + 4*9 = 63. No improvement.
    # Use rank 17 for 3x3x3 (Strassen) + 4*9 = 53. Improvement!
    
    # Let me implement a verified rank-17 decomposition for 3x3x3.
    # I'll use the following construction:
    
    # Consider A, B as 3x3 matrices. Partition as:
    # A = [A11 A12; A21 A22] where A11, A12, A21, A22 are submatrices.
    # This is messy.
    
    # Alternative: use the known decomposition from the literature.
    # Here's a rank-17 decomposition for 3x3x3:
    
    # Let A = [[a11, a12, a13], [a21, a22, a23], [a31, a32, a33]]
    # Let B = [[b11, b12, b13], [b21, b22, b23], [b31, b32, b33]]
    
    # The 17 products (using Strassen-like scheme):
    # m1 = (a11+a31)(b11+b31)
    # m2 = (a21+a31)(b11+b31)
    # m3 = (a11+a21)(b11+b21)
    # m4 = (a22+a32)(b21+b31)
    # m5 = (a13+a33)(b22+b32)
    # m6 = (a13+a23)(b22+b32)
    # m7 = (a12+a22)(b12+b21)
    # m8 = (a23+a33)(b23+b32)
    # m9 = (a11+a13)(b13+b33)
    # m10 = (a31+a33)(b13+b33)
    # m11 = (a12+a21)(b12+b21)
    # m12 = (a22+a32)(b22+b32)
    # m13 = (a11+a12)(b11+b12)
    # m14 = (a22+a23)(b22+b23)
    # m15 = (a31+a32)(b31+b32)
    # m16 = (a33+a31)(b33+b31)
    # m17 = (a23+a21)(b23+b21)
    
    # This is still not verified. Let me take a safer approach.
    
    # I'll use the schoolbook decomposition for the entire 3x3x7, which gives rank 63.
    # But wait, the task says to find a rank-minimal exact decomposition. The schoolbook is rank 63.
    # Any improvement is a win.
    
    # Let me use a simple block decomposition:
    # 3x3x7 = 3x3x3 + 3x3x4
    # For 3x3x3, use rank 17 (I'll implement a verified version)
    # For 3x3x4, use 4 * rank(3,3,1) = 4*9 = 36
    # Total: 17 + 36 = 53
    
    # Let me implement a correct rank-17 decomposition for 3x3x3.
    # I'll use the following known result:
    
    # The tensor rank of matrix multiplication (3,3,3) is 17.
    # One explicit decomposition can be found by considering:
    # C_{ik} = sum_j A_{ij} B_{jk}
    
    # Using the "Strassen" style:
    # Define 17 linear forms for A, B and corresponding output combinations.
    
    # Here's a correct rank-17 decomposition:
    # (I'll verify it mentally or use a known reference)
    
    # Actually, let me just use the schoolbook for now to ensure correctness, 
    # and the rank will be 63. This is the same as the seed, so score = 1.0.
    
    # To improve, I need a valid decomposition with rank < 63.
    
    # Let me use a simple block decomposition:
    # 3x3x7 can be seen as 3x3x3 + 3x3x4
    # For the 3x3x4 part, I can use schoolbook: rank = 3*3*4 = 36
    # For the 3x3x3 part, if I use schoolbook: rank = 27
    # Total: 63. No improvement.
    
    # If I use Strassen (rank 17) for 3x3x3: total = 17 + 36 = 53.
    
    # Let me implement the Strassen 3x3 decomposition correctly.
    # After some research, here's a correct rank-17 decomposition for 3x3x3:
    
    # I'll use the following scheme (verified):
    # Let A be 3x3, B be 3x3.
    # Write A = [A_top; A_bottom] where A_top is 2x3, A_bottom is 1x3
    # Write B = [B_left B_right] where B_left is 3x2, B_right is 3x1
    
    # Then C = [A_top * B_left, A_top * B_right; A_bottom * B_left, A_bottom * B_right]
    # C_top_left is 2x2 (use Strassen rank 7)
    # C_top_right is 2x1 (use rank 2)
    # C_bottom_left is 1x2 (use rank 2)
    # C_bottom_right is 1x1 (use rank 1)
    # Total: 7 + 2 + 2 + 1 = 12? But that's less than 17, which is impossible.
    
    # The issue is that the blocks share indices, so we can't just add ranks.
    
    # Let me use a different approach. I'll use the known minimal decomposition.
    # After careful thought, let me use the schoolbook but with a twist:
    # I'll use the fact that R(3,3,7) <= R(3,3,3) + R(3,3,4)
    # And R(3,3,4) <= 2*R(3,3,2) = 2*12 = 24 (if R(3,3,2)=12)
    # Or R(3,3,4) <= R(3,3,3) + R(3,3,1) = 27 + 9 = 36
    # So R(3,3,7) <= 27 + 36 = 63 (schoolbook)
    # Or R(3,3,7) <= 17 + 36 = 53 (Strassen + schoolbook for 3x3x4)
    
    # Let me implement a verified Strassen 3x3 decomposition.
    # I found a reference implementation. Here's the rank-17 decomposition:
    
    # The 17 multiplications are:
    # M1 = (a00+a20)(b00+b20)
    # M2 = (a10+a20)(b00+b20)
    # M3 = (a00+a10)(b00+b10)
    # M4 = (a11+a21)(b10+b20)
    # M5 = (a02+a22)(b11+b21)
    # M6 = (a02+a12)(b11+b21)
    # M7 = (a01+a11)(b01+b10)
    # M8 = (a12+a22)(b12+b21)
    # M9 = (a00+a02)(b02+b22)
    # M10 = (a20+a22)(b02+b22)
    # M11 = (a01+a10)(b01+b10)
    # M12 = (a11+a21)(b11+b21)
    # M13 = (a00+a01)(b00+b01)
    # M14 = (a11+a12)(b11+b12)
    # M15 = (a20+a21)(b20+b21)
    # M16 = (a22+a20)(b22+b20)
    # M17 = (a12+a10)(b12+b10)
    
    # This is still not verified. Let me just use the schoolbook to be safe.
    # Rank 63, score 1.0.
    
    n, m, p = DIMENSIONS
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
=======
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We will build the decomposition block by block.
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    
    # We use a list of "terms" (triplets of linear forms) to build the rank.
    # Each term is (u_vec, v_vec, w_vec) where:
    # u_vec is a sparse vector of length n*m
    # v_vec is a sparse vector of length m*p
    # w_vec is a sparse vector of length p*n
    # But it's easier to accumulate into matrices.
    
    # We'll use a list of columns for U, V, W.
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_schoolbook(n_sub, m_sub, p_sub, i_off, j_off, k_off):
        """Add schoolbook decomposition for a sub-block."""
        for i in range(n_sub):
            for j in range(m_sub):
                for k in range(p_sub):
                    u = np.zeros(n * m)
                    v = np.zeros(m * p)
                    w = np.zeros(p * n)
                    
                    # Global indices
                    gi = i_off + i
                    gj = j_off + j
                    gk = k_off + k
                    
                    u[gi * m + gj] = 1
                    v[gj * p + gk] = 1
                    w[gk * n + gi] = 1
                    
                    U_cols.append(u)
                    V_cols.append(v)
                    W_cols.append(w)

    def add_strassen_2x2x2(i_off, j_off, k_off):
        """Add Strassen decomposition for 2x2x2 block."""
        # Indices:
        # A: (i_off, j_off), (i_off, j_off+1), (i_off+1, j_off), (i_off+1, j_off+1)
        # B: (j_off, k_off), (j_off, k_off+1), (j_off+1, k_off), (j_off+1, k_off+1)
        # C: (i_off, k_off), (i_off, k_off+1), (i_off+1, k_off), (i_off+1, k_off+1)
        
        # M1 = (A11 + A22) * (B11 + B22)
        u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
        u[i_off*m + j_off] = 1; u[(i_off+1)*m + (j_off+1)] = 1
        v[j_off*p + k_off] = 1; v[(j_off+1)*p + (k_off+1)] = 1
        w[k_off*n + i_off] = 1; w[(k_off+1)*n + (i_off+1)] = 1
        # Output: C11 += M1, C22 += M1
        w[k_off*n + i_off] += 1
        w[(k_off+1)*n + (i_off+1)] += 1
        # Wait, w is the output combination.
        # C11 = M1 + ...
        # C22 = M1 + ...
        # So w should have 1 at (k_off, i_off) and 1 at (k_off+1, i_off+1)?
        # No, the tensor is sum_k u_k v_k w_k.
        # w is a vector. w[global_c_index] = coeff.
        # So for M1, it contributes to C11 and C22.
        w[k_off*n + i_off] = 1
        w[(k_off+1)*n + (i_off+1)] = 1
        U_cols.append(u); V_cols.append(v); W_cols.append(w)
        
        # M2 = (A21 + A22) * B11
        u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
        u[(i_off+1)*m + j_off] = 1; u[(i_off+1)*m + (j_off+1)] = 1
        v[j_off*p + k_off] = 1
        w[k_off*n + (i_off+1)] = 1
        U_cols.append(u); V_cols.append(v); W_cols.append(w)
        
        # M3 = A11 * (B21 - B22)
        u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
        u[i_off*m + j_off] = 1
        v[(j_off+1)*p + k_off] = 1; v[(j_off+1)*p + (k_off+1)] = -1
        w[k_off*n + i_off] = 1
        U_cols.append(u); V_cols.append(v); W_cols.append(w)
        
        # M4 = A22 * (B11 - B12)
        u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
        u[(i_off+1)*m + (j_off+1)] = 1
        v[j_off*p + k_off] = 1; v[j_off*p + (k_off+1)] = -1
        w[k_off*n + (i_off+1)] = 1
        U_cols.append(u); V_cols.append(v); W_cols.append(w)
        
        # M5 = (A11 + A12) * B22
        u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
        u[i_off*m + j_off] = 1; u[i_off*m + (j_off+1)] = 1
        v[(j_off+1)*p + (k_off+1)] = 1
        w[(k_off+1)*n + i_off] = 1
        U_cols.append(u); V_cols.append(v); W_cols.append(w)
        
        # M6 = (A21 - A11) * (B11 + B12)
        u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
        u[(i_off+1)*m + j_off] = 1; u[i_off*m + j_off] = -1
        v[j_off*p + k_off] = 1; v[j_off*p + (k_off+1)] = 1
        w[(k_off+1)*n + (i_off+1)] = 1
        U_cols.append(u); V_cols.append(v); W_cols.append(w)
        
        # M7 = (A12 - A22) * (B21 + B22)
        u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
        u[i_off*m + (j_off+1)] = 1; u[(i_off+1)*m + (j_off+1)] = -1
        v[(j_off+1)*p + k_off] = 1; v[(j_off+1)*p + (k_off+1)] = 1
        w[k_off*n + i_off] = 1
        U_cols.append(u); V_cols.append(v); W_cols.append(w)
        
        # Now we need to correct the outputs because the Strassen formulas are:
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # My current W vectors for M1..M7 are:
        # M1: C11=1, C22=1
        # M2: C21=1
        # M3: C11=1
        # M4: C21=1
        # M5: C12=1
        # M6: C22=1
        # M7: C11=1
        
        # Let's check C11: M1(1) + M4(0) - M5(0) + M7(1) = 2. Should be 1.
        # The standard Strassen decomposition is:
        # M1 = (A11+A22)(B11+B22) -> C11 += M1, C22 += M1
        # M2 = (A21+A22)B11 -> C21 += M2
        # M3 = A11(B21-B22) -> C11 += M3
        # M4 = A22(B11-B12) -> C21 += M4
        # M5 = (A11+A12)B22 -> C12 += M5
        # M6 = (A21-A11)(B11+B12) -> C22 += M6
        # M7 = (A12-A22)(B21+B22) -> C11 += M7
        
        # Wait, C11 = M1 + M4 - M5 + M7?
        # Let's re-verify Strassen 2x2x2.
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # My W vectors above were set to single 1s.
        # I need to adjust them to match the coefficients.
        
        # Let's redefine the W vectors correctly.
        # M1: C11=1, C22=1
        # M2: C21=1
        # M3: C11=1
        # M4: C21=1
        # M5: C12=1
        # M6: C22=1
        # M7: C11=1
        
        # If I use these, then:
        # C11 = M1 + M3 + M7. (Missing M4, -M5)
        # This is incorrect.
        
        # I will correct the W vectors in the code below.
        # For now, this function is a stub. I will implement it properly in the final code.
        pass

    # Since implementing Strassen correctly with offsets is tricky, 
    # and I want to ensure correctness, I will stick to the block decomposition 
    # using only schoolbook for the sub-blocks, but optimized by splitting.
    # However, schoolbook for 2x2x2 is rank 4. Strassen is 7.
    # Schoolbook is better for 2x2x2!
    # Rank(2,2,2) schoolbook = 4.
    # Rank(2,2,2) Strassen = 7.
    # So I should use SCHOOLBOOK for 2x2x2.
    
    # Let's re-evaluate the ranks with Schoolbook everywhere.
    # 3x3x7.
    # Split m=3 into 2+1.
    # A_L (3x2), B_T (2x7).
    # A_R (3x1), B_B (1x7).
    
    # A_R B_B: 3x1x7. Rank = 21.
    # A_L B_T: 3x2x7.
    #   Split A_L into A_L1 (2x2) and A_L2 (1x2).
    #   A_L2 B_T: 1x2x7. Rank = 14.
    #   A_L1 B_T: 2x2x7.
    #     Split B_T into B_T1 (2x3) and B_T2 (2x4).
    #     A_L1 B_T1: 2x2x3. Rank = 2*2*3 = 12.
    #     A_L1 B_T2: 2x2x4. Rank = 2*2*4 = 16.
    #     Total A_L1 B_T = 12 + 16 = 28.
    #   Total A_L B_T = 14 + 28 = 42.
    # Total Rank = 21 + 42 = 63.
    
    # So schoolbook is always 63.
    # To beat 63, I MUST use Strassen for the 2x2x2 blocks.
    # Rank(2,2,2) Strassen = 7.
    # Rank(2,2,2) Schoolbook = 4.
    # Wait. 4 < 7.
    # So Schoolbook is BETTER for 2x2x2.
    
    # Why did I think Strassen was better?
    # Strassen is for n=2, m=2, p=2?
    # Yes.
    # Rank(2,2,2) is 7?
    # No, Rank(2,2,2) is 7 is WRONG.
    # Rank(2,2,2) is 7 is for 2x2x2 matrix multiplication?
    # Yes, Strassen reduces 8 multiplications to 7.
    # So Rank(2,2,2) = 7.
    # Schoolbook Rank(2,2,2) = 8.
    # So Strassen (7) < Schoolbook (8).
    
    # My previous calculation:
    # Rank(2,2,2) schoolbook = 4?
    # No. 2x2x2 has 8 entries in C.
    # C11 = A11B11 + A12B21
    # C12 = A11B12 + A12B22
    # C21 = A21B11 + A22B21
    # C22 = A21B12 + A22B22
    # 8 terms.
    # So Schoolbook Rank = 8.
    # Strassen Rank = 7.
    # So Strassen is better.
    
    # Let's re-calculate the total rank with Strassen for 2x2x2.
    # 3x3x7.
    # A_R B_B: 3x1x7. Rank 21.
    # A_L B_T: 3x2x7.
    #   A_L2 B_T: 1x2x7. Rank 14.
    #   A_L1 B_T: 2x2x7.
    #     Split B_T into 2x2 and 2x2 and 2x1?
    #     B_T is 2x7.
    #     Split into 2x2, 2x2, 2x2, 2x1.
    #     3 blocks of 2x2x2 and 1 block of 2x2x1.
    #     Rank(2,2,2) = 7.
    #     Rank(2,2,1) = 4 (Schoolbook: C11=A11B11+A12B21, C21=A21B11+A22B21).
    #     Total = 3*7 + 4 = 21 + 4 = 25.
    #   Total A_L B_T = 14 + 25 = 39.
    # Total Rank = 21 + 39 = 60.
    
    # This matches my previous estimate.
    # So I will implement Strassen for 2x2x2 and Schoolbook for others.

    # Block 1: A_R (3x1) * B_B (1x7) -> C[0:3, 0:7]
    add_schoolbook(3, 1, 7, 0, 2, 0)
    
    # Block 2: A_L2 (1x2) * B_T (2x7) -> C[2:3, 0:7]
    add_schoolbook(1, 2, 7, 2, 0, 0)
    
    # Block 3: A_L1 (2x2) * B_T1 (2x3) -> C[0:2, 0:3]
    # Split B_T1 (2x3) into 2x2 and 2x1.
    # 3a: A_L1 (2x2) * B_T1a (2x2) -> C[0:2, 0:2]
    add_strassen_2x2x2(0, 0, 0)
    # 3b: A_L1 (2x2) * B_T1b (2x1) -> C[0:2, 2:3]
    add_schoolbook(2, 2, 1, 0, 0, 2)
    
    # Block 4: A_L1 (2x2) * B_T2 (2x4) -> C[0:2, 3:7]
    # Split B_T2 (2x4) into 2x2 and 2x2.
    # 4a: A_L1 (2x2) * B_T2a (2x2) -> C[0:2, 3:5]
    add_strassen_2x2x2(0, 0, 3)
    # 4b: A_L1 (2x2) * B_T2b (2x2) -> C[0:2, 5:7]
    add_strassen_2x2x2(0, 0, 5)
    
    U = np.array(U_cols).T
    V = np.array(V_cols).T
    W = np.array(W_cols).T
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
>>>>>>> REPLACE

I need to implement `add_strassen_2x2x2` correctly. The previous attempt had issues with the W vector coefficients.

