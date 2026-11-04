import numpy as np
import time

def construct():
    n, m, p = 4, 5, 5
    # Target: multiply A (n x m) by B (m x p)
    # U shape: (n*m, R), V shape: (m*p, R), W shape: (p*n, R)
    # Convention: C[i, k] = sum_j A[i, j] B[j, k]
    # Tensor T[i*m+j, j*p+k, k*n+i] = 1
    
    # Block decomposition:
    # Split A into A1 (4x4) and A2 (4x1)
    # Split B into B1 (4x5) and B2 (1x5)
    # C = A1 B1 + A1 B2 + A2 B1 + A2 B2
    
    # Terms:
    # 1. A1 B1: (4x4) * (4x5)
    #    We can compute A1 * B1 by treating B1 as [B11 (4x4), B12 (4x1)]
    #    A1 * B11 (4x4x4) can use Strassen (rank 7).
    #    A1 * B12 (4x4x1) is 4 scalar mults (or 4 outer products, rank 4).
    #    Total for A1 B1: 7 + 4 = 11? 
    #    Wait, A1 (4x4) * B11 (4x4) is rank 7 (Strassen).
    #    A1 (4x4) * B12 (4x1): 
    #       Let A1 = rows a1..a4 (vectors in R4), B12 = col b (vector in R4).
    #       Result C12 (4x1) has entries c_i = a_i . b.
    #       This is 4 dot products. Each dot product is rank 1? No, dot product of two vectors of length 4 requires 4 mults.
    #       So 4 * 4 = 16 mults?
    #       Let's check: c_i = sum_j a_ij b_j.
    #       This is a tensor of size 4x4x4? No.
    #       A1 is 4x4. B12 is 4x1.
    #       The term is sum_j A1[i,j] B12[j,0].
    #       This is a rank-3 tensor with indices (i, j, 0).
    #       Since the last index is fixed (0), it's effectively a 2D operation?
    #       No, the decomposition is for the whole matrix mult.
    #       For fixed i, c_i depends on A1 row i and B12 col 0.
    #       The tensor contribution is T[i*4+j, j*5+0, 0*4+i] = 1 for j in 0..3.
    #       This is a sub-tensor.
    #       Can we compute A1 * B12 with less than 16 mults?
    #       A1 (4x4) * v (4x1). This is matrix-vector mult. Rank is 16 (schoolbook).
    #       Is there a faster matrix-vector mult? No, for 4x4 it's 16.
        
    #    So A1 * B1 (4x4 * 4x5):
    #    Split B1 into B11 (4x4) and B12 (4x1).
    #    A1 * B11: Rank 7 (Strassen).
    #    A1 * B12: Rank 16 (Schoolbook).
    #    Total: 7 + 16 = 23.
    
    # 2. A1 B2: (4x4) * (1x5)
    #    B2 is 1x5. A1 is 4x4.
    #    C = A1 * B2.
    #    B2[j] is just B2[0, j].
    #    C[i, k] = sum_j A1[i, j] B2[0, k] = B2[0, k] * (sum_j A1[i, j]).
    #    Let s_i = sum_j A1[i, j].
    #    Then C[i, k] = s_i * B2[0, k].
    #    Computing s_i requires 4 additions, no mults.
    #    Then C is outer product of s (4x1) and B2[0] (1x5).
    #    Outer product of u (4) and v (5) requires 20 mults?
    #    Wait. s_i is a variable. B2[0,k] is a variable.
    #    C[i,k] = s_i B2[0,k].
    #    This is a rank-1 tensor in the variables?
    #    No, s_i is a sum of A1 entries.
    #    The "multiplications" are between the input variables.
    #    s_i = a_i0 + a_i1 + a_i2 + a_i3.
    #    C[i,k] = (a_i0+...+a_i3) b_0k.
    #    This expands to sum_j a_ij b_0k.
    #    This is 4 * 5 = 20 terms.
    #    Each term a_ij b_0k is a bilinear form.
    #    Are they independent? Yes.
    #    So rank is 20.
    #    Can we do better?
    #    We can compute s_i first (0 mults), then multiply s_i * b_0k (20 mults).
    #    But s_i is not an input variable, it's an intermediate.
    #    In tensor decomposition, we count the number of scalar multiplications of input variables.
    #    Usually, "rank" means the number of terms in the decomposition T = sum_r U_r \otimes V_r \otimes W_r.
    #    Each term corresponds to one multiplication of coefficients.
    #    So if we have s_i b_0k, that's 20 terms.
    #    So A1 * B2 rank is 20.
    
    # 3. A2 B1: (4x1) * (4x5)
    #    A2 is 4x1 (column vector). B1 is 4x5.
    #    C = A2 * B1.
    #    C[i, k] = A2[i, 0] * B1[0, k] ... wait.
    #    A2 is 4x1. B1 is 4x5.
    #    Dimensions don't match for standard mult!
    #    A is 4x5. B is 5x5.
    #    A = [A1 (4x4), A2 (4x1)].
    #    B = [B1 (4x5); B2 (1x5)].
    #    A * B = A1 B1 + A1 B2 + A2 B1 + A2 B2.
    #    A2 is 4x1. B1 is 4x5.
    #    A2 * B1: (4x1) * (4x5) is undefined.
    #    Ah, the split of B must be horizontal or vertical correctly.
    #    A (n x m) = [A1 (n x m1), A2 (n x m2)] with m = m1 + m2.
    #    B (m x p) = [B1 (m1 x p); B2 (m2 x p)] with m = m1 + m2.
    #    So B1 is top part, B2 is bottom part.
    #    A2 (4x1) * B2 (1x5) -> (4x5). OK.
    #    A1 (4x4) * B1 (4x5) -> (4x5). OK.
    #    A1 (4x4) * B2 (1x5) -> (4x5). OK.
    #    A2 (4x1) * B1 (4x5) -> (4x5). OK.
    
    #    So:
    #    A2 B1: (4x1) * (4x5).
    #    C[i, k] = sum_j A2[i, j] B1[j, k].
    #    Here A2 has only 1 column? No, A2 is 4x1.
    #    So j ranges 0..0?
    #    No, A is 4x5. Split into 4x4 and 4x1.
    #    So A2 is 4x1.
    #    B is 5x5. Split into 4x5 and 1x5.
    #    B1 is 4x5. B2 is 1x5.
    #    A2 B1: (4x1) * (4x5) is NOT defined.
    #    The inner dimensions must match.
    #    A2 is (n x m2) = (4 x 1).
    #    B1 is (m1 x p) = (4 x 5).
    #    A2 * B1 is (4x1) * (4x5) -> Undefined.
    #    
    #    I messed up the block multiplication.
    #    A = [A1 A2] where A1 is n x m1, A2 is n x m2.
    #    B = [B1; B2] where B1 is m1 x p, B2 is m2 x p.
    #    A B = A1 B1 + A2 B2.
    #    There are no cross terms like A1 B2 or A2 B1 in standard block multiplication if B is split vertically and A horizontally.
    #    Wait.
    #    A (n x m), B (m x p).
    #    Split A into columns: A1 (n x m1), A2 (n x m2).
    #    Split B into rows: B1 (m1 x p), B2 (m2 x p).
    #    A B = A1 B1 + A2 B2.
    #    Yes! Only two terms.
    
    #    So C = A1 B1 + A2 B2.
    
    #    Term 1: A1 B1 (4x4 * 4x5)
    #    Term 2: A2 B2 (4x1 * 1x5)
    
    #    Term 2: A2 (4x1) * B2 (1x5).
    #    C2 = A2 B2.
    #    C2[i, k] = A2[i, 0] B2[0, k].
    #    This is outer product of column A2 and row B2.
    #    Rank = 4 * 5 = 20.
    
    #    Term 1: A1 (4x4) * B1 (4x5).
    #    Split B1 into B11 (4x4) and B12 (4x1).
    #    A1 B1 = A1 B11 + A1 B12.
    #    A1 B11: 4x4 * 4x4. Rank 7 (Strassen).
    #    A1 B12: 4x4 * 4x1. Rank 16 (Schoolbook).
    #    Total Term 1: 7 + 16 = 23.
    
    #    Total Rank = 23 + 20 = 43.
    
    #    Let's verify this logic.
    #    Is A1 B12 really rank 16?
    #    A1 (4x4) * B12 (4x1).
    #    Bilinear map from R^16 to R^4.
    #    Rank is 16.
    
    #    Is A2 B2 really rank 20?
    #    A2 (4x1) * B2 (1x5).
    #    Bilinear map from R^9 to R^20.
    #    C[i,k] = a_i b_k.
    #    20 terms.
    
    #    Total 43.
    
    #    Can we do better?
    #    Maybe combine A1 B12 and A2 B2?
    #    A1 B12: C[i,0] = sum_j A1[i,j] B12[j,0].
    #    A2 B2: C[i,k] = A2[i,0] B2[0,k].
    #    They write to different columns of C (col 0 vs cols 1-4? No, B12 is 4x1, so it writes to C's column 0? No, B1 is 4x5. B12 is the last column of B1?
    #    B1 is 4x5. If we split B1 into B11 (4x4) and B12 (4x1), B12 is the 5th column of B1.
    #    So A1 B12 writes to the 5th column of C.
    #    A2 B2 writes to ALL columns of C.
    #    Specifically, C2[i, k] = A2[i,0] B2[0,k].
    #    So for k=4 (0-indexed), C[i,4] gets contributions from A1 B12 and A2 B2.
    #    C[i,4] = (A1 B12)[i,0] + A2[i,0] B2[0,4].
    #    (A1 B12)[i,0] = sum_j A1[i,j] B1[0+4, j] ? No.
    #    B12 is 4x1. It contains B1[j, 4] for j=0..3.
    #    So (A1 B12)[i,0] = sum_{j=0}^3 A1[i,j] B1[j,4].
    #    A2 B2 term for k=4 is A2[i,0] B2[0,4].
    #    So C[i,4] = sum_{j=0}^3 A1[i,j] B1[j,4] + A2[i,0] B2[0,4].
    #    This is sum_{j=0}^4 A[i,j] B[j,4] if we identify A2[i,0] with A[i,4] and B2[0,4] with B[4,4].
    #    Yes, this is just the standard dot product for column 4.
    
    #    So the decomposition is valid.
    #    Rank 43.
    
    #    Let's construct the matrices.
    #    We need U, V, W of size (20, 43), (25, 43), (20, 43).
    
    #    Indices:
    #    U index: i*m + j = i*5 + j
    #    V index: j*p + k = j*5 + k
    #    W index: k*n + i = k*4 + i
    
    R = 43
    U = np.zeros((n*m, R))
    V = np.zeros((m*p, R))
    W = np.zeros((p*n, R))
    
    r = 0
    
    # --- Part 1: A1 B11 (Strassen 4x4x4) ---
    # A1 indices: i in 0..3, j in 0..3.
    # B11 indices: j in 0..3, k in 0..3.
    # Result C11: i in 0..3, k in 0..3.
    
    # Strassen for 4x4x4?
    # Standard Strassen is for 2x2x2 (rank 7).
    # For 4x4x4, we can use 2x2x2 Strassen on 2x2 blocks.
    # A1 = [[A11, A12], [A21, A22]] (2x2 blocks of 2x2).
    # B11 = [[B11, B12], [B21, B22]].
    # C11 = [[C11, C12], [C21, C22]].
    # This would use 7 * (2x2x2 rank 7)? No.
    # Strassen for 4x4 uses 7 mults of 2x2 matrices.
    # Each 2x2 mult has rank 4.
    # Total rank 7 * 4 = 28?
    # Wait, Strassen for 2x2 is rank 7.
    # If we use Strassen for 4x4 by treating it as 2x2 of 2x2, the rank is 7 * (rank of 2x2 mult).
    # Rank of 2x2 mult is 4.
    # So 7 * 4 = 28.
    # But schoolbook for 4x4x4 is 64.
    # Is there a better 4x4x4?
    # Laderman's algorithm for 4x4x4 has rank 21? No, 21 is for 3x3x3?
    # Laderman 3x3x3 rank 23.
    # For 4x4x4, Strassen is 28.
    # Schoolbook is 64.
    
    # Wait, I used "Strassen 4x4x4 rank 7" in my mental math. That's WRONG.
    # Strassen 2x2x2 is rank 7.
    # Strassen 4x4x4 (recursive) is 7 * 4 = 28.
    # So A1 B11 rank is 28.
    
    # Let's re-evaluate total rank.
    # A1 B11: 28.
    # A1 B12: 16.
    # A2 B2: 20.
    # Total: 28 + 16 + 20 = 64.
    
    # This is NOT better than schoolbook 100?
    # 64 < 100. Yes, it is better!
    
    # Can we do better?
    # Maybe use Strassen for A1 B1 (4x4 * 4x5)?
    # Split B1 into B11 (4x4) and B12 (4x1).
    # A1 B11 (28) + A1 B12 (16) = 44.
    # A2 B2 (20).
    # Total 64.
    
    # What if we use Laderman for 3x3x3?
    # Not directly applicable to 4x4.
    
    # What if we use a different split?
    # A (4x5) B (5x5).
    # Split A into 2x2 blocks?
    # This is getting complicated.
    
    # Let's stick with the 64 rank decomposition.
    # It's a valid improvement over 100.
    
    # Implement Strassen for 4x4x4 (rank 28).
    # Strassen steps for 2x2 matrices M, N:
    # P1 = (M00 + M11) (N00 + N11)
    # P2 = (M10 + M11) N00
    # P3 = M00 (N10 - N11)
    # P4 = M11 (N01 - N00)
    # P5 = (M00 + M01) N11
    # P6 = (M10 - M00) (N00 + N10)
    # P7 = (M01 - M11) (N01 + N11)
    
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # Each Pi is a 2x2 matrix product.
    # So for 4x4, we have 7 such products.
    # Each 2x2 product has rank 4.
    # Total 28 terms.
    
    # Let's code this.
    
    # Helper to add a Strassen 2x2x2 decomposition
    # Maps U indices for M (2x2), V indices for N (2x2), W indices for C (2x2)
    
    def add_strassen_2x2(U_base, V_base, W_base, U, V, W, r_offset):
        # U_base: 2x2 array of U indices
        # V_base: 2x2 array of V indices
        # W_base: 2x2 array of W indices
        
        # M00, M01, M10, M11
        m00, m01, m10, m11 = U_base[0,0], U_base[0,1], U_base[1,0], U_base[1,1]
        # N00, N01, N10, N11
        n00, n01, n10, n11 = V_base[0,0], V_base[0,1], V_base[1,0], V_base[1,1]
        # C00, C01, C10, C11
        c00, c01, c10, c11 = W_base[0,0], W_base[0,1], W_base[1,0], W_base[1,1]
        
        r = r_offset
        # P1 = (M00 + M11) (N00 + N11)
        U[m00, r] += 1; U[m11, r] += 1
        V[n00, r] += 1; V[n11, r] += 1
        W[c00, r] += 1; W[c11, r] += 1
        r += 1
        
        # P2 = (M10 + M11) N00
        U[m10, r] += 1; U[m11, r] += 1
        V[n00, r] += 1
        W[c10, r] += 1
        r += 1
        
        # P3 = M00 (N10 - N11)
        U[m00, r] += 1
        V[n10, r] += 1; V[n11, r] -= 1
        W[c01, r] += 1
        r += 1
        
        # P4 = M11 (N01 - N00)
        U[m11, r] += 1
        V[n01, r] += 1; V[n00, r] -= 1
        W[c00, r] += 1; W[c11, r] += 1
        r += 1
        
        # P5 = (M00 + M01) N11
        U[m00, r] += 1; U[m01, r] += 1
        V[n11, r] += 1
        W[c00, r] -= 1; W[c10, r] += 1
        r += 1
        
        # P6 = (M10 - M00) (N00 + N10)
        U[m10, r] += 1; U[m00, r] -= 1
        V[n00, r] += 1; V[n10, r] += 1
        W[c11, r] += 1
        r += 1
        
        # P7 = (M01 - M11) (N01 + N11)
        U[m01, r] += 1; U[m11, r] -= 1
        V[n01, r] += 1; V[n11, r] += 1
        W[c00, r] += 1; W[c01, r] += 1
        r += 1
        
        return r

    # We need to map the 4x4 block indices to U, V, W indices.
    # A1 is rows 0-3, cols 0-3 of A.
    # B1 is rows 0-3, cols 0-4 of B.
    
    # Split A1 into 2x2 blocks:
    # A1 = [[A00, A01], [A10, A11]]
    # A00: rows 0-1, cols 0-1
    # A01: rows 0-1, cols 2-3
    # A10: rows 2-3, cols 0-1
    # A11: rows 2-3, cols 2-3
    
    # Split B1 into 2x2 blocks (for the 4x4 part B11):
    # B11: rows 0-3, cols 0-3
    # B11 = [[B00, B01], [B10, B11]]
    # B00: rows 0-1, cols 0-1
    # B01: rows 0-1, cols 2-3
    # B10: rows 2-3, cols 0-1
    # B11: rows 2-3, cols 2-3
    
    # Result C11:
    # C11 = [[C00, C01], [C10, C11]]
    
    # U indices for A1 blocks:
    # U_A00: [[0*5+0, 0*5+1], [1*5+0, 1*5+1]]
    # U_A01: [[0*5+2, 0*5+3], [1*5+2, 1*5+3]]
    # U_A10: [[2*5+0, 2*5+1], [3*5+0, 3*5+1]]
    # U_A11: [[2*5+2, 2*5+3], [3*5+2, 3*5+3]]
    
    U_A = np.array([
        [U_A00, U_A01],
        [U_A10, U_A11]
    ]) # This is a 2x2 array of 2x2 arrays
    
    # V indices for B11 blocks:
    # V_B00: [[0*5+0, 0*5+1], [1*5+0, 1*5+1]]
    # V_B01: [[0*5+2, 0*5+3], [1*5+2, 1*5+3]]
    # V_B10: [[2*5+0, 2*5+1], [3*5+0, 3*5+1]]
    # V_B11: [[2*5+2, 2*5+3], [3*5+2, 3*5+3]]
    
    V_B11 = np.array([
        [V_B00, V_B01],
        [V_B10, V_B11]
    ])
    
    # W indices for C11 blocks:
    # W_C00: [[0*4+0, 0*4+1], [1*4+0, 1*4+1]]
    # W_C01: [[0*4+2, 0*4+3], [1*4+2, 1*4+3]]
    # W_C10: [[2*4+0, 2*4+1], [3*4+0, 3*4+1]]
    # W_C11: [[2*4+2, 2*4+3], [3*4+2, 3*4+3]]
    
    W_C11 = np.array([
        [W_C00, W_C01],
        [W_C10, W_C11]
    ])
    
    # Add Strassen for A1 * B11
    r = add_strassen_2x2(U_A, V_B11, W_C11, U, V, W, r)
    
    # --- Part 2: A1 B12 (4x4 * 4x1) ---
    # B12 is col 4 of B1 (which is col 4 of B).
    # B12 indices: j in 0..3, k=4.
    # V indices: j*5 + 4.
    # Result C12 is col 4 of C11? No, C11 is cols 0-3.
    # A1 B1 writes to C cols 0-4.
    # A1 B11 writes to C cols 0-3.
    # A1 B12 writes to C col 4.
    
    # A1 (4x4) * B12 (4x1).
    # C[i, 4] = sum_j A1[i, j] B12[j, 4].
    # Schoolbook: 16 terms.
    
    for i in range(4):
        for j in range(4):
            U[i*5 + j, r] = 1
            V[j*5 + 4, r] = 1
            W[4*4 + i, r] = 1 # k=4, i=i -> 4*4+i
            r += 1
    
    # --- Part 3: A2 B2 (4x1 * 1x5) ---
    # A2 is col 4 of A.
    # A2 indices: i in 0..3, j=4.
    # U indices: i*5 + 4.
    # B2 is row 4 of B.
    # B2 indices: j=4, k in 0..4.
    # V indices: 4*5 + k.
    # Result C2[i, k] = A2[i, 4] B2[4, k].
    # W indices: k*4 + i.
    
    for i in range(4):
        for k in range(5):
            U[i*5 + 4, r] = 1
            V[4*5 + k, r] = 1
            W[k*4 + i, r] = 1
            r += 1
            
    # Total r should be 43.
    assert r == R, f"Rank mismatch: {r} != {R}"
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
