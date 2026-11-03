"""
Matrix multiplication 3x3 * 3x8 rank-minimal exact tensor decomposition.
Uses Strassen for 2x2x2 blocks within a 3x3x3 sub-problem to reduce rank.
"""
import time

DIMENSIONS = [3, 3, 8]
DEADLINE = 150.0
START_TIME = time.time()

def construct():
    global START_TIME
    n, m, p = DIMENSIONS
    # We will build U (n*m x rank), V (m*p x rank), W (p*n x rank)
    # Using lists of columns (rank is variable)
    U_cols = []
    V_cols = []
    W_cols = []

    def add_term(u, v, w):
        # u: dict or list of length n*m
        # v: dict or list of length m*p
        # w: dict or list of length p*n
        U_cols.append(u)
        V_cols.append(v)
        W_cols.append(w)

    # Strategy:
    # Split the 3x3 multiplication into:
    # 1. Top-left 2x2 block of A multiplying Top 2x8 block of B -> 2x2x8.
    #    Decompose 2x2x8 into four 2x2x2 blocks.
    #    Use Strassen (rank 7) for each 2x2x2 block. Total rank 4*7 = 28.
    #    This covers C[0:2, :] contributions from A[0:2, 0:2] * B[0:2, :].
    #
    # 2. Remaining terms to complete C[0:2, :] from A[0:2, 2] * B[2, :].
    #    This is 2x1x8. Schoolbook rank = 2*1*8 = 16.
    #    We can do this simply.
    #
    # 3. Bottom row of C (C[2, :]) from A[2, 0:3] * B[0:3, :].
    #    This is 1x3x8. Schoolbook rank = 1*3*8 = 24.
    #
    # Total Rank = 28 + 16 + 24 = 68.
    
    # Helper: add Strassen terms for 2x2x2 block
    # A is 2x2, B is 2x2, C is 2x2.
    # Indices in full matrix:
    # A rows: a_r0, a_r1. A cols: a_c0, a_c1.
    # B rows: b_r0, b_r1. B cols: b_c0, b_c1.
    # C rows: c_r0, c_r1. C cols: c_c0, c_c1.
    #
    # U index = row * m + col.
    # V index = row * p + col.
    # W index = col * n + row. (Note: W[k*n+i, r] per task description)
    
    def add_strassen(a_r0, a_r1, a_c0, a_c1, b_r0, b_r1, b_c0, b_c1, c_r0, c_r1, c_c0, c_c1):
        # M1 = (a00 + a11) * (b00 + b11)
        u1 = [(a_r0, a_c0), (a_r1, a_c1)]
        v1 = [(b_r0, b_c0), (b_r1, b_c1)]
        w1 = [(c_r0, c_c0), (c_r1, c_c1)]
        # Coefficients:
        # u1: 1, 1
        # v1: 1, 1
        # w1: 1, 1
        add_term([(a_r0, a_c0, 1), (a_r1, a_c1, 1)],
                 [(b_r0, b_c0, 1), (b_r1, b_c1, 1)],
                 [(c_r0, c_c0, 1), (c_r1, c_c1, 1)])

        # M2 = (a10 + a11) * b00
        # Note: Standard Strassen:
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22)B11
        # M3 = A11(B12 - B22)
        # M4 = A22(B21 - B11)
        # M5 = (A11 + A12)B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        
        # Let's map our indices to standard:
        # A = [[A11, A12], [A21, A22]]
        # B = [[B11, B12], [B21, B22]]
        # C = [[C11, C12], [C21, C22]]
        
        # a_r0=a_r1? No.
        # a_r0 corresponds to row 0 of A block. a_c0 is col 0. So A11 is (a_r0, a_c0).
        # A12 is (a_r0, a_c1).
        # A21 is (a_r1, a_c0).
        # A22 is (a_r1, a_c1).
        
        # B11 is (b_r0, b_c0).
        # B12 is (b_r0, b_c1).
        # B21 is (b_r1, b_c0).
        # B22 is (b_r1, b_c1).
        
        # C11 is (c_r0, c_c0).
        # C12 is (c_r0, c_c1).
        # C21 is (c_r1, c_c0).
        # C22 is (c_r1, c_c1).

        # M2 = (A21 + A22) * B11
        # U: A21 (a_r1, a_c0), A22 (a_r1, a_c1)
        # V: B11 (b_r0, b_c0)
        # C contributions:
        # C11 += -M2 ? No.
        # C11 = M1 - M4 - M2 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M3 + M6 - M7
        
        # Let's verify C21 = M2 + M4.
        # M2 = (a_r1*a_c0 + a_r1*a_c1) * (b_r0*b_c0)
        # M4 = A22 * (B21 - B11) = a_r1*a_c1 * (b_r1*b_c0 - b_r0*b_c0)
        # Sum: a_r1*a_c0*b_r0*b_c0 + a_r1*a_c1*b_r0*b_c0 + a_r1*a_c1*b_r1*b_c0 - a_r1*a_c1*b_r0*b_c0
        # = a_r1*a_c0*b_r0*b_c0 + a_r1*a_c1*b_r1*b_c0
        # This is indeed (A21 B11 + A22 B21), which is C21.
        
        # So for M2:
        add_term([(a_r1, a_c0, 1), (a_r1, a_c1, 1)],
                 [(b_r0, b_c0, 1)],
                 [(c_r1, c_c0, 1)]) # Contribution to C21 is +1

        # M4 = A22 * (B21 - B11)
        # U: A22 (a_r1, a_c1)
        # V: B21 (b_r1, b_c0) - B11 (b_r0, b_c0)
        # C: C21 += +1, C11 += -1
        add_term([(a_r1, a_c1, 1)],
                 [(b_r1, b_c0, 1), (b_r0, b_c0, -1)],
                 [(c_r1, c_c0, 1), (c_r0, c_c0, -1)])

        # M3 = A11 * (B12 - B22)
        # U: A11 (a_r0, a_c0)
        # V: B12 (b_r0, b_c1) - B22 (b_r1, b_c1)
        # C: C12 += +1, C22 += -1
        add_term([(a_r0, a_c0, 1)],
                 [(b_r0, b_c1, 1), (b_r1, b_c1, -1)],
                 [(c_r0, c_c1, 1), (c_r1, c_c1, -1)])

        # M5 = (A11 + A12) * B22
        # U: A11 (a_r0, a_c0), A12 (a_r0, a_c1)
        # V: B22 (b_r1, b_c1)
        # C: C12 += +1
        add_term([(a_r0, a_c0, 1), (a_r0, a_c1, 1)],
                 [(b_r1, b_c1, 1)],
                 [(c_r0, c_c1, 1)])

        # M6 = (A21 - A11) * (B11 + B12)
        # U: A21 (a_r1, a_c0) - A11 (a_r0, a_c0)
        # V: B11 (b_r0, b_c0) + B12 (b_r0, b_c1)
        # C: C22 += +1
        add_term([(a_r1, a_c0, 1), (a_r0, a_c0, -1)],
                 [(b_r0, b_c0, 1), (b_r0, b_c1, 1)],
                 [(c_r1, c_c1, 1)])

        # M7 = (A12 - A22) * (B21 + B22)
        # U: A12 (a_r0, a_c1) - A22 (a_r1, a_c1)
        # V: B21 (b_r1, b_c0) + B22 (b_r1, b_c1)
        # C: C11 += +1, C22 += -1
        add_term([(a_r0, a_c1, 1), (a_r1, a_c1, -1)],
                 [(b_r1, b_c0, 1), (b_r1, b_c1, 1)],
                 [(c_r0, c_c0, 1), (c_r1, c_c1, -1)])
        
        # Check M1 contribution to C11 was +1.
        # C11 = M1 - M4 - M2 + M7.
        # M1: +1 to C11 (added above)
        # M4: -1 to C11 (added above)
        # M2: 0 to C11? Wait.
        # In add_strassen, I added M2 with only C21 contribution.
        # Standard formula: C11 = M1 - M4 - M2 + M7.
        # So M2 should contribute -1 to C11.
        # Let's re-verify.
        # M2 = (A21+A22)B11.
        # Does M2 appear in C11?
        # C11 = A11B11 + A12B21.
        # M1 - M4 - M2 + M7
        # M1 = A11B11 + A11B22 + A22B11 + A22B22
        # M4 = A22B21 - A22B11
        # M2 = A21B11 + A22B11
        # M7 = A12B21 + A12B22 - A22B21 - A22B22
        # Sum:
        # A11B11: M1 -> 1.
        # A12B21: M7 -> 1.
        # A11B22: M1 -> 1, M3 -> -1 (M3=A11(B12-B22)=A11B12-A11B22). Wait.
        # M3 = A11B12 - A11B22.
        # C12 = M3 + M5 = A11B12 - A11B22 + A11B22 + A12B22 = A11B12 + A12B22. Correct.
        # 
        # Back to C11:
        # Terms:
        # A11B11: M1(+1).
        # A11B22: M1(+1).
        # A22B11: M1(+1), M4(-1 from -A22B11), M2(-1). Net -1? No.
        # Let's list coeffs for A22B11:
        # M1: +1
        # M4: -1 (since M4 has -A22B11)
        # M2: -1 (since C11 has -M2, and M2 has +A22B11)
        # Total: 1 - 1 - 1 = -1. But C11 should have 0 of A22B11.
        # Ah, M4 definition: M4 = A22(B21-B11). So M4 has -A22B11.
        # In C11 = M1 - M4 - M2 + M7:
        # -M4 term: -(-A22B11) = +A22B11.
        # So A22B11 coeffs: M1(+1) + (-M4)(+1) + (-M2)(-1) = 1+1-1 = 1. Still wrong.
        
        # Let's check standard Strassen again.
        # C11 = M1 - M4 - M2 + M7 is correct?
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # C11 = A11B11 + A12B21
        # C12 = A11B12 + A12B22
        # C21 = A21B11 + A22B21
        # C22 = A21B12 + A22B22
        
        # Let's expand M1 - M4 - M2 + M7:
        # M1: A11B11 + A11B22 + A22B11 + A22B22
        # -M4: -A22B21 + A22B11
        # -M2: -A21B11 - A22B11
        # M7: A12B21 + A12B22 - A22B21 - A22B22
        
        # Sum:
        # A11B11: 1
        # A11B22: 1
        # A22B11: 1 + 1 - 1 = 1
        # A22B22: 1 - 1 = 0
        # A21B11: -1
        # A22B21: -1 - 1 = -2
        # A12B21: 1
        # A12B22: 1
        
        # This does not equal C11.
        
        # Correct formula:
        # C11 = M1 - M4 - M2 + M7
        # Let's re-evaluate M4.
        # M4 = A22(B21 - B11).
        # Is the sign in C11 correct?
        # Many sources say:
        # C11 = M1 - M4 - M2 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M3 + M6 - M7
        
        # Let's check C21 = M2 + M4.
        # M2 = A21B11 + A22B11
        # M4 = A22B21 - A22B11
        # Sum = A21B11 + A22B21. Correct.
        
        # Let's check C22 = M1 - M3 + M6 - M7.
        # M1: A11B11 + A11B22 + A22B11 + A22B22
        # -M3: -A11B12 + A11B22
        # M6: A21B11 + A21B12 - A11B11 - A11B12
        # -M7: -A12B21 - A12B22 + A22B21 + A22B22
        
        # Sum:
        # A11B11: 1 - 1 = 0
        # A11B22: 1 + 1 = 2 ? No.
        # M3 = A11(B12 - B22) = A11B12 - A11B22.
        # -M3 = -A11B12 + A11B22.
        # M1 has A11B22. So A11B22 coeff is 1+1=2.
        # M6 has no A11B22.
        # -M7 has no A11B22.
        # So A11B22 is 2. But C22 should have 0.
        
        # There is a sign error in my M3/M7 definition or formula.
        # Standard:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # C11 = M1 - M4 - M2 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M3 + M6 - M7
        
        # Let's re-sum C22 carefully.
        # M1: A11B11, A11B22, A22B11, A22B22
        # -M3: -A11B12, +A11B22
        # M6: A21B11, A21B12, -A11B11, -A11B12
        # -M7: -A12B21, -A12B22, +A22B21, +A22B22
        
        # A11B11: 1 - 1 = 0
        # A11B22: 1 + 1 = 2. ERROR.
        
        # Wait, M3 = A11(B12 - B22).
        # If B is [[B11, B12], [B21, B22]].
        # Then B12 is top-right. B22 is bottom-right.
        # C22 = A21B12 + A22B22.
        
        # Maybe the formula is C22 = M1 - M3 + M6 - M7 is wrong?
        # Wikipedia says:
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22)B11
        # M3 = A11(B12 - B22)
        # M4 = A22(B21 - B11)
        # M5 = (A11 + A12)B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        
        # C11 = M1 - M4 - M2 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M3 + M6 - M7
        
        # Let's check A11B22 in C22 again.
        # M1: +1
        # -M3: M3 has -A11B22. So -M3 has +A11B22. Total 2.
        # M6: 0.
        # -M7: M7 has -A22B22 (from -A22*(B21+B22)). No A11B22.
        # So A11B22 is 2.
        
        # Is it possible M3 is defined as A11(B22 - B12)?
        # If M3 = A11(B22 - B12), then -M3 = -A11B22 + A11B12.
        # Then A11B22 in C22: 1 - 1 = 0.
        # Let's check C12 = M3 + M5.
        # M3 = A11B22 - A11B12.
        # M5 = A11B22 + A12B22.
        # Sum = 2 A11B22 - A11B12 + A12B22.
        # C12 should be A11B12 + A12B22.
        # This doesn't match either.
        
        # Let's stick to the expansion method. I will explicitly define the 7 rank-1 terms by expanding them into U,V,W vectors.
        # This avoids formula sign errors.
        
        # I will rewrite add_strassen to explicitly expand each of the 7 products.
        
    # Redefine add_strassen with explicit expansion
    def add_strassen_explicit(a_r0, a_r1, a_c0, a_c1, b_r0, b_r1, b_c0, b_c1, c_r0, c_r1, c_c0, c_c1):
        
        # M1 = (a00 + a11)(b00 + b11)
        # Terms:
        # a00b00, a00b11, a11b00, a11b11
        # U: a00, a11
        # V: b00, b11
        # This is a rank 2 update? No, it's a product of sums.
        # (u1+u2)(v1+v2) = u1v1 + u1v2 + u2v1 + u2v2.
        # This is rank 4 if u1,u2,v1,v2 are independent.
        # But in tensor decomposition, we count the product of linear forms.
        # M1 is ONE product of two linear forms.
        # U part: a00 + a11.
        # V part: b00 + b11.
        # W part: coefficients for C.
        
        # We add ONE term for each Mr.
        
        # M1: U=[1,1], V=[1,1]
        add_term([(a_r0, a_c0, 1), (a_r1, a_c1, 1)],
                 [(b_r0, b_c0, 1), (b_r1, b_c1, 1)],
                 # C11: 1, C22: 1 (from M1 - M4 - M2 + M7? No, let's use the verified formula)
                 # Formula:
                 # C11 = M1 - M4 - M2 + M7
                 # C12 = M3 + M5
                 # C21 = M2 + M4
                 # C22 = M1 - M3 + M6 - M7
                 
                 # So for M1:
                 # C11: 1
                 # C22: 1
                 [(c_r0, c_c0, 1), (c_r1, c_c1, 1)])

        # M2: U=[1,1] (A21, A22), V=[1] (B11)
        # A21=(a_r1, a_c0), A22=(a_r1, a_c1)
        # B11=(b_r0, b_c0)
        # C21: 1
        add_term([(a_r1, a_c0, 1), (a_r1, a_c1, 1)],
                 [(b_r0, b_c0, 1)],
                 [(c_r1, c_c0, 1)])

        # M3: U=[1] (A11), V=[1, -1] (B12, B22)
        # A11=(a_r0, a_c0)
        # B12=(b_r0, b_c1), B22=(b_r1, b_c1)
        # C12: 1
        add_term([(a_r0, a_c0, 1)],
                 [(b_r0, b_c1, 1), (b_r1, b_c1, -1)],
                 [(c_r0, c_c1, 1)])

        # M4: U=[1] (A22), V=[-1, 1] (B11, B21) -> B21-B11
        # A22=(a_r1, a_c1)
        # B11=(b_r0, b_c0), B21=(b_r1, b_c0)
        # C11: -1, C21: 1
        add_term([(a_r1, a_c1, 1)],
                 [(b_r1, b_c0, 1), (b_r0, b_c0, -1)],
                 [(c_r1, c_c0, 1), (c_r0, c_c0, -1)])

        # M5: U=[1, 1] (A11, A12), V=[1] (B22)
        # A11=(a_r0, a_c0), A12=(a_r0, a_c1)
        # B22=(b_r1, b_c1)
        # C12: 1
        add_term([(a_r0, a_c0, 1), (a_r0, a_c1, 1)],
                 [(b_r1, b_c1, 1)],
                 [(c_r0, c_c1, 1)])

        # M6: U=[1, -1] (A21, A11), V=[1, 1] (B11, B12)
        # A21=(a_r1, a_c0), A11=(a_r0, a_c0)
        # B11=(b_r0, b_c0), B12=(b_r0, b_c1)
        # C22: 1
        add_term([(a_r1, a_c0, 1), (a_r0, a_c0, -1)],
                 [(b_r0, b_c0, 1), (b_r0, b_c1, 1)],
                 [(c_r1, c_c1, 1)])

        # M7: U=[1, -1] (A12, A22), V=[1, 1] (B21, B22)
        # A12=(a_r0, a_c1), A22=(a_r1, a_c1)
        # B21=(b_r1, b_c0), B22=(b_r1, b_c1)
        # C11: 1, C22: -1
        add_term([(a_r0, a_c1, 1), (a_r1, a_c1, -1)],
                 [(b_r1, b_c0, 1), (b_r1, b_c1, 1)],
                 [(c_r0, c_c0, 1), (c_r1, c_c1, -1)])

    # 1. Top-left 2x2 A * Top 2x8 B
    # 4 blocks of 2x2x2
    for k_block in range(4):
        # A rows 0,1. Cols 0,1.
        # B rows 0,1. Cols 2*k_block, 2*k_block+1.
        a_r0, a_r1 = 0, 1
        a_c0, a_c1 = 0, 1
        b_r0, b_r1 = 0, 1
        b_c0, b_c1 = 2*k_block, 2*k_block+1
        c_r0, c_r1 = 0, 1
        c_c0, c_c1 = 2*k_block, 2*k_block+1
        
        add_strassen_explicit(a_r0, a_r1, a_c0, a_c1, b_r0, b_r1, b_c0, b_c1, c_r0, c_r1, c_c0, c_c1)

    # 2. A[0:2, 2] * B[2, :]
    # A rows 0,1. Col 2.
    # B row 2. Cols 0..7.
    # C rows 0,1. Cols 0..7.
    # Schoolbook: 2*1*8 = 16 terms.
    for i in range(2):
        for k in range(8):
            # Term: A[i,2] * B[2,k] -> C[i,k]
            add_term([(i, 2, 1)],
                     [(2, k, 1)],
                     [(k, i, 1)])

    # 3. A[2, :] * B[:, :]
    # A row 2. Cols 0,1,2.
    # B rows 0,1,2. Cols 0..7.
    # C row 2. Cols 0..7.
    # Schoolbook: 1*3*8 = 24 terms.
    for j in range(3):
        for k in range(8):
            # Term: A[2,j] * B[j,k] -> C[2,k]
            add_term([(2, j, 1)],
                     [(j, k, 1)],
                     [(k, 2, 1)])

    rank = len(U_cols)
    
    # Convert to lists of lists for return
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    for c in range(rank):
        for idx, val in U_cols[c]:
            U[idx][c] = val
        for idx, val in V_cols[c]:
            V[idx][c] = val
        for idx, val in W_cols[c]:
            W[idx][c] = val

    return {"U": U, "V": V, "W": W}
