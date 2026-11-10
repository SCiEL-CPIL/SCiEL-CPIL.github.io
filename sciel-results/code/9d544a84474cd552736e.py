import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Helper to build trivial decomposition for n x m x p
    def trivial_decomp(n, m, p):
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    # Strassen 2x2x2 decomposition
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # C = [[p, q], [r, s]]
    # p = (a+d)(e+h) - ((d-b)(g+h) + (a+d)(e-g))
    # ... standard Strassen formulas
    # We need the tensor coefficients.
    # U, V, W for 2x2x2, rank 7.
    # Indices: A: 0..3, B: 0..3, C: 0..3
    # A: 00, 01, 10, 11 -> 0, 1, 2, 3
    # B: 00, 01, 10, 11 -> 0, 1, 2, 3
    # C: 00, 01, 10, 11 -> 0, 1, 2, 3
    
    U_s = np.zeros((4, 7))
    V_s = np.zeros((4, 7))
    W_s = np.zeros((4, 7))
    
    # M1 = (a+d)(e+h)
    # U: a, d; V: e, h; W: p, q, r, s (with signs)
    # p = M1 - M4 - M6 + M7
    # q = M5 + M7
    # r = M4 + M6
    # s = M1 - M2 + M3 - M7
    
    # M1: (a+d)(e+h)
    U_s[0, 0] = 1; U_s[2, 0] = 1
    V_s[0, 0] = 1; V_s[3, 0] = 1
    W_s[0, 0] = 1; W_s[1, 0] = 1; W_s[2, 0] = 1; W_s[3, 0] = 1
    
    # M2: (c+d)e
    U_s[1, 1] = 1; U_s[2, 1] = 1  # c is index 2? No. A: 00, 01, 10, 11. c is 10 -> index 2. d is 11 -> index 3.
    # Correction:
    # A: 0:00, 1:01, 2:10, 3:11
    # B: 0:00, 1:01, 2:10, 3:11
    # C: 0:00, 1:01, 2:10, 3:11
    
    # M1 = (a+d)(e+h)
    U_s[0, 0] = 1; U_s[3, 0] = 1
    V_s[0, 0] = 1; V_s[3, 0] = 1
    W_s[0, 0] = 1; W_s[1, 0] = 1; W_s[2, 0] = 1; W_s[3, 0] = 1
    
    # M2 = (c+d)e
    U_s[2, 1] = 1; U_s[3, 1] = 1
    V_s[0, 1] = 1
    W_s[2, 1] = -1; W_s[3, 1] = 1  # r = M4+M6, s = M1-M2+M3-M7. So M2 contributes -1 to s.
    # Wait, s = M1 - M2 + M3 - M7. So W_s[3, 1] = -1.
    # r = M4 + M6. M2 does not contribute to r.
    # Let's re-verify Strassen.
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-d)(g+h)
    # M7 = (b-c)(e+g)
    # p = M1 - M4 - M6 + M7
    # q = M5 + M7
    # r = M4 + M6
    # s = M1 - M2 + M3 - M7
    
    # M1
    U_s[0, 0] = 1; U_s[3, 0] = 1
    V_s[0, 0] = 1; V_s[3, 0] = 1
    W_s[0, 0] = 1; W_s[1, 0] = 1; W_s[2, 0] = 1; W_s[3, 0] = 1
    
    # M2
    U_s[2, 1] = 1; U_s[3, 1] = 1
    V_s[0, 1] = 1
    W_s[3, 1] = -1
    
    # M3
    U_s[0, 2] = 1
    V_s[1, 2] = 1; V_s[3, 2] = -1
    W_s[3, 2] = 1
    
    # M4
    U_s[3, 3] = 1
    V_s[2, 3] = 1; V_s[0, 3] = -1
    W_s[0, 3] = -1; W_s[2, 3] = 1
    
    # M5
    U_s[0, 4] = 1; U_s[1, 4] = 1
    V_s[3, 4] = 1
    W_s[1, 4] = 1
    
    # M6
    U_s[2, 5] = 1; U_s[3, 5] = -1
    V_s[2, 5] = 1; V_s[3, 5] = 1
    W_s[0, 5] = -1; W_s[2, 5] = 1
    
    # M7
    U_s[1, 6] = 1; U_s[2, 6] = -1
    V_s[0, 6] = 1; V_s[2, 6] = 1
    W_s[0, 6] = 1; W_s[1, 6] = 1
    
    return U_s, V_s, W_s

    # Trivial 2x2x1
    U_t, V_t, W_t = trivial_decomp(2, 2, 1)
    
    # Construct 2x2x3 (rank 11)
    # B is 2x3. Split into B1 (2x2) and B2 (2x1).
    # U is 4x11. V is 6x11. W is 2x11? No, C is 2x3, so W is 6x11.
    U_223 = np.zeros((4, 11))
    V_223 = np.zeros((6, 11))
    W_223 = np.zeros((6, 11))
    
    # Part 1: 2x2x2 (cols 0,1 of B)
    U_223[:, 0:7] = U_s
    # V_s is 4x7. Rows correspond to B1 (2x2).
    # B1 is cols 0,1 of B.
    # B indices: 0:00, 1:01, 2:10, 3:11.
    # V_223 rows: 0:00, 1:01, 2:10, 3:11, 4:02, 5:12.
    # B1 uses rows 0,1,2,3 of V_223.
    V_223[0:4, 0:7] = V_s
    # W_s is 4x7. Rows correspond to C1 (2x2).
    # C1 is cols 0,1 of C.
    # C indices: 0:00, 1:01, 2:10, 3:11, 4:02, 5:12.
    # W_223 rows 0,1,2,3 correspond to C1.
    W_223[0:4, 0:7] = W_s
    
    # Part 2: 2x2x1 (col 2 of B)
    U_223[:, 7:11] = U_t
    # V_t is 4x4. Rows correspond to B2 (2x1).
    # B2 is col 2 of B.
    # B indices for col 2: 4:02, 5:12.
    # V_t rows: 0:00, 1:01, 2:10, 3:11? No.
    # trivial_decomp(2,2,1) returns V of shape (2*1, 4) = (2,4).
    # Wait, m=2, p=1. V shape (2, 4).
    # V_t[0, :] corresponds to B[0,0].
    # V_t[1, :] corresponds to B[1,0].
    # In B_223, B[0,0] is index 4, B[1,0] is index 5.
    V_223[4:6, 7:11] = V_t
    # W_t is 4x4? No.
    # trivial_decomp(2,2,1) returns W of shape (1*2, 4) = (2,4).
    # W_t[0, :] corresponds to C[0,0].
    # W_t[1, :] corresponds to C[1,0].
    # In C_223, C[0,0] is index 4, C[1,0] is index 5.
    W_223[4:6, 7:11] = W_t
    
    return U_223, V_223, W_223

    # This is getting complex to chain.
    # I will implement the full 4x4x7 construction directly in the main function.
    pass

# EVOLVE-BLOCK-END
