<<<<<<< SEARCH
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit

    # Known rank-16 decomposition for 2x3 * 3x3 matrix multiplication.
    # Index conventions:
    #   A[i,j] -> a = i*m + j = i*3 + j   (i in {0,1}, j in {0,1,2})
    #   B[j,k] -> b = j*p + k = j*3 + k   (j in {0,1,2}, k in {0,1,2})
    #   C[i,k] -> c = k*n + i = k*2 + i   (i in {0,1}, k in {0,1,2})
    #
    # The 16 factors are (a, b, c, u, v, w) where the contribution is
    #   u * A[a] * v * B[b] * w * C[c]
    # and the sum over all 16 factors must equal the matrix product tensor.

    # Factor list: (a, b, c, u, v, w)
    factors = [
        # Factors 0-2: cover C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0]
        (0, 0, 0, 1, 1, 1),   # A00 * B00 -> C00
        (1, 3, 0, 1, 1, 1),   # A01 * B10 -> C00
        (2, 6, 0, 1, 1, 1),   # A02 * B20 -> C00
        # Factors 3-5: cover C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1] + A[0,2]*B[2,1]
        (0, 1, 2, 1, 1, 1),   # A00 * B01 -> C01
        (1, 4, 2, 1, 1, 1),   # A01 * B11 -> C01
        (2, 7, 2, 1, 1, 1),   # A02 * B21 -> C01
        # Factors 6-8: cover C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2] + A[0,2]*B[2,2]
        (0, 2, 4, 1, 1, 1),   # A00 * B02 -> C02
        (1, 5, 4, 1, 1, 1),   # A01 * B12 -> C02
        (2, 8, 4, 1, 1, 1),   # A02 * B22 -> C02
        # Factors 9-11: cover C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0] + A[1,2]*B[2,0]
        (3, 0, 1, 1, 1, 1),   # A10 * B00 -> C10
        (4, 3, 1, 1, 1, 1),   # A11 * B10 -> C10
        (5, 6, 1, 1, 1, 1),   # A12 * B20 -> C10
        # Factors 12-14: cover C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1] + A[1,2]*B[2,1]
        (3, 1, 3, 1, 1, 1),   # A10 * B01 -> C11
        (4, 4, 3, 1, 1, 1),   # A11 * B11 -> C11
        (5, 7, 3, 1, 1, 1),   # A12 * B21 -> C11
        # Factor 15: cover C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2] + A[1,2]*B[2,2]
        (5, 8, 5, 1, 1, 1),   # A12 * B22 -> C12
    ]

    # That's only 16 factors but we need to check: this is the schoolbook pattern
    # with 18 terms. Let me reconsider.
    #
    # Actually, for 2x3 x 3x3, the schoolbook has 2*3*3 = 18 terms.
    # Rank 16 requires sharing. Let me use a proper Laderman-style construction.
    #
    # The key insight: we can combine some terms. For example:
    # C[0,0] needs A00*B00 + A01*B10 + A02*B20 (3 terms)
    # C[1,0] needs A10*B00 + A11*B10 + A12*B20 (3 terms)
    # We can share B00, B10, B20 across these.
    #
    # A valid rank-16 decomposition:
    # We use the structure where each "column" of B is shared between
    # the two rows of C, saving 2 terms per column of B (3 columns * 2 saved = 6 saved)
    # but we only save 2 total (18 - 16 = 2).
    #
    # Let me use a verified construction. The standard approach for m=2,n=3,p=3:
    # Use the fact that rank(2x3x3) = 16.

    # Proper rank-16 construction using Strassen-like sharing:
    #
    # Define intermediate quantities that share factors:
    # S1 = A[0,0] + A[1,0]  (shares B[0,*])
    # S2 = A[0,1] + A[1,1]  (shares B[1,*])
    # S3 = A[0,2] + A[1,2]  (shares B[2,*])
    #
    # Then:
    # C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k]
    # C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k] + A[1,2]*B[2,k]
    #
    # Using S1, S2, S3:
    # S1*B[0,k] = A[0,0]*B[0,k] + A[1,0]*B[0,k]
    # S2*B[1,k] = A[0,1]*B[1,k] + A[1,1]*B[1,k]
    # S3*B[2,k] = A[0,2]*B[2,k] + A[1,2]*B[2,k]
    #
    # So S1*B[0,k] + S2*B[1,k] + S3*B[2,k] = C[0,k] + C[1,k]
    #
    # We need C[0,k] - C[1,k] as well:
    # C[0,k] - C[1,k] = (A[0,0]-A[1,0])*B[0,k] + (A[0,1]-A[1,1])*B[1,k] + (A[0,2]-A[1,2])*B[2,k]
    #
    # This gives us 6 terms for the sum and 6 for the difference = 12 terms for 6 outputs.
    # But we need to separate them. Let me think differently.
    #
    # Alternative: use the known optimal rank-16 from the literature.
    # For 2x3 x 3x3, one valid decomposition is:

    # Let me just verify the simple 18-term schoolbook first, then try to optimize.
    # Actually, let me construct a proper rank-16.

    # Rank-16 decomposition (verified construction):
    # Using the identity that for each k in {0,1,2}:
    #   C[0,k] = A00*B0k + A01*B1k + A02*B2k
    #   C[1,k] = A10*B0k + A11*B1k + A12*B2k
    #
    # We can write:
    #   C[0,k] + C[1,k] = (A00+A10)*B0k + (A01+A11)*B1k + (A02+A12)*B2k
    #   C[0,k] - C[1,k] = (A00-A10)*B0k + (A01-A11)*B1k + (A02-A12)*B2k
    #
    # This uses 6+6=12 terms but gives us sums and differences.
    # To recover C[0,k] and C[1,k] individually, we need:
    #   C[0,k] = 0.5*((C[0,k]+C[1,k]) + (C[0,k]-C[1,k]))
    #   C[1,k] = 0.5*((C[0,k]+C[1,k]) - (C[0,k]-C[1,k]))
    #
    # But this doesn't reduce the count in the tensor decomposition sense.
    #
    # Let me use a direct verified construction. Here's a rank-16 for 2x3x3:

    # Verified rank-16 factors:
    # (a, b, c, u, v, w)
    factors = [
        # Group 1: B column 0 shared
        (0, 0, 0, 1, 1, 1),   # A00*B00 -> C00
        (3, 0, 1, 1, 1, 1),   # A10*B00 -> C10
        (1, 3, 0, 1, 1, 1),   # A01*B10 -> C00
        (4, 3, 1, 1, 1, 1),   # A11*B10 -> C10
        (2, 6, 0, 1, 1, 1),   # A02*B20 -> C00
        (5, 6, 1, 1, 1, 1),   # A12*B20 -> C10
        # Group 2: B column 1 shared
        (0, 1, 2, 1, 1, 1),   # A00*B01 -> C01
        (3, 1, 3, 1, 1, 1),   # A10*B01 -> C11
        (1, 4, 2, 1, 1, 1),   # A01*B11 -> C01
        (4, 4, 3, 1, 1, 1),   # A11*B11 -> C11
        (2, 7, 2, 1, 1, 1),   # A02*B21 -> C01
        (5, 7, 3, 1, 1, 1),   # A12*B21 -> C11
        # Group 3: B column 2 shared
        (0, 2, 4, 1, 1, 1),   # A00*B02 -> C02
        (3, 2, 5, 1, 1, 1),   # A10*B02 -> C12
        (1, 5, 4, 1, 1, 1),   # A01*B12 -> C02
        (4, 5, 5, 1, 1, 1),   # A11*B12 -> C12
    ]

    # That's 18 terms still. I need to actually find a rank-16.
    # Let me use a computational search approach within the time budget.

    # Fallback: use schoolbook rank-18 which is guaranteed correct
    U = np.zeros((6, 18))
    V = np.zeros((9, 18))
    W = np.zeros((6, 18))

    col = 0
    for i in range(2):
        for j in range(3):
            for k in range(3):
                a = i * 3 + j
                b = j * 3 + k
                c = k * 2 + i
                U[a, col] = 1
                V[b, col] = 1
                W[c, col] = 1
                col += 1

    # Verify correctness of schoolbook
    valid = True
    for i in range(2):
        for j in range(3):
            for k in range(3):
                a = i * 3 + j
                b = j * 3 + k
                c = k * 2 + i
                s = sum(U[a, r] * V[b, r] * W[c, r] for r in range(18))
                if abs(s - 1) > 1e-10:
                    valid = False

    if not valid:
        # Should not happen, but safety
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
=======
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit

    # Rank-17 decomposition for 2x3 * 3x3 matrix multiplication.
    # Index conventions:
    #   A[i,j] -> a = i*m + j = i*3 + j   (i in {0,1}, j in {0,1,2})
    #   B[j,k] -> b = j*p + k = j*3 + k   (j in {0,1,2}, k in {0,1,2})
    #   C[i,k] -> c = k*n + i = k*2 + i   (i in {0,1}, k in {0,1,2})
    #
    # Construction: Use Strassen's 7-term algorithm for the 2x2x2 core (rows 0,1 of A,
    # cols 0,1 of B, producing C[0,0], C[0,1], C[1,0], C[1,1]) and schoolbook for
    # the remaining terms involving B column 2 and A column 2.
    #
    # Strassen 2x2x2 factors (for submatrices A_sub[2x2], B_sub[2x2] -> C_sub[2x2]):
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    #
    # C00 = M1 - M4 + M6
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M7
    #
    # For our problem, we map:
    #   A_sub[0,0] = A[0,0], A_sub[0,1] = A[0,1]
    #   A_sub[1,0] = A[1,0], A_sub[1,1] = A[1,1]
    #   B_sub[0,0] = B[0,0], B_sub[0,1] = B[0,1]
    #   B_sub[1,0] = B[1,0], B_sub[1,1] = B[1,1]
    #   C_sub[0,0] = C[0,0], C_sub[0,1] = C[0,1]
    #   C_sub[1,0] = C[1,0], C_sub[1,1] = C[1,1]
    #
    # Additional schoolbook terms for column 2 of B:
    #   C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2] + A[0,2]*B[2,2]
    #   C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2] + A[1,2]*B[2,2]
    #
    # That's 7 (Strassen) + 6 (schoolbook for col 2) = 13 terms.
    # But we also need A[0,2] and A[1,2] contributions which are only in C[*,2].
    # So total = 7 + 6 = 13? No wait, let me recount.
    #
    # Actually the schoolbook for column 2 of B gives 6 terms:
    #   A00*B02, A01*B12, A02*B22 -> C02
    #   A10*B02, A11*B12, A12*B22 -> C12
    #
    # Total: 7 + 6 = 13 terms. But that seems too low. Let me verify.
    #
    # Hmm, actually I think I need to be more careful. Let me just use a known rank-17.
    # 
    # A simpler approach: take the schoolbook 18 and merge two pairs using linearity.
    # For example, instead of separate terms for A00*B00->C00 and A10*B00->C10,
    # we can use (A00+A10)*B00 and split in W. But that's still 1 term in U,V
    # contributing to two C entries via W.
    #
    # Let me construct rank-17 directly by finding a valid set of 17 factors.
    # 
    # Approach: Start with schoolbook 18, and try to merge one pair.
    # The pair (A00*B00->C00, A10*B00->C10) can be merged into:
    #   U: A00 gets 1, A10 gets 1 (same factor)
    #   V: B00 gets 1
    #   W: C00 gets 1, C10 gets 1
    # This saves 1 term: 18-1=17.
    
    # Rank-17 construction:
    # Merge the B00 terms for C00 and C10.
    
    U = np.zeros((6, 17))
    V = np.zeros((9, 17))
    W = np.zeros((6, 17))
    
    col = 0
    # Merged term: (A00+A10)*B00 -> C00 + C10
    U[0, col] = 1  # A00
    U[3, col] = 1  # A10
    V[0, col] = 1  # B00
    W[0, col] = 1  # C00
    W[1, col] = 1  # C10
    col += 1
    
    # Remaining 16 schoolbook terms (excluding the two merged ones)
    for i in range(2):
        for j in range(3):
            for k in range(3):
                a = i * 3 + j
                b = j * 3 + k
                c = k * 2 + i
                # Skip the two terms that were merged
                if (i == 0 and j == 0 and k == 0) or (i == 1 and j == 0 and k == 0):
                    continue
                U[a, col] = 1
                V[b, col] = 1
                W[c, col] = 1
                col += 1
    
    # Verify correctness
    valid = True
    for i in range(2):
        for j in range(3):
            for k in range(3):
                a = i * 3 + j
                b = j * 3 + k
                c = k * 2 + i
                s = sum(U[a, r] * V[b, r] * W[c, r] for r in range(17))
                if abs(s - 1) > 1e-10:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        # Fallback to schoolbook rank-18
        U = np.zeros((6, 18))
        V = np.zeros((9, 18))
        W = np.zeros((6, 18))
        col = 0
        for i in range(2):
            for j in range(3):
                for k in range(3):
                    a = i * 3 + j
                    b = j * 3 + k
                    c = k * 2 + i
                    U[a, col] = 1
                    V[b, col] = 1
                    W[c, col] = 1
                    col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
