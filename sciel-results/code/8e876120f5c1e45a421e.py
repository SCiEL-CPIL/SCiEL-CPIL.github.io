import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=6, m=6, p=6
    # We use 2x2 blocks of size 3x3.
    # Block size b = 3.
    # Number of blocks nb = 2.
    # Strassen rank for 2x2 is 7.
    # Schoolbook rank for 3x3 is 27.
    # Total rank = 7 * 27 = 189.
    
    rank = 7 * 27
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen coefficients for 2x2 blocks
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
    
    # Define the linear combinations for A and B for each M_t
    # A_parts[t][I][J] = coefficient
    # B_parts[t][I][J] = coefficient
    # I, J in {0, 1}
    
    A_parts = [
        [[1, 0], [0, 1]], # M1
        [[0, 0], [1, 1]], # M2
        [[1, 0], [0, 0]], # M3
        [[0, 0], [0, 1]], # M4
        [[1, 1], [0, 0]], # M5
        [[-1, 0], [1, 0]],# M6
        [[0, 1], [0, -1]] # M7
    ]
    
    B_parts = [
        [[1, 1], [0, 0]], # M1
        [[1, 0], [0, 0]], # M2
        [[0, 1], [0, -1]],# M3
        [[-1, 0], [0, 1]],# M4
        [[0, 0], [0, 1]], # M5
        [[1, 1], [0, 0]], # M6
        [[0, 0], [1, 1]]  # M7
    ]
    
    # C combination coefficients
    # C[I][J] = sum_t coeff[t][I][J] * M_t
    C_coeffs = [
        [[1, 0], [0, 0]], # M1
        [[0, 0], [0, -1]],# M2
        [[0, 1], [0, 1]], # M3
        [[1, 0], [1, 0]], # M4
        [[-1, 1], [0, 0]],# M5
        [[0, 0], [0, 1]], # M6
        [[1, 0], [0, 0]]  # M7
    ]
    
    # Iterate over Strassen terms
    for t in range(7):
        # For each schoolbook term in the 3x3 multiplication
        for i_local in range(3):
            for j_local in range(3):
                for k_local in range(3):
                    col_idx = t * 27 + (i_local * 3 + j_local) * 3 + k_local
                    
                    # U: Coefficients for A
                    for I in range(2):
                        for J in range(2):
                            coeff_A = A_parts[t][I][J]
                            if coeff_A != 0:
                                # Global index for A
                                # Row: I * 3 + i_local
                                # Col: J * 3 + j_local
                                u_row = I * 3 + i_local
                                u_col = J * 3 + j_local
                                U[u_row * m + u_col, col_idx] += coeff_A
                                
                    # V: Coefficients for B
                    for I in range(2):
                        for J in range(2):
                            coeff_B = B_parts[t][I][J]
                            if coeff_B != 0:
                                # Global index for B
                                # Row: I * 3 + k_local (since B is m x p, and m=6, p=6)
                                # Note: In matrix multiplication C = A B,
                                # A is n x m, B is m x p.
                                # Here n=6, m=6, p=6.
                                # Block row I corresponds to rows of A and cols of B?
                                # A[I, J] multiplies B[J, K].
                                # So for M_t, the B part is a sum of blocks B[J, K].
                                # The "I" in B_parts[t][I][J] above was a bit confusing.
                                # Let's re-verify Strassen structure.
                                # M_t = (sum I,J A[I,J] c_A[I,J]) (sum J,K B[J,K] c_B[J,K])
                                # My A_parts and B_parts definitions above:
                                # A_parts[t][I][J] is coeff for A[I,J].
                                # B_parts[t][I][J] is coeff for B[I,J]?
                                # In Strassen:
                                # M1 = (A00+A11)(B00+B11). A parts: A00, A11. B parts: B00, B11.
                                # M2 = (A10+A11)B00. A parts: A10, A11. B parts: B00.
                                # M3 = A00(B01-B11). A parts: A00. B parts: B01, B11.
                                # M4 = A11(B10-B00). A parts: A11. B parts: B10, B00.
                                # M5 = (A00+A01)B11. A parts: A00, A01. B parts: B11.
                                # M6 = (A10-A00)(B00+B01). A parts: A10, A00. B parts: B00, B01.
                                # M7 = (A01-A11)(B10+B11). A parts: A01, A11. B parts: B10, B11.
                                
                                # So B_parts[t] should be indexed by [J][K] where B[J,K] is the block.
                                # In my previous B_parts definition:
                                # M1: B00, B11. Indices [0][0], [1][1].
                                # M2: B00. Index [0][0].
                                # M3: B01, B11. Indices [0][1], [1][1].
                                # M4: B10, B00. Indices [1][0], [0][0].
                                # M5: B11. Index [1][1].
                                # M6: B00, B01. Indices [0][0], [0][1].
                                # M7: B10, B11. Indices [1][0], [1][1].
                                
                                # My previous B_parts list:
                                # M1: [[1, 1], [0, 0]] -> B00=1, B01=1, B10=0, B11=0. WRONG. Should be B00, B11.
                                # Let's fix B_parts.
                                
                                pass # Placeholder, will rewrite logic below
