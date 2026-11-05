"""Matrix multiplication 3,3,7 - rank 60 decomposition using block Strassen."""

import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank 60 exact tensor decomposition for multiplying
    a 3x3 matrix by a 3x7 matrix.
    
    Decomposition strategy:
    Split A and B into 2x2, 2x1, 1x2, 1x1 blocks.
    1. A[0:2, 0:2] * B[0:2, :] -> M[0:2, :] (Rank 25 using Strassen on 2x2x2 blocks)
    2. A[0:2, 2] * B[2, :] -> M[0:2, :] (Rank 14)
    3. A[2, 0:2] * B[0:2, :] -> M[2, :] (Rank 14)
    4. A[2, 2] * B[2, :] -> M[2, :] (Rank 7)
    Total Rank: 25 + 14 + 14 + 7 = 60.
    """
    n, m, p = DIMENSIONS
    rank = 60
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    # Helper to add a term: u_idx, v_idx, w_idx, coeff
    def add_term(u_idx, v_idx, w_idx, coeff=1.0):
        nonlocal col
        U[u_idx, col] = coeff
        V[v_idx, col] = 1.0
        W[w_idx, col] = 1.0
        col += 1

    # Helper to add a Strassen M221 term
    # A indices: (r1, c1), (r2, c1) for B col c1? No.
    # M221: A is 2x2, B is 2x1.
    # C1 = A00*B00 + A01*B10
    # C2 = A10*B00 + A11*B10
    def add_m221(a_r1, a_c1, a_r2, a_c2, b_r1, b_c1, b_r2, b_c2, w_r1, w_r2, w_c):
        # Term 1: A[a_r1, a_c1] * B[b_r1, b_c1] -> C[w_r1, w_c]
        add_term(a_r1 * m + a_c1, b_r1 * p + b_c1, w_c * n + w_r1)
        # Term 2: A[a_r1, a_c2] * B[b_r2, b_c1] -> C[w_r1, w_c]
        add_term(a_r1 * m + a_c2, b_r2 * p + b_c1, w_c * n + w_r1)
        # Term 3: A[a_r2, a_c1] * B[b_r1, b_c1] -> C[w_r2, w_c]
        add_term(a_r2 * m + a_c1, b_r1 * p + b_c1, w_c * n + w_r2)
        # Term 4: A[a_r2, a_c2] * B[b_r2, b_c1] -> C[w_r2, w_c]
        add_term(a_r2 * m + a_c2, b_r2 * p + b_c1, w_c * n + w_r2)

    # Helper to add a Strassen M222 term
    # A is 2x2, B is 2x2.
    # Strassen formulas
    def add_m222(a_r1, a_c1, a_r2, a_c2, b_r1, b_c1, b_r2, b_c2, w_r1, w_r2, w_c1, w_c2):
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # We need to expand these into U, V, W terms.
        # This is complex because U and V terms are sums.
        # Instead of expanding manually, we can use the fact that the rank is 7.
        # We will generate the 7 terms explicitly.
        
        # M1
        # (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11
        # This is not a single rank 1 term.
        # Strassen's trick works because we compute M1..M7 as intermediate variables, 
        # but in tensor decomposition, we need to express C directly as sum of rank 1 tensors.
        # The rank 7 decomposition of M222 is:
        # C00 = M1 - M4 + M6
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M3 + M7
        
        # Where:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # Each Mi is a product of a linear combination of A and a linear combination of B.
        # This corresponds to a rank 1 tensor in the "intermediate" space, but 
        # we need to map it to the final C.
        
        # Let's construct the 7 terms.
        
        # Term 1: M1 contributes to C00 (+), C11 (+)
        # M1 = (A00+A11)(B00+B11)
        # U: A00+A11. V: B00+B11.
        # W: C00 (+1), C11 (+1)
        
        # We cannot just put sums in U and V if we want integer/half-integer entries 
        # and standard rank 1 tensors? 
        # Yes we can. The rank 1 tensor is u v^T w^T.
        # If u is a vector with two non-zeros, it's still rank 1.
        
        # Let's define the 7 vectors for U, V, and the coefficients for W.
        
        # M1: U = [1, 0, 1, 0] (for A00, A01, A10, A11 relative to block)
        # Actually, let's use global indices.
        
        # M1:
        # U: A00, A11 have coeff 1.
        # V: B00, B11 have coeff 1.
        # W: C00 coeff 1, C11 coeff 1.
        add_term_mixed([ (a_r1*m+a_c1, 1), (a_r2*m+a_c2, 1) ], 
                       [ (b_r1*p+b_c1, 1), (b_r2*p+b_c2, 1) ],
                       [ (w_c1*n+w_r1, 1), (w_c2*n+w_r2, 1) ])
        
        # M2: (A10+A11)B00
        # U: A10, A11 coeff 1.
        # V: B00 coeff 1.
        # W: C10 coeff 1.
        add_term_mixed([ (a_r2*m+a_c1, 1), (a_r2*m+a_c2, 1) ],
                       [ (b_r1*p+b_c1, 1) ],
                       [ (w_c1*n+w_r2, 1) ])
        
        # M3: A00(B01-B11)
        # U: A00 coeff 1.
        # V: B01 coeff 1, B11 coeff -1.
        # W: C01 coeff 1.
        add_term_mixed([ (a_r1*m+a_c1, 1) ],
                       [ (b_r1*p+b_c2, 1), (b_r2*p+b_c2, -1) ],
                       [ (w_c2*n+w_r1, 1) ])
        
        # M4: A11(B10-B00)
        # U: A11 coeff 1.
        # V: B10 coeff 1, B00 coeff -1.
        # W: C00 coeff -1, C10 coeff 1.
        add_term_mixed([ (a_r2*m+a_c2, 1) ],
                       [ (b_r2*p+b_c1, 1), (b_r1*p+b_c1, -1) ],
                       [ (w_c1*n+w_r1, -1), (w_c1*n+w_r2, 1) ])
        
        # M5: (A00+A01)B11
        # U: A00, A01 coeff 1.
        # V: B11 coeff 1.
        # W: C01 coeff 1.
        add_term_mixed([ (a_r1*m+a_c1, 1), (a_r1*m+a_c2, 1) ],
                       [ (b_r2*p+b_c2, 1) ],
                       [ (w_c2*n+w_r1, 1) ])
        
        # M6: (A10-A00)(B00+B01)
        # U: A10 coeff 1, A00 coeff -1.
        # V: B00, B01 coeff 1.
        # W: C00 coeff 1.
        add_term_mixed([ (a_r2*m+a_c1, 1), (a_r1*m+a_c1, -1) ],
                       [ (b_r1*p+b_c1, 1), (b_r1*p+b_c2, 1) ],
                       [ (w_c1*n+w_r1, 1) ])
        
        # M7: (A01-A11)(B10+B11)
        # U: A01 coeff 1, A11 coeff -1.
        # V: B10, B11 coeff 1.
        # W: C11 coeff 1.
        add_term_mixed([ (a_r1*m+a_c2, 1), (a_r2*m+a_c2, -1) ],
                       [ (b_r2*p+b_c1, 1), (b_r2*p+b_c2, 1) ],
                       [ (w_c2*n+w_r2, 1) ])

    def add_term_mixed(u_terms, v_terms, w_terms):
        nonlocal col
        for u_idx, u_coeff in u_terms:
            U[u_idx, col] = u_coeff
        for v_idx, v_coeff in v_terms:
            V[v_idx, col] = v_coeff
        for w_idx, w_coeff in w_terms:
            W[w_idx, col] = w_coeff
        col += 1

    # --- Block 1: A[0:2, 0:2] * B[0:2, :] ---
    # Split B columns into 2, 2, 2, 1
    
    # Group 1: B cols 0, 1. M222.
    # A rows 0,1 cols 0,1. B rows 0,1 cols 0,1. C rows 0,1 cols 0,1.
    add_m222(0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1)
    
    # Group 2: B cols 2, 3. M222.
    add_m222(0, 0, 1, 1, 0, 2, 1, 3, 0, 1, 2, 3)
    
    # Group 3: B cols 4, 5. M222.
    add_m222(0, 0, 1, 1, 0, 4, 1, 5, 0, 1, 4, 5)
    
    # Group 4: B col 6. M221.
    # A rows 0,1 cols 0,1. B rows 0,1 col 6. C rows 0,1 col 6.
    add_m221(0, 0, 1, 1, 0, 6, 1, 6, 0, 1, 6)

    # --- Block 2: A[0:2, 2] * B[2, :] ---
    # A rows 0,1 col 2. B row 2 cols 0..6. C rows 0,1 cols 0..6.
    for k in range(7):
        # C0k = A02 * B2k
        add_term(0 * m + 2, 2 * p + k, k * n + 0)
        # C1k = A12 * B2k
        add_term(1 * m + 2, 2 * p + k, k * n + 1)

    # --- Block 3: A[2, 0:2] * B[0:2, :] ---
    # A row 2 cols 0,1. B rows 0,1 cols 0..6. C row 2 cols 0..6.
    for k in range(7):
        # C2k += A20 * B0k
        add_term(2 * m + 0, 0 * p + k, k * n + 2)
        # C2k += A21 * B1k
        add_term(2 * m + 1, 1 * p + k, k * n + 2)

    # --- Block 4: A[2, 2] * B[2, :] ---
    # A row 2 col 2. B row 2 cols 0..6. C row 2 cols 0..6.
    for k in range(7):
        # C2k += A22 * B2k
        add_term(2 * m + 2, 2 * p + k, k * n + 2)

    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
