"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    # Rank 50 decomposition using Strassen for 2x2x2 blocks and schoolbook for 2x2x1
    
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a Strassen block for 2x2 A part, 2x2 B part
    # A indices: (ai, aj) where ai in 0..1, aj in 0..1 (local)
    # B indices: (bk, bl) where bk in 0..1, bl in 0..1 (local)
    # Global A: (i, j)
    # Global B: (k, l)
    # Global W: (l, i)
    
    def add_strassen_block(a_row_offset, a_col_offset, b_row_offset, b_col_offset):
        nonlocal col
        # Local A: i in 0,1; j in 0,1
        # Local B: k in 0,1; l in 0,1
        # Strassen coefficients
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        terms = [
            # (U_coeffs, V_coeffs, W_coeffs)
            # Each term is a list of (index, value) for U, V, W
            # U index: i*m + j
            # V index: k*p + l
            # W index: l*n + i
            
            # M1
            {
                'U': [((0, 0), 1), ((1, 1), 1)],
                'V': [((0, 0), 1), ((1, 1), 1)],
                'W': [((0, 0), 1), ((1, 1), 1)]
            },
            # M2
            {
                'U': [((1, 0), 1), ((1, 1), 1)],
                'V': [((0, 0), 1)],
                'W': [((0, 0), 1), ((1, 0), 1)] # Wait, W is output C.
                # C00 += M2? No.
                # C00 = M1 + M4 - M5 + M7
                # C01 = M3 + M5
                # C10 = M2 + M4
                # C11 = M1 - M2 + M3 + M6
                # The W matrix in the decomposition sum C = sum U_c V_c W_c^T
                # Actually, the standard form is C[i,k] = sum_r U[i,r] V[k,r] W[r]?? No.
                # The tensor T[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
                # Here T is the multiplication tensor.
                # T[i, j, k] = 1 if C[i,k] = A[i,j] B[j,k]
                # So U is indexed by (i,j), V by (j,k), W by (k,i)
                # The sum is over r.
                # For a rank-1 term, U[i,j] * V[j,k] * W[k,i]
                # This corresponds to the contribution to C[i,k] from A[i,j] B[j,k].
                # So for each Strassen term, we define the linear combination of A, B, C.
                # The coefficients in U, V, W correspond to the coefficients in A, B, C.
            }
        ]
        
        # Let's explicitly define the 7 Strassen terms for local indices
        # Local A indices: (0,0), (0,1), (1,0), (1,1)
        # Local B indices: (0,0), (0,1), (1,0), (1,1)
        # Local C indices: (0,0), (0,1), (1,0), (1,1)
        
        # Term 1: M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
        # U: A00=1, A11=1
        # V: B00=1, B11=1
        # W: C00=1, C11=1
        u1 = {0: 1, 3: 1} # A00, A11
        v1 = {0: 1, 3: 1} # B00, B11
        w1 = {0: 1, 3: 1} # C00, C11
        
        # Term 2: M2 = (A10+A11)B00 -> C10 += M2, C11 -= M2
        u2 = {2: 1, 3: 1} # A10, A11
        v2 = {0: 1} # B00
        w2 = {2: 1, 3: -1} # C10, C11
        
        # Term 3: M3 = A00(B10-B11) -> C01 += M3, C11 += M3
        u3 = {0: 1} # A00
        v3 = {2: 1, 3: -1} # B10, B11
        w3 = {1: 1, 3: 1} # C01, C11
        
        # Term 4: M4 = A11(B01-B00) -> C00 += M4, C10 += M4
        u4 = {3: 1} # A11
        v4 = {1: 1, 0: -1} # B01, B00
        w4 = {0: 1, 2: 1} # C00, C10
        
        # Term 5: M5 = (A00+A01)B11 -> C00 -= M5, C01 += M5
        u5 = {0: 1, 1: 1} # A00, A01
        v5 = {3: 1} # B11
        w5 = {0: -1, 1: 1} # C00, C01
        
        # Term 6: M6 = (A10-A00)(B00+B01) -> C11 += M6
        u6 = {2: 1, 0: -1} # A10, A00
        v6 = {0: 1, 1: 1} # B00, B01
        w6 = {3: 1} # C11
        
        # Term 7: M7 = (A01-A11)(B10+B11) -> C00 += M7
        u7 = {1: 1, 3: -1} # A01, A11
        v7 = {2: 1, 3: 1} # B10, B11
        w7 = {0: 1} # C00
        
        strassen_terms = [
            (u1, v1, w1),
            (u2, v2, w2),
            (u3, v3, w3),
            (u4, v4, w4),
            (u5, v5, w5),
            (u6, v6, w6),
            (u7, v7, w7)
        ]
        
        for u, v, w in strassen_terms:
            for (li, lj), val_u in u.items():
                gi = a_row_offset + li
                gj = a_col_offset + lj
                U[gi * m + gj][col] += val_u
            for (lk, ll), val_v in v.items():
                gk = b_row_offset + lk
                gl = b_col_offset + ll
                V[gk * p + gl][col] += val_v
            for (ll, li), val_w in w.items():
                gl = b_col_offset + ll
                gi = a_row_offset + li
                W[gl * n + gi][col] += val_w
            col += 1

    # Helper to add a schoolbook block for 2x2 A part, 2x1 B part
    def add_schoolbook_2x2x1_block(a_row_offset, a_col_offset, b_row_offset, b_col_offset):
        nonlocal col
        # A is 2x2, B is 2x1. C is 2x1.
        # C0 = A00 B00 + A01 B10
        # C1 = A10 B00 + A11 B10
        # Terms:
        # 1. A00 B00 -> C0
        # 2. A01 B10 -> C0
        # 3. A10 B00 -> C1
        # 4. A11 B10 -> C1
        
        terms = [
            {0: 1}, {0: 1}, {0: 1}, # A00, B00, C0
            {1: 1}, {3: 1}, {0: 1}, # A01, B10, C0
            {2: 1}, {0: 1}, {1: 1}, # A10, B00, C1
            {3: 1}, {3: 1}, {1: 1}  # A11, B10, C1
        ]
        
        for u, v, w in terms:
            for (li, lj), val_u in u.items():
                gi = a_row_offset + li
                gj = a_col_offset + lj
                U[gi * m + gj][col] += val_u
            for (lk, ll), val_v in v.items():
                gk = b_row_offset + lk
                gl = b_col_offset + ll
                V[gk * p + gl][col] += val_v
            for (ll, li), val_w in w.items():
                gl = b_col_offset + ll
                gi = a_row_offset + li
                W[gl * n + gi][col] += val_w
            col += 1

    # Block 1: A_L (cols 0,1) * B_T1 (rows 0,1, cols 0,1)
    add_strassen_block(0, 0, 0, 0)
    # Block 2: A_L (cols 0,1) * B_T2 (rows 0,1, cols 2,3)
    add_strassen_block(0, 0, 0, 2)
    # Block 3: A_L (cols 0,1) * B_T3 (rows 0,1, cols 4,5)
    add_strassen_block(0, 0, 0, 4)
    # Block 4: A_L (cols 0,1) * B_T4 (rows 0,1, col 6)
    add_schoolbook_2x2x1_block(0, 0, 0, 6)
    
    # Block 5: A_R (cols 2,3) * B_B1 (rows 2,3, cols 0,1)
    add_strassen_block(0, 2, 2, 0)
    # Block 6: A_R (cols 2,3) * B_B2 (rows 2,3, cols 2,3)
    add_strassen_block(0, 2, 2, 2)
    # Block 7: A_R (cols 2,3) * B_B3 (rows 2,3, cols 4,5)
    add_strassen_block(0, 2, 2, 4)
    # Block 8: A_R (cols 2,3) * B_B4 (rows 2,3, col 6)
    add_schoolbook_2x2x1_block(0, 2, 2, 6)

    if col != rank:
        # Fallback to schoolbook if something went wrong with column count
        return {"U": [[0]*56 for _ in range(8)], "V": [[0]*56 for _ in range(28)], "W": [[0]*56 for _ in range(14)]}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
