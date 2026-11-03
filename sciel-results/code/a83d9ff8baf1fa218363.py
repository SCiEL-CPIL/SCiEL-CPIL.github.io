import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 5, 6
    
    # Global storage for rank-1 terms
    # Each term is (u_indices, u_values, v_indices, v_values, w_indices, w_values)
    # Or simpler: since we are building sparse vectors, we can just accumulate
    # into dictionaries or lists of (row, val) pairs for each column.
    
    # We will generate the terms recursively.
    terms = []
    
    def add_strassen(i0, j0, k0):
        # Adds 7 terms for a 2x2x2 block starting at (i0, j0, k0)
        # Strassen formulas:
        # M1 = (A11 + A22) * (B11 + B22)
        # M2 = (A21 + A22) * B11
        # M3 = A11 * (B12 - B22)
        # M4 = A22 * (B21 - B11)
        # M5 = (A11 + A12) * B22
        # M6 = (A21 - A11) * (B11 + B12)
        # M7 = (A12 - A22) * (B21 + B22)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # We need to express these as linear combinations of rank-1 terms.
        # Each M is a product of a linear combo of A entries and a linear combo of B entries.
        # This results in rank-1 terms with coefficients.
        
        # Helper to add a term: u_coeff * A[i,j] * v_coeff * B[j,k] -> w_coeff * C[i,k]
        # The term is: (u_vec) (v_vec) (w_vec)
        # u_vec has u_coeff at (i, j)
        # v_vec has v_coeff at (j, k)
        # w_vec has w_coeff at (k, i)
        
        def add_term(i, j, k, coeff):
            # i, j, k are local indices 0 or 1
            # coeff is the scalar multiplier
            # We store the global row indices and the value
            u_row = (i0 + i) * m + (j0 + j)
            v_row = (j0 + j) * p + (k0 + k)
            w_row = (k0 + k) * n + (i0 + i)
            terms.append((u_row, v_row, w_row, coeff))
            
        # M1: (A00 + A11) * (B00 + B11)
        # = A00 B00 + A00 B11 + A11 B00 + A11 B11
        add_term(0, 0, 0, 1)
        add_term(0, 0, 1, 1)
        add_term(1, 1, 0, 1)
        add_term(1, 1, 1, 1)
        
        # M2: (A10 + A11) * B00
        # = A10 B00 + A11 B00
        add_term(1, 0, 0, 1)
        add_term(1, 1, 0, 1)
        
        # M3: A00 * (B01 - B11)
        # = A00 B01 - A00 B11
        add_term(0, 0, 1, 1)
        add_term(0, 1, 1, -1)
        
        # M4: A11 * (B10 - B00)
        # = A11 B10 - A11 B00
        add_term(1, 1, 0, 1)
        add_term(1, 0, 0, -1)
        
        # M5: (A00 + A01) * B11
        # = A00 B11 + A01 B11
        add_term(0, 1, 1, 1)
        add_term(0, 0, 1, 1) # Wait, A01 is (0,1), B11 is (1,1). So i=0, j=1, k=1.
        # Correction: A01 is row 0, col 1. B11 is row 1, col 1.
        # Term is A[0,1] * B[1,1].
        # My add_term takes i, j, k.
        # A[i,j] * B[j,k].
        # For A01 * B11: i=0, j=1, k=1.
        # I called add_term(0, 0, 1, 1) above? No, I called add_term(0, 1, 1, 1) for A00*B11?
        # Let's re-verify M5.
        # M5 = (A00 + A01) * B11.
        # A00 * B11: i=0, j=0, k=1. add_term(0, 0, 1, 1).
        # A01 * B11: i=0, j=1, k=1. add_term(0, 1, 1, 1).
        
        # Let's restart the term generation carefully.
        
        # M1: (A00+A11)(B00+B11)
        # A00B00 (0,0,0), A00B11 (0,0,1), A11B00 (1,1,0), A11B11 (1,1,1)
        
        # M2: (A10+A11)B00
        # A10B00 (1,0,0), A11B00 (1,1,0)
        
        # M3: A00(B01-B11)
        # A00B01 (0,0,1), A00B11 (0,1,1) with -1
        
        # M4: A11(B10-B00)
        # A11B10 (1,1,0), A11B00 (1,0,0) with -1
        
        # M5: (A00+A01)B11
        # A00B11 (0,0,1), A01B11 (0,1,1)
        
        # M6: (A10-A00)(B00+B01)
        # A10B00 (1,0,0), A10B01 (1,0,1), -A00B00 (0,0,0), -A00B01 (0,0,1)
        
        # M7: (A01-A11)(B10+B11)
        # A01B10 (0,1,0), A01B11 (0,1,1), -A11B10 (1,1,0), -A11B11 (1,1,1)

        # We need to sum these coefficients for the same (i,j,k) triples?
        # No, each add_term call creates a *new* rank-1 term in the decomposition.
        # The Strassen algorithm says C = M1...M7.
        # M1 = ...
        # So the total tensor is the sum of all these expanded terms.
        # We just add all of them to `terms`.
        
        # However, we can simplify by combining identical (u_row, v_row, w_row) if we wanted,
        # but for rank minimization, we just need the set of terms.
        # The rank is the number of terms.
        # Strassen gives 7 multiplications, but expanded they are more terms?
        # No, Strassen *is* the decomposition.
        # The 7 "M" values are the rank-1 terms?
        # No, M1 is a scalar. The rank-1 term is $M_1 \cdot (u_1 \otimes v_1 \otimes w_1)$.
        # Wait.
        # In Strassen, we compute 7 intermediate products.
        # $M_1 = (A_{11}+A_{22})(B_{11}+B_{22})$.
        # This is not a rank-1 term in the original tensor basis.
        # It is a linear combination of rank-1 terms.
        # To get the rank-1 decomposition of the *original* tensor, we must expand the M's.
        # So yes, we list all the $(i,j,k)$ contributions with their coefficients.
        
        # The "rank" of the decomposition is the number of such terms we keep.
        # If we just list all 4+2+2+2+2+4+4 = 20 terms, the rank is 20?
        # No, that's the schoolbook expansion of Strassen.
        # The rank of Strassen is 7.
        # How?
        # The decomposition is $\sum_{r=1}^7 u_r \otimes v_r \otimes w_r$.
        # $u_r, v_r, w_r$ are vectors.
        # For M1:
        # $u_1 = e_{11} + e_{22}$
        # $v_1 = e_{11} + e_{22}$
        # $w_1 = e_{11} - e_{22}$ (Wait, C11 uses M1, C22 uses -M1? No.)
        
        # Let's look at the standard Strassen rank-1 terms.
        # Term 1: $u=(1,0,0,1), v=(1,0,0,1), w=(1,0,0,-1)$?
        # Let's derive the vectors.
        # $C_{11} = M_1 + M_4 - M_5 + M_7$
        # $C_{12} = M_3 + M_5$
        # $C_{21} = M_2 + M_4$
        # $C_{22} = M_1 - M_2 + M_3 + M_6$
        
        # The rank-1 terms are defined by the M's.
        # Term 1 (M1):
        # $u_1$ has 1 at (1,1) and (2,2). (Local 0,0 and 1,1)
        # $v_1$ has 1 at (1,1) and (2,2). (Local 0,0 and 1,1)
        # $w_1$ has 1 at (1,1) and 1 at (2,2)?
        # $M_1$ contributes +1 to C11 and +1 to C22.
        # So $w_1$ has 1 at (1,1) and 1 at (2,2).
        
        # Term 2 (M2):
        # $u_2$ has 1 at (2,1) and (2,2). (Local 1,0 and 1,1)
        # $v_2$ has 1 at (1,1). (Local 0,0)
        # $w_2$ has 1 at (2,1). (Local 1,0)
        # $M_2$ contributes +1 to C21.
        
        # Term 3 (M3):
        # $u_3$ has 1 at (1,1). (Local 0,0)
        # $v_3$ has 1 at (1,2) and -1 at (2,2). (Local 0,1 and 1,1)
        # $w_3$ has 1 at (1,2). (Local 0,1)
        # $M_3$ contributes +1 to C12.
        
        # Term 4 (M4):
        # $u_4$ has 1 at (2,2). (Local 1,1)
        # $v_4$ has 1 at (2,1) and -1 at (1,1). (Local 1,0 and 0,0)
        # $w_4$ has 1 at (2,1) and 1 at (1,1). (Local 1,0 and 0,0)
        # $M_4$ contributes +1 to C21 and +1 to C11.
        
        # Term 5 (M5):
        # $u_5$ has 1 at (1,1) and (1,2). (Local 0,0 and 0,1)
        # $v_5$ has 1 at (2,2). (Local 1,1)
        # $w_5$ has -1 at (1,1) and 1 at (1,2). (Local 0,0 and 0,1)
        # $M_5$ contributes -1 to C11 and +1 to C12.
        
        # Term 6 (M6):
        # $u_6$ has 1 at (2,1) and -1 at (1,1). (Local 1,0 and 0,0)
        # $v_6$ has 1 at (1,1) and (1,2). (Local 0,0 and 0,1)
        # $w_6$ has 1 at (2,2). (Local 1,1)
        # $M_6$ contributes +1 to C22.
        
        # Term 7 (M7):
        # $u_7$ has 1 at (1,2) and -1 at (2,2). (Local 0,1 and 1,1)
        # $v_7$ has 1 at (2,1) and (2,2). (Local 1,0 and 1,1)
        # $w_7$ has 1 at (1,1) and 1 at (2,2). (Local 0,0 and 1,1)
        # $M_7$ contributes +1 to C11 and +1 to C22.

        # We will store these 7 terms.
        
        # Term 1
        u1 = [(0,0,1), (1,1,1)]
        v1 = [(0,0,1), (1,1,1)]
        w1 = [(0,0,1), (1,1,1)] # C11, C22
        # Term 2
        u2 = [(1,0,1), (1,1,1)]
        v2 = [(0,0,1)]
        w2 = [(1,0,1)] # C21
        # Term 3
        u3 = [(0,0,1)]
        v3 = [(0,1,1), (1,1,-1)]
        w3 = [(0,1,1)] # C12
        # Term 4
        u4 = [(1,1,1)]
        v4 = [(1,0,1), (0,0,-1)]
        w4 = [(1,0,1), (0,0,1)] # C21, C11
        # Term 5
        u5 = [(0,0,1), (0,1,1)]
        v5 = [(1,1,1)]
        w5 = [(0,0,-1), (0,1,1)] # C11, C12
        # Term 6
        u6 = [(1,0,1), (0,0,-1)]
        v6 = [(0,0,1), (0,1,1)]
        w6 = [(1,1,1)] # C22
        # Term 7
        u7 = [(0,1,1), (1,1,-1)]
        v7 = [(1,0,1), (1,1,1)]
        w7 = [(0,0,1), (1,1,1)] # C11, C22

        # Append these 7 terms to the global list
        # Each term is a list of (u_row, val), (v_row, val), (w_row, val)
        # But we need to map local (i,j) to global row.
        
        for u_pts, v_pts, w_pts in [(u1,v1,w1), (u2,v2,w2), (u3,v3,w3), (u4,v4,w4), (u5,v5,w5), (u6,v6,w6), (u7,v7,w7)]:
            terms.append((u_pts, v_pts, w_pts, i0, j0, k0))

    def add_base_2x1x2(i0, j0, k0):
        # 2x1x2 multiplication
        # C00 = A00 B00
        # C01 = A00 B01
        # C10 = A10 B00
        # C11 = A10 B01
        # 4 terms
        terms.append(([(0,0,1)], [(0,0,1)], [(0,0,1)], i0, j0, k0))
        terms.append(([(0,0,1)], [(0,1,1)], [(0,1,1)], i0, j0, k0))
        terms.append(([(1,0,1)], [(0,0,1)], [(1,0,1)], i0, j0, k0))
        terms.append(([(1,0,1)], [(0,1,1)], [(1,1,1)], i0, j0, k0))

    # Recursive decomposition for 4x5x6
    # Split m=5 into 2 and 3
    
    # Part 1: 4x2x6
    # Split p=6 into 2,2,2
    for k_offset in [0, 2, 4]:
        # 4x2x2
        # Split n=4 into 2,2
        for i_offset in [0, 2]:
            add_strassen(i_offset, 0, k_offset)
            
    # Part 2: 4x3x6
    # Split p=6 into 2,2,2
    for k_offset in [0, 2, 4]:
        # 4x3x2
        # Split n=4 into 2,2
        for i_offset in [0, 2]:
            # 2x3x2
            # Split m=3 into 2,1
            # Left: 2x2x2 at j_offset=2 (global j 2,3)
            # Wait, j_offset for this block is 2.
            # Local j 0,1 -> Global j 2,3.
            # Local j 2 -> Global j 4.
            
            # Sub-part 2a: 2x2x2
            add_strassen(i_offset, 2, k_offset)
            
            # Sub-part 2b: 2x1x2
            add_base_2x1x2(i_offset, 4, k_offset)

    # Now convert `terms` to U, V, W matrices
    rank = len(terms)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for idx, (u_pts, v_pts, w_pts, i0, j0, k0) in enumerate(terms):
        for (li, lj, val) in u_pts:
            gi = i0 + li
            gj = j0 + lj
            row = gi * m + gj
            U[row][idx] = val
            
        for (lj, lk, val) in v_pts:
            gj = j0 + lj
            gk = k0 + lk
            row = gj * p + gk
            V[row][idx] = val
            
        for (lk, li, val) in w_pts:
            gk = k0 + lk
            gi = i0 + li
            row = gk * n + gi
            W[row][idx] = val
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
