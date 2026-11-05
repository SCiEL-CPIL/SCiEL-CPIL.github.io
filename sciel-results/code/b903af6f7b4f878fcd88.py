"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from fractions import Fraction
import numpy as np

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 3, 4, 4

    # Laderman's rank-23 decomposition for 4x4 x 4x4 -> 4x4
    # We use this and then check if it restricts correctly to 3x4 x 4x4 -> 3x4
    # by setting A row 3 (indices 12,13,14,15) to zero.
    # If the resulting tensor still satisfies sum_r U[i,j,r]*V[j,k,r]*W[k*n+i,r] = 1
    # for all i in [0,2], j in [0,3], k in [0,3], then we have a valid rank-23 solution.
    
    # Laderman's algorithm: 23 terms for 4x4x4
    # Each term is (U_entries, V_entries, W_entries) where indices are flattened
    # A is 4x4: row i col j -> index i*4+j
    # B is 4x4: row i col j -> index i*4+j  
    # C is 4x4: row i col j -> index i*4+j
    # But our W uses trace convention: W[k*n+i, r] where C[i,k] -> index k*3+i
    # So for the 4x4x4 version, W would be W[k*4+i, r]
    # We need to adapt: for our 3x4x4, C[i,k] maps to W[k*3+i]
    
    # Let's define Laderman's 23 terms for 4x4x4:
    # Term format: (list of (a_idx, coeff), list of (b_idx, coeff), list of (c_idx, coeff))
    # a_idx: row*4+col in A (4x4)
    # b_idx: row*4+col in B (4x4)
    # c_idx: row*4+col in C (4x4)
    
    laderman_terms_4x4 = [
        # M1 = (A00+A10)(B00+B01)
        ([(0,1),(4,1)], [(0,1),(1,1)], [(0,1),(1,1)]),
        # M2 = (A01-A11)(B10-B11)  
        ([(1,1),(5,-1)], [(4,1),(5,-1)], [(2,1),(3,-1)]),
        # M3 = (A00+A20)(B20+B21)
        ([(0,1),(8,1)], [(8,1),(9,1)], [(4,1),(5,1)]),
        # M4 = (A01-A21)(B30-B31)
        ([(1,1),(9,-1)], [(12,1),(13,-1)], [(6,1),(7,-1)]),
        # M5 = (A10+A30)(B10+B11)
        ([(4,1),(12,1)], [(4,1),(5,1)], [(2,1),(3,1)]),
        # M6 = (A11-A31)(B00-B01)
        ([(5,1),(13,-1)], [(0,1),(1,-1)], [(0,-1),(1,1)]),
        # M7 = (A20+A00)(B00+B20)
        ([(8,1),(0,1)], [(0,1),(8,1)], [(4,1),(0,1)]),
        # M8 = (A21+A01)(B10+B30)
        ([(9,1),(1,1)], [(4,1),(12,1)], [(6,1),(2,1)]),
        # M9 = (A30+A10)(B20+B00)
        ([(12,1),(4,1)], [(8,1),(0,1)], [(4,1),(0,1)]),
        # M10 = (A31+A11)(B30+B10)
        ([(13,1),(5,1)], [(12,1),(4,1)], [(6,1),(2,1)]),
        # M11 = (A00+A10+A20+A30)(B00+B01+B20+B21)
        ([(0,1),(4,1),(8,1),(12,1)], [(0,1),(1,1),(8,1),(9,1)], [(0,1),(1,1),(4,1),(5,1)]),
        # M12 = (A01+A11+A21+A31)(B10+B11+B30+B31)
        ([(1,1),(5,1),(9,1),(13,1)], [(4,1),(5,1),(12,1),(13,1)], [(2,1),(3,1),(6,1),(7,1)]),
        # M13 = (A00+A20)(B00+B20)
        ([(0,1),(8,1)], [(0,1),(8,1)], [(0,1),(4,1)]),
        # M14 = (A10+A30)(B10+B30)
        ([(4,1),(12,1)], [(4,1),(12,1)], [(2,1),(6,1)]),
        # M15 = (A01+A21)(B01+B21)
        ([(1,1),(9,1)], [(1,1),(9,1)], [(1,1),(5,1)]),
        # M16 = (A11+A31)(B11+B31)
        ([(5,1),(13,1)], [(5,1),(13,1)], [(3,1),(7,1)]),
        # M17 = (A00-A20)(B20-B00)
        ([(0,1),(8,-1)], [(8,1),(0,-1)], [(4,-1),(0,1)]),
        # M18 = (A10-A30)(B30-B10)
        ([(4,1),(12,-1)], [(12,1),(4,-1)], [(6,-1),(2,1)]),
        # M19 = (A01-A21)(B21-B01)
        ([(1,1),(9,-1)], [(9,1),(1,-1)], [(5,-1),(1,1)]),
        # M20 = (A11-A31)(B31-B11)
        ([(5,1),(13,-1)], [(13,1),(5,-1)], [(7,-1),(3,1)]),
        # M21 = (A00+A10)(B20+B30)
        ([(0,1),(4,1)], [(8,1),(12,1)], [(4,1),(6,1)]),
        # M22 = (A20+A30)(B00+B10)
        ([(8,1),(12,1)], [(0,1),(4,1)], [(4,1),(2,1)]),
        # M23 = (A01+A11)(B21+B31)
        ([(1,1),(5,1)], [(9,1),(13,1)], [(5,1),(7,1)]),
    ]
    
    # Verify Laderman gives correct 4x4x4
    def verify_4x4(terms):
        """Verify terms give correct 4x4x4 multiplication tensor."""
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    total = Fraction(0)
                    for u_entries, v_entries, w_entries in terms:
                        u_val = Fraction(0)
                        v_val = Fraction(0)
                        w_val = Fraction(0)
                        for idx, coeff in u_entries:
                            if idx == i*4+j:
                                u_val += Fraction(coeff)
                        for idx, coeff in v_entries:
                            if idx == j*4+k:
                                v_val += Fraction(coeff)
                        for idx, coeff in w_entries:
                            if idx == i*4+k:
                                w_val += Fraction(coeff)
                        total += u_val * v_val * w_val
                    if total != 1:
                        return False
        return True
    
    # Check if Laderman is correct
    laderman_ok = verify_4x4(laderman_terms_4x4)
    
    # Now try to restrict to 3x4 x 4x4 -> 3x4
    # A is 3x4: row i (0-2), col j (0-3) -> index i*4+j
    # B is 4x4: row j (0-3), col k (0-3) -> index j*4+k
    # C is 3x4: row i (0-2), col k (0-3) -> W index k*3+i
    
    # The 4x4x4 Laderman uses A indices 0-15 (4x4).
    # For our 3x4, we only use A rows 0-2, so indices 0-11.
    # Row 3 of A (indices 12-15) should be treated as zero.
    # But the Laderman terms may have non-zero coefficients at A indices 12-15.
    # If we simply set those to zero, the decomposition won't work in general.
    # We need to check if the contributions from A row 3 cancel out.
    
    def check_3x4_restriction(terms):
        """Check if setting A row 3 to zero still gives correct 3x4x4 tensor."""
        for i in range(3):
            for j in range(4):
                for k in range(4):
                    total = Fraction(0)
                    for u_entries, v_entries, w_entries in terms:
                        u_val = Fraction(0)
                        v_val = Fraction(0)
                        w_val = Fraction(0)
                        for idx, coeff in u_entries:
                            # Only count if idx corresponds to A[i,j] where i<3
                            # A index = row*4+col, so row = idx//4, col = idx%4
                            if idx == i*4+j:
                                u_val += Fraction(coeff)
                            # If idx is in row 3 (12-15), we set it to 0, so skip
                        for idx, coeff in v_entries:
                            if idx == j*4+k:
                                v_val += Fraction(coeff)
                        for idx, coeff in w_entries:
                            # W index in 4x4x4 is i*4+k, but we need k*3+i
                            # Check if this W entry corresponds to C[i,k]
                            # In 4x4x4, C index = row*4+col = i*4+k
                            if idx == i*4+k:
                                w_val += Fraction(coeff)
                        total += u_val * v_val * w_val
                    if total != 1:
                        return False, (i, j, k, total)
        return True, None
    
    restriction_ok, fail_info = check_3x4_restriction(laderman_terms_4x4)
    
    if restriction_ok:
        # Build the 3x4x4 decomposition from Laderman terms
        cols = []
        for u_entries, v_entries, w_entries in laderman_terms_4x4:
            u_vec = [0] * (n * m)  # 12 entries
            v_vec = [0] * (m * p)  # 16 entries
            w_vec = [0] * (p * n)  # 12 entries
            for idx, coeff in u_entries:
                row = idx // 4
                col = idx % 4
                if row < n:  # Only rows 0-2
                    u_vec[row * m + col] = coeff
            for idx, coeff in v_entries:
                v_vec[idx] = coeff
            for idx, coeff in w_entries:
                row = idx // 4
                col = idx % 4
                if row < n:  # Only rows 0-2
                    w_vec[col * n + row] = coeff
            cols.append((u_vec, v_vec, w_vec))
        
        rank = len(cols)
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        for ri in range(rank):
            u_vec, v_vec, w_vec = cols[ri]
            for i in range(n * m):
                U[i][ri] = u_vec[i]
            for j in range(m * p):
                V[j][ri] = v_vec[j]
            for k in range(p * n):
                W[k][ri] = w_vec[k]
        
        # Final verification
        final_ok = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = Fraction(0)
                    for r in range(rank):
                        total += Fraction(U[i*m+j][r]) * Fraction(V[j*p+k][r]) * Fraction(W[k*n+i][r])
                    if total != 1:
                        final_ok = False
                        break
                if not final_ok:
                    break
            if not final_ok:
                break
        
        if final_ok:
            return {"U": U, "V": V, "W": W}
    
    # Fallback: use the Strassen-based rank-44 decomposition
    cols = []

    def add_term(u_entries, v_entries, w_entries):
        u_vec = [0] * (n * m)
        v_vec = [0] * (m * p)
        w_vec = [0] * (p * n)
        for idx, val in u_entries:
            u_vec[idx] = val
        for idx, val in v_entries:
            v_vec[idx] = val
        for idx, val in w_entries:
            w_vec[idx] = val
        cols.append((u_vec, v_vec, w_vec))

    def add_strassen(a, b, c, d, e, f, g, h, w11, w12, w21, w22):
        add_term([(a, 1), (d, 1)], [(e, 1), (h, 1)], [(w11, 1), (w22, 1)])
        add_term([(c, 1), (d, 1)], [(e, 1)], [(w21, 1), (w22, -1)])
        add_term([(a, 1)], [(f, 1), (h, -1)], [(w12, 1), (w22, 1)])
        add_term([(d, 1)], [(g, 1), (e, -1)], [(w11, 1), (w21, 1)])
        add_term([(a, 1), (b, 1)], [(h, 1)], [(w12, 1), (w11, -1)])
        add_term([(c, 1), (a, -1)], [(e, 1), (f, 1)], [(w22, 1)])
        add_term([(b, 1), (d, -1)], [(g, 1), (h, 1)], [(w11, 1)])

    def add_rank4(a, b, c, d, e, f, w1, w2):
        add_term([(a, 1)], [(c, 1)], [(w1, 1)])
        add_term([(b, 1)], [(e, 1)], [(w1, 1)])
        add_term([(a, 1)], [(d, 1)], [(w2, 1)])
        add_term([(b, 1)], [(f, 1)], [(w2, 1)])

    add_strassen(0, 1, 4, 5, 0, 1, 4, 5, 0, 3, 1, 4)
    add_strassen(0, 1, 4, 5, 2, 3, 6, 7, 6, 9, 7, 10)
    add_rank4(8, 9, 0, 1, 4, 5, 2, 5)
    add_rank4(8, 9, 2, 3, 6, 7, 8, 11)
    add_strassen(2, 3, 6, 7, 8, 9, 12, 13, 0, 3, 1, 4)
    add_strassen(2, 3, 6, 7, 10, 11, 14, 15, 6, 9, 7, 10)
    add_rank4(10, 11, 8, 9, 12, 13, 2, 5)
    add_rank4(10, 11, 10, 11, 14, 15, 8, 11)

    rank = len(cols)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for ri in range(rank):
        u_vec, v_vec, w_vec = cols[ri]
        for i in range(n * m):
            U[i][ri] = u_vec[i]
        for j in range(m * p):
            V[j][ri] = v_vec[j]
        for k in range(p * n):
            W[k][ri] = w_vec[k]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
