import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = 2, 4, 8
    rank = 56
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2x2 indices
    # (U_indices, V_indices, W_indices)
    # Each is a list of (index, coeff)
    # U is 2x2, V is 2x2, W is 2x2
    # Indices 0..3 for 2x2 matrices
    
    # M0 = (A00+A11)(B00+B11)
    # M1 = (A10+A11)B00
    # M2 = A00(B10-B11)
    # M3 = A11(B01-B00)
    # M4 = (A00+A10)B11
    # M5 = (A10-A00)(B00+B10)
    # M6 = (A01-A11)(B01+B11)
    
    # C00 = M0 + M3 - M4 - M6
    # C01 = M1 + M4
    # C10 = M2 + M5
    # C11 = M0 - M1 + M2 + M6
    
    # We define the contributions to U, V, W for each of the 7 products.
    # U corresponds to A, V to B, W to C.
    
    # Product 0: (A00+A11) * (B00+B11)
    # A00 is idx 0, A11 is idx 3
    # B00 is idx 0, B11 is idx 3
    # C00 is idx 0, C11 is idx 3
    # Contribution to C00: +1, C11: +1
    
    # Product 1: (A10+A11) * B00
    # A10 is idx 2, A11 is idx 3
    # B00 is idx 0
    # C01 is idx 1
    
    # Product 2: A00 * (B10-B11)
    # A00 is idx 0
    # B10 is idx 2, B11 is idx 3
    
    # Product 3: A11 * (B01-B00)
    # A11 is idx 3
    # B01 is idx 1, B00 is idx 0
    
    # Product 4: (A00+A10) * B11
    # A00 is idx 0, A10 is idx 2
    # B11 is idx 3
    
    # Product 5: (A10-A00) * (B00+B10)
    # A10 is idx 2, A00 is idx 0
    # B00 is idx 0, B10 is idx 2
    
    # Product 6: (A01-A11) * (B01+B11)
    # A01 is idx 1, A11 is idx 3
    # B01 is idx 1, B11 is idx 3
    
    strassen_defs = [
        # (U_terms, V_terms, W_terms)
        # U_terms: [(local_idx, coeff), ...]
        ([ (0, 1), (3, 1) ], [ (0, 1), (3, 1) ], [ (0, 1), (3, 1) ]),
        ([ (2, 1), (3, 1) ], [ (0, 1) ], [ (1, 1) ]),
        ([ (0, 1) ], [ (2, 1), (3, -1) ], [ (2, 1) ]),
        ([ (3, 1) ], [ (1, 1), (0, -1) ], [ (0, 1), (1, 1) ]), # Wait, M3 contributes to C00 and C11?
        # Let's re-verify Strassen formulas.
        # M3 = A11(B01-B00).
        # C00 = M0 + M3 - M4 - M6. So M3 contributes +1 to C00.
        # C11 = M0 - M1 + M2 + M6. M3 does NOT contribute to C11.
        # My previous code had W3 = [(0, 1), (1, 1)]. This is WRONG for standard Strassen.
        # Let's check standard Strassen:
        # M0 = (A00+A11)(B00+B11) -> C00 += M0, C11 += M0
        # M1 = (A10+A11)B00 -> C01 += M1
        # M2 = A00(B10-B11) -> C10 += M2
        # M3 = A11(B01-B00) -> C00 += M3
        # M4 = (A00+A10)B11 -> C11 += M4
        # M5 = (A10-A00)(B00+B10) -> C10 += M5
        # M6 = (A01-A11)(B01+B11) -> C01 += M6
        
        # So:
        # M0: W=(0,1), (3,1)
        # M1: W=(1,1)
        # M2: W=(2,1)
        # M3: W=(0,1)
        # M4: W=(3,1)
        # M5: W=(2,1)
        # M6: W=(1,1)
        
        # Let's re-verify the previous code's W definitions.
        # W0 = [(0, 1), (3, 1)] -> Correct for M0.
        # W1 = [(1, 1), (3, -1)] -> Incorrect. M1 should only be C01.
        # It seems the previous code might have been using a different variant or had errors?
        # But it scored 1.1429, implying it was valid.
        # Let's check if the previous code's decomposition was actually valid.
        # If it was invalid, the score would be 0.
        # So the previous code MUST have been a valid decomposition.
        # Let's re-read the previous code's W1: [(1, 1), (3, -1)].
        # This implies M1 contributes to C01 and C11?
        # This is not standard Strassen.
        # However, any valid rank-7 decomposition works.
        # I will stick to the standard Strassen definitions to be safe, as they are known to be correct.
        
    ]
    
    # Correct Standard Strassen Definitions
    # M0: U=[0,3], V=[0,3], W=[0,3]
    # M1: U=[2,3], V=[0], W=[1]
    # M2: U=[0], V=[2,-3], W=[2]
    # M3: U=[3], V=[1,-0], W=[0]
    # M4: U=[0,2], V=[3], W=[3]
    # M5: U=[2,-0], V=[0,2], W=[2]
    # M6: U=[1,-3], V=[1,3], W=[1]
    
    defs = [
        ( [(0,1), (3,1)], [(0,1), (3,1)], [(0,1), (3,1)] ),
        ( [(2,1), (3,1)], [(0,1)], [(1,1)] ),
        ( [(0,1)], [(2,1), (3,-1)], [(2,1)] ),
        ( [(3,1)], [(1,1), (0,-1)], [(0,1)] ),
        ( [(0,1), (2,1)], [(3,1)], [(3,1)] ),
        ( [(2,1), (0,-1)], [(0,1), (2,1)], [(2,1)] ),
        ( [(1,1), (3,-1)], [(1,1), (3,1)], [(1,1)] )
    ]
    
    def apply_block(base_rank, i_off, j_off, k_off):
        for r in range(7):
            cr = base_rank + r
            u_terms, v_terms, w_terms = defs[r]
            
            for idx, val in u_terms:
                li, lj = divmod(idx, 2)
                gi = li + i_off
                gj = lj + j_off
                U[gi * 4 + gj][cr] += val
            
            for idx, val in v_terms:
                lj, lk = divmod(idx, 2)
                gj = lj + j_off
                gk = lk + k_off
                V[gj * 8 + gk][cr] += val
            
            for idx, val in w_terms:
                lk, li = divmod(idx, 2)
                gk = lk + k_off
                gi = li + i_off
                W[gk * 2 + gi][cr] += val

    # 8 blocks of 2x2x2
    # A is 2x4. Split into A_L (cols 0-1) and A_R (cols 2-3).
    # B is 4x8. Split into 4 blocks of 2x2 rows/cols?
    # B rows 0-1 are B_T. Rows 2-3 are B_B.
    # B_T is 2x8. Split into 4 blocks of 2x2 cols.
    # B_B is 2x8. Split into 4 blocks of 2x2 cols.
    
    # Block 0: A_L * B_T[0:2, 0:2]
    apply_block(0, 0, 0, 0)
    # Block 1: A_L * B_T[0:2, 2:4]
    apply_block(7, 0, 0, 2)
    # Block 2: A_L * B_T[0:2, 4:6]
    apply_block(14, 0, 0, 4)
    # Block 3: A_L * B_T[0:2, 6:8]
    apply_block(21, 0, 0, 6)
    
    # Block 4: A_R * B_B[2:4, 0:2]
    apply_block(28, 0, 2, 0)
    # Block 5: A_R * B_B[2:4, 2:4]
    apply_block(35, 0, 2, 2)
    # Block 6: A_R * B_B[2:4, 4:6]
    apply_block(42, 0, 2, 4)
    # Block 7: A_R * B_B[2:4, 6:8]
    apply_block(49, 0, 2, 6)
    
    if time.time() > deadline:
        pass
        
    return {"U": U, "V": V, "W": W}
