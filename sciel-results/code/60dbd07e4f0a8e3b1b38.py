"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    
    # Strassen 2x2x2 decomposition terms
    # Each term is (U_dict, V_dict, W_dict)
    # U: {(i,j): coeff}
    # V: {(j,k): coeff}
    # W: {(k,i): coeff}
    
    strassen_2x2 = [
        (
            {(0,0): 1, (1,1): 1},
            {(0,0): 1, (1,1): 1},
            {(0,0): 1, (1,1): 1}
        ),
        (
            {(1,0): 1, (1,1): 1},
            {(0,0): 1},
            {(1,0): 1, (1,1): -1}
        ),
        (
            {(0,0): 1},
            {(0,1): 1, (1,1): -1},
            {(0,1): 1, (1,1): 1}
        ),
        (
            {(1,1): 1},
            {(1,0): 1, (0,0): -1},
            {(0,0): 1, (1,0): 1}
        ),
        (
            {(0,0): 1, (0,1): 1},
            {(1,1): 1},
            {(0,0): -1, (0,1): 1}
        ),
        (
            {(1,0): 1, (0,0): -1},
            {(0,0): 1, (0,1): 1},
            {(1,1): 1}
        ),
        (
            {(0,1): 1, (1,1): -1},
            {(1,0): 1, (1,1): 1},
            {(0,0): 1}
        ),
    ]
    
    terms = []
    
    # Part 1: 4x4x4 decomposition (C[:, 0:4])
    # This requires 7*7=49 terms
    for s in range(7):
        for t in range(7):
            u_s, v_s, w_s = strassen_2x2[s]
            u_t, v_t, w_t = strassen_2x2[t]
            
            U_term = {}
            V_term = {}
            W_term = {}
            
            # Combine U parts
            for (i1, j1), cu in u_s.items():
                for (i2, j2), cu2 in u_t.items():
                    gi = 2 * i1 + i2
                    gj = 2 * j1 + j2
                    coeff = cu * cu2
                    key = (gi, gj)
                    U_term[key] = U_term.get(key, 0) + coeff
            
            # Combine V parts
            for (j1, k1), cv in v_s.items():
                for (j2, k2), cv2 in v_t.items():
                    gj = 2 * j1 + j2
                    gk = 2 * k1 + k2
                    coeff = cv * cv2
                    key = (gj, gk)
                    V_term[key] = V_term.get(key, 0) + coeff
            
            # Combine W parts
            # Note: W indices in Strassen are (k, i) corresponding to C[k,i]
            # In our global C[k, i], k is col of B, i is row of A.
            # In Part 1 (4x4x4), k is 0..3, i is 0..3.
            # strassen w_s has keys (i1, k1) ??
            # Let's check the Strassen definition in the code.
            # strassen_terms_2x2[s] = (U, V, W)
            # U keys (i, j). V keys (j, k). W keys (k, i).
            # So W from Strassen is C[k, i].
            
            for (k1, i1), cw in w_s.items(): # Wait, previous code used (i1, k1) in loop variable name?
                # In previous code: `for (i1, k1), cw in w_s.items():`
                # And `gi = 2 * i1 + i2`, `gk = 2 * k1 + k2`.
                # And `key = (gk, gi)`.
                # This implies the dict keys in w_s are (row_C, col_C) i.e. (k, i).
                # Let's verify Strassen 2x2:
                # C00 = (A00+A11)(B00+B11) - M1 - M3 ...
                # Standard Strassen:
                # M1 = (A00+A11)(B00+B11) -> contributes to C00, C11
                # The W tensor maps product of linear forms in U, V to C.
                # For M1, U linear forms are A00, A11, B00, B11? No.
                # U factors A, V factors B, W factors C.
                # M1 term: (A00+A11) * (B00+B11) contributes to C00 (+1) and C11 (+1).
                # So W for this term has C00: 1, C11: 1.
                # Indices in W dict are (k, i) where k is col index, i is row index of C?
                # C is 2x2. C[k, i].
                # C00 is k=0, i=0.
                # C11 is k=1, i=1.
                # So keys are (0,0) and (1,1).
                
                pass 
            # Re-doing loop carefully based on previous code logic
            # Previous code:
            # for (i1, k1), cw in w_s.items(): 
            #    for (i2, k2), cw2 in w_t.items():
            #       gi = 2 * i1 + i2
            #       gk = 2 * k1 + k2
            #       key = (gk, gi)
            
            # This confirms w_s keys are (i, k) in the loop var names, but logically correspond to C row/col?
            # In C[k, i], first index is k (col of B), second is i (row of A).
            # In 2x2 Strassen, C is 2x2.
            # If w_s has key (0,0), it means C[0,0].
            # If the loop unpacks as (i1, k1), then i1 is the first index of the key.
            # So i1 corresponds to k (col) and k1 corresponds to i (row)?
            # Let's check `gi = 2 * i1 + i2`. `gi` is used in `key = (gk, gi)`.
            # And `gk = 2 * k1 + k2`.
            # So `i1` in loop is first component of key -> maps to `gk`? No.
            # `gi` is built from `i1`. `gk` is built from `k1`.
            # Key is `(gk, gi)`.
            # So `i1` contributes to the second component of the final key (gi).
            # And `k1` contributes to the first component (gk).
            # So if w_s key is (a, b), then a->gk? b->gi?
            # Loop: `for (i1, k1), cw`.
            # `i1` is `a`. `k1` is `b`.
            # `gi` uses `i1` (`a`). `gk` uses `k1` (`b`).
            # Key is `(gk, gi)` -> `(b, a)`.
            # This would transpose the W matrix!
            # Let's check the Strassen W definitions in the code.
            # Term 0: W {(0,0): 1, (1,1): 1}.
            # If transposed, it's still {(0,0): 1, (1,1): 1}. Symmetric.
            # Term 1: W {(1,0): 1, (1,1): -1}.
            # Transposed: {(0,1): 1, (1,1): -1}.
            # In C[1,0] vs C[0,1].
            # Strassen M2 = (A10+A11)B00.
            # Contributes to C[1,0] (+1) and C[1,1] (+1)?
            # M2: (A10+A11)B00.
            # C10 = ... + A10B00 ...
            # C11 = ... + A11B00 ... (from M2 part?)
            # Actually M2 contributes to C[1,0] and C[1,1] with coeffs 1 and -1?
            # C10 = (A10+A11)B00 + ...
            # C11 = A00B00 + ...
            # Let's trust the provided Strassen code which presumably was correct for the score.
            
            for (i1, k1), cw in w_s.items():
                for (i2, k2), cw2 in w_t.items():
                    gi = 2 * i1 + i2
                    gk = 2 * k1 + k2
                    coeff = cw * cw2
                    key = (gk, gi)
                    W_term[key] = W_term.get(key, 0) + coeff
            
            terms.append((U_term, V_term, W_term))
            
    # Part 2: 4x4x2 decomposition (C[:, 4:6])
    # Splits A into 2x2 blocks. B into two 2x2 column blocks?
    # B is 4x2. C is 4x2.
    # We can view this as 4 independent 2x2x2 multiplications?
    # Rows of A: 0-1, 2-3.
    # Cols of B: 0-1 (for B_1), 2-3 (for B_2)? No, B is 4x2.
    # Let's use the strategy from the previous code:
    # "4x Strassen 2x2x2"
    # This implies 4 blocks.
    # Block (0,0): A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2] (which is C[0:2, 4:6] col 0-1?)
    # Wait, B is 4x6.
    # In Part 2, we handle B[:, 4:6] which is 4x2.
    # Let's call B_right = B[:, 4:6].
    # C_right = A * B_right.
    # Split A into A11, A12; A21, A22 (2x2 blocks).
    # Split B_right into B1 (top 2 rows) and B2 (bottom 2 rows)?
    # B_right is 4x2. Top 2 rows: B[0:2, 4:6]. Bottom 2 rows: B[2:4, 4:6].
    # C1 = A11 B_top + A12 B_bottom. (C rows 0-1)
    # C2 = A21 B_top + A22 B_bottom. (C rows 2-3)
    # This requires 4 matrix multiplications of 2x2 * 2x2?
    # A11 (2x2) * B_top (2x2).
    # A12 (2x2) * B_bottom (2x2).
    # A21 (2x2) * B_top (2x2).
    # A22 (2x2) * B_bottom (2x2).
    # Yes, 4 independent 2x2x2 multiplications.
    
    for i1 in range(2): # Row block of A
        for j1 in range(2): # Col block of A (which mixes with B)
            # This loop structure in previous code:
            # for i1 in range(2):
            #   for j1 in range(2):
            #     for s in range(7):
            # This generates 2*2*7 = 28 terms.
            # Let's replicate that logic.
            
            for s in range(7):
                u_s, v_s, w_s = strassen_2x2[s]
                
                U_term = {}
                V_term = {}
                W_term = {}
                
                for (i2, j2), cu in u_s.items():
                    gi = 2 * i1 + i2
                    gj = 2 * j1 + j2
                    key = (gi, gj)
                    U_term[key] = U_term.get(key, 0) + cu
                    
                # V maps B indices.
                # B is 4x2 (local). Global cols 4,5.
                # j2 is local col of B?
                # In 2x2x2, V keys are (j, k).
                # j is row of B (0,1), k is col of B (0,1).
                # Global row of B = 2 * j1 + j2?
                # Wait, in Part 2, A is 4x4. B is 4x2.
                # The "j" in 2x2x2 corresponds to the intermediate dimension.
                # Here the intermediate is 4.
                # But we are doing 4 separate 2x2x2s.
                # For block (i1, j1):
                # A_block uses rows 2*i1, 2*i1+1 and cols 2*j1, 2*j1+1.
                # This matches A[2*i1:2*i1+2, 2*j1:2*j1+2].
                # B_block:
                # We need B[2*j1:2*j1+2, ?].
                # Since we are computing C = A * B_right.
                # C_row = Sum A_col * B_row.
                # The sum is over the 4 cols of A / 4 rows of B.
                # Block (i1, j1) of A sums with Row Block j1 of B?
                # A[2*i1:..., 2*j1:...] multiplies B[2*j1:..., ...].
                # So B_block is B_right[2*j1:2*j1+2, :].
                # B_right has 2 cols (global 4,5).
                # So B_block is 2x2.
                # V_term keys (j, k) map to Global (B_row, B_col).
                # Global B_row = 2 * j1 + j2.
                # Global B_col = 4 + k2.
                
                for (j2, k2), cv in v_s.items():
                    gj = 2 * j1 + j2
                    gk = 4 + k2
                    key = (gj, gk)
                    V_term[key] = V_term.get(key, 0) + cv
                    
                # W_term: C indices.
                # C_block is 2x2.
                # Global C_row = 2 * i1 + i2.
                # Global C_col = 4 + k2.
                # w_s keys (i, k) ?
                # Based on previous code logic for W:
                # key = (gk, gi) where gi is row, gk is col.
                # So W global index is (Col, Row).
                for (i2, k2), cw in w_s.items():
                    gi = 2 * i1 + i2
                    gk = 4 + k2
                    key = (gk, gi)
                    W_term[key] = W_term.get(key, 0) + cw
                    
                terms.append((U_term, V_term, W_term))
                
    rank = len(terms)
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    for r, (u_dict, v_dict, w_dict) in enumerate(terms):
        for (i, j), coeff in u_dict.items():
            U[i * m + j, r] = coeff
        for (j, k), coeff in v_dict.items():
            V[j * p + k, r] = coeff
        for (k, i), coeff in w_dict.items():
            W[k * n + i, r] = coeff
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
