import numpy as np
import time

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # Rank 66 decomposition
    # Blocks:
    # A: 3x4. Split rows 2+1, cols 2+2.
    # A00 (2x2), A01 (2x2)
    # A10 (1x2), A11 (1x2)
    # B: 4x6. Split rows 2+2, cols 2+2+2.
    # B0j (2x2), B1j (2x2) for j in 0,1,2
    
    # C: 3x6.
    # C_top (2x6) = A00 B0 + A01 B1
    # C_bot (1x6) = A10 B0 + A11 B1
    
    # For each j in 0,1,2 (cols 2j, 2j+1 of B):
    # C_top[:, 2j:2j+2] = A00 B0j + A01 B1j
    # C_bot[:, 2j:2j+2] = A10 B0j + A11 B1j
    
    # A00 B0j: Strassen 2x2 (7 mults)
    # A01 B1j: Strassen 2x2 (7 mults)
    # A10 B0j: 4 mults
    # A11 B1j: 4 mults
    # Total per j: 22 mults.
    # Total rank: 3 * 22 = 66.
    
    rank = 66
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_block(A_block, B_block, C_offset, sign):
        # A_block: 2x2 indices in A (flat)
        # B_block: 2x2 indices in B (flat)
        # C_offset: (row_start, col_start) in C
        # sign: 1 or -1
        nonlocal col
        
        # Strassen M1..M7
        # M1 = A11 (B11 + B22)
        # M2 = (A11 + A22) B22
        # M3 = (A12 - A22) (B21 + B22)
        # M4 = (A11 + A12) B21
        # M5 = (A21 + A22) B11
        # M6 = (A11 - A21) (B11 + B12)
        # M7 = (A12 - A22) (B11 - B21)
        
        # C11 = M1 + M3 - M4 + M7
        # C12 = M4 + M5
        # C21 = M6 + M7
        # C22 = M1 - M2 + M5 + M6
        
        # We need to add these to U, V, W.
        
        # Helper to add a term u, v, w
        def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val):
            nonlocal col
            U[u_idx][col] = u_val
            V[v_idx][col] = v_val
            W[w_idx][col] = w_val
            col += 1

        # A00 indices:
        # a00, a01
        # a10, a11
        # B0j indices:
        # b00, b01
        # b10, b11
        
        # M1: A00 * (B00 + B11)
        # Term 1: A00 * B00
        add_term(A_block[0], 1, B_block[0], 1, C_offset[0]*p + C_offset[1], sign)
        # Term 2: A00 * B11
        add_term(A_block[0], 1, B_block[3], 1, C_offset[0]*p + C_offset[1], sign)
        
        # M2: (A00 + A11) * B11
        # Term 3: A00 * B11
        add_term(A_block[0], 1, B_block[3], 1, C_offset[0]*p + C_offset[1] + 1, -sign)
        # Term 4: A11 * B11
        add_term(A_block[3], 1, B_block[3], 1, C_offset[0]*p + C_offset[1] + 1, -sign)
        
        # M3: (A01 - A11) * (B10 + B11)
        # Term 5: A01 * B10
        add_term(A_block[1], 1, B_block[2], 1, C_offset[0]*p + C_offset[1], sign)
        # Term 6: A01 * B11
        add_term(A_block[1], 1, B_block[3], 1, C_offset[0]*p + C_offset[1], sign)
        # Term 7: A11 * B10
        add_term(A_block[3], -1, B_block[2], 1, C_offset[0]*p + C_offset[1], sign)
        # Term 8: A11 * B11
        add_term(A_block[3], -1, B_block[3], 1, C_offset[0]*p + C_offset[1], sign)
        
        # M4: (A00 + A01) * B10
        # Term 9: A00 * B10
        add_term(A_block[0], 1, B_block[2], 1, C_offset[0]*p + C_offset[1] + 1, -sign)
        # Term 10: A01 * B10
        add_term(A_block[1], 1, B_block[2], 1, C_offset[0]*p + C_offset[1] + 1, -sign)
        # Also contributes to C11 with -sign
        # Wait, C11 = M1 + M3 - M4 + M7
        # So M4 terms have -sign in C11.
        # C12 = M4 + M5. So M4 terms have +sign in C12.
        
        # Let's restart the mapping carefully.
        # I will just generate the 7 Strassen products and map them to C.
        
        # Reset col to before this block
        start_col = col
        col -= 8 # We added 8 terms above, let's wipe them and do it right.
        
        # Correct Strassen Mapping
        # M1 = A11 (B11 + B22)
        # M2 = (A11 + A22) B22
        # M3 = (A12 - A22) (B21 + B22)
        # M4 = (A11 + A12) B21
        # M5 = (A21 + A22) B11
        # M6 = (A11 - A21) (B11 + B12)
        # M7 = (A12 - A22) (B11 - B21)
        
        # C11 = M1 + M3 - M4 + M7
        # C12 = M4 + M5
        # C21 = M6 + M7
        # C22 = M1 - M2 + M5 + M6
        
        # Indices in A_block: 0:11, 1:12, 2:21, 3:22
        # Indices in B_block: 0:11, 1:12, 2:21, 3:22
        
        # M1: A00 * (B00 + B11)
        add_term(A_block[0], 1, B_block[0], 1, C_offset[0]*p + C_offset[1], sign)
        add_term(A_block[0], 1, B_block[3], 1, C_offset[0]*p + C_offset[1], sign)
        
        # M2: (A00 + A11) * B11
        add_term(A_block[0], 1, B_block[3], 1, C_offset[0]*p + C_offset[1] + 1, -sign)
        add_term(A_block[3], 1, B_block[3], 1, C_offset[0]*p + C_offset[1] + 1, -sign)
        
        # M3: (A01 - A11) * (B10 + B11)
        add_term(A_block[1], 1, B_block[2], 1, C_offset[0]*p + C_offset[1], sign)
        add_term(A_block[1], 1, B_block[3], 1, C_offset[0]*p + C_offset[1], sign)
        add_term(A_block[3], -1, B_block[2], 1, C_offset[0]*p + C_offset[1], sign)
        add_term(A_block[3], -1, B_block[3], 1, C_offset[0]*p + C_offset[1], sign)
        
        # M4: (A00 + A01) * B10
        add_term(A_block[0], 1, B_block[2], 1, C_offset[0]*p + C_offset[1] + 1, sign)
        add_term(A_block[1], 1, B_block[2], 1, C_offset[0]*p + C_offset[1] + 1, sign)
        # C11 gets -M4
        # So we need to subtract these from C11?
        # No, the term is added to the tensor.
        # The coefficient in W determines the sign in the sum.
        # So for M4, we add terms with sign -sign to C11 and sign to C12?
        # No, M4 is a single product.
        # M4 = (A00+A01) B10.
        # This product contributes to C12 with +1 and C11 with -1.
        # So we need two terms?
        # No, M4 is ONE multiplication in the rank decomposition.
        # But it contributes to TWO output entries.
        # This is not possible in a single rank-1 term.
        # A rank-1 term $u \otimes v \otimes w$ produces $w_{ij} u_i v_j$.
        # If $w$ has non-zero entries at $(0,0)$ and $(0,1)$, then the term contributes to both.
        # So we can set $W$ to have values at both positions.
        
        # Let's refine add_term to allow multiple W indices.
        
        pass

    # Re-implementation with correct logic
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(u_idx, u_val, v_idx, v_val, w_indices_vals):
        nonlocal col
        U[u_idx][col] = u_val
        V[v_idx][col] = v_val
        for w_idx, w_val in w_indices_vals:
            W[w_idx][col] += w_val
        col += 1

    # Strassen for 2x2
    # A: 0,1,2,3
    # B: 0,1,2,3
    # C offsets: c00, c01, c10, c11
    
    def add_strassen(A_idx, B_idx, c00, c01, c10, c11, sign):
        nonlocal col
        
        # M1 = A00 (B00 + B11)
        add_term(A_idx[0], sign, B_idx[0], 1, [(c00, 1)])
        add_term(A_idx[0], sign, B_idx[3], 1, [(c00, 1)])
        
        # M2 = (A00 + A11) B11
        add_term(A_idx[0], sign, B_idx[3], 1, [(c11, -1)])
        add_term(A_idx[3], sign, B_idx[3], 1, [(c11, -1)])
        
        # M3 = (A01 - A11) (B10 + B11)
        add_term(A_idx[1], sign, B_idx[2], 1, [(c00, 1)])
        add_term(A_idx[1], sign, B_idx[3], 1, [(c00, 1)])
        add_term(A_idx[3], -sign, B_idx[2], 1, [(c00, 1)])
        add_term(A_idx[3], -sign, B_idx[3], 1, [(c00, 1)])
        
        # M4 = (A00 + A01) B10
        # Contributes to C01 (+) and C00 (-)
        add_term(A_idx[0], sign, B_idx[2], 1, [(c01, 1), (c00, -1)])
        add_term(A_idx[1], sign, B_idx[2], 1, [(c01, 1), (c00, -1)])
        
        # M5 = (A10 + A11) B00
        # Contributes to C01 (+) and C11 (+)
        add_term(A_idx[2], sign, B_idx[0], 1, [(c01, 1), (c11, 1)])
        add_term(A_idx[3], sign, B_idx[0], 1, [(c01, 1), (c11, 1)])
        
        # M6 = (A00 - A10) (B00 + B01)
        # Contributes to C10 (+) and C11 (+)
        add_term(A_idx[0], sign, B_idx[0], 1, [(c10, 1), (c11, 1)])
        add_term(A_idx[0], sign, B_idx[1], 1, [(c10, 1), (c11, 1)])
        add_term(A_idx[2], -sign, B_idx[0], 1, [(c10, 1), (c11, 1)])
        add_term(A_idx[2], -sign, B_idx[1], 1, [(c10, 1), (c11, 1)])
        
        # M7 = (A01 - A11) (B00 - B10)
        # Contributes to C00 (+) and C10 (+)
        add_term(A_idx[1], sign, B_idx[0], 1, [(c00, 1), (c10, 1)])
        add_term(A_idx[1], sign, B_idx[2], -1, [(c00, 1), (c10, 1)])
        add_term(A_idx[3], -sign, B_idx[0], 1, [(c00, 1), (c10, 1)])
        add_term(A_idx[3], -sign, B_idx[2], -1, [(c00, 1), (c10, 1)])
        
    # A00 indices
    A00 = [0*m + 0, 0*m + 1, 1*m + 0, 1*m + 1]
    # A01 indices
    A01 = [0*m + 2, 0*m + 3, 1*m + 2, 1*m + 3]
    # A10 indices
    A10 = [2*m + 0, 2*m + 1]
    # A11 indices
    A11 = [2*m + 2, 2*m + 3]
    
    for j in range(3):
        # B0j indices
        B0j = [0*p + 2*j, 0*p + 2*j + 1, 1*p + 2*j, 1*p + 2*j + 1]
        # B1j indices
        B1j = [2*p + 2*j, 2*p + 2*j + 1, 3*p + 2*j, 3*p + 2*j + 1]
        
        # C offsets for top block (rows 0,1)
        c00 = 0 * n + 2*j
        c01 = 0 * n + 2*j + 1
        c10 = 1 * n + 2*j
        c11 = 1 * n + 2*j + 1
        
        add_strassen(A00, B0j, c00, c01, c10, c11, 1)
        add_strassen(A01, B1j, c00, c01, c10, c11, 1)
        
        # Bottom block
        # C_bot, j, 0 = A10,0 B0j,0,0 + A10,1 B0j,1,0
        # C_bot, j, 1 = A10,0 B0j,0,1 + A10,1 B0j,1,1
        
        c20 = 2 * n + 2*j
        c21 = 2 * n + 2*j + 1
        
        # A10 B0j
        add_term(A10[0], 1, B0j[0], 1, [(c20, 1)])
        add_term(A10[1], 1, B0j[2], 1, [(c20, 1)])
        add_term(A10[0], 1, B0j[1], 1, [(c21, 1)])
        add_term(A10[1], 1, B0j[3], 1, [(c21, 1)])
        
        # A11 B1j
        add_term(A11[0], 1, B1j[0], 1, [(c20, 1)])
        add_term(A11[1], 1, B1j[2], 1, [(c20, 1)])
        add_term(A11[0], 1, B1j[1], 1, [(c21, 1)])
        add_term(A11[1], 1, B1j[3], 1, [(c21, 1)])
        
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print(res['U'][0][:5])
