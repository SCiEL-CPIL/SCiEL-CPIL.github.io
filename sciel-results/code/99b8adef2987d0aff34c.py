import numpy as np
import time

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Decomposition:
    # Split B (3x7) into rows: B_top (2x7), B_bot (1x7)
    # C = A_top (2x2) * B_top (2x7) + A_bot (2x1) * B_bot (1x7)
    # 
    # For A_top * B_top: split B_top columns into [0,1], [2,3], [4,5], [6]
    #   - 3 Strassen blocks (2x2 x 2x2): 3 * 7 = 21
    #   - 1 naive block (2x2 x 2x1): 4
    #   Subtotal: 25
    #
    # For A_bot * B_bot (2x1 * 1x7): 14 scalar products
    #
    # Total rank = 25 + 14 = 39
    
    rank = 39
    
    U = [[0] * rank for _ in range(n * m)]  # n*m = 6
    V = [[0] * rank for _ in range(m * p)]  # m*p = 21
    W = [[0] * rank for _ in range(p * n)]  # p*n = 14
    
    col = 0
    
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        nonlocal col
        U[u_idx][col] += u_val
        V[v_idx][col] += v_val
        W[w_idx][col] += w_val
        col += 1

    def add_strassen_block(k_start):
        """Add 7 Strassen terms for 2x2 x 2x2 block at columns k_start, k_start+1"""
        # A indices (2x2 part of 2x3 A): A[i,j] -> i*3+j
        a00, a01, a10, a11 = 0, 1, 3, 4
        
        # B indices (2x2 part of 3x7 B, rows 0-1, cols k_start, k_start+1): B[i,j] -> i*7+j
        b00 = 0 * 7 + k_start
        b01 = 0 * 7 + k_start + 1
        b10 = 1 * 7 + k_start
        b11 = 1 * 7 + k_start + 1
        
        # C indices (2x2 part of 2x7 C): C[i,j] -> j*2+i
        c00 = k_start * 2 + 0
        c01 = (k_start + 1) * 2 + 0
        c10 = k_start * 2 + 1
        c11 = (k_start + 1) * 2 + 1
        
        # M1 = (A00+A11)(B00+B11)
        # Contributes: +C00, +C11
        add_term(a00, 1, b00, 1, c00, 1)
        add_term(a00, 1, b11, 1, c00, 1)
        add_term(a11, 1, b00, 1, c00, 1)
        add_term(a11, 1, b11, 1, c00, 1)
        # Wait, that's 4 terms for one multiplication. Strassen uses ONE multiplication
        # for (A00+A11) * (B00+B11). So U has (a00+a11) and V has (b00+b11).
        # This is 1 term, not 4. Let me redo.
        
        # I made an error. Each Strassen "multiplication" is ONE rank-1 term.
        # M1: U has a00 and a11 (both +1), V has b00 and b11 (both +1), W has c00 and c11
        # But we need to distribute: M1 contributes to C00 (+1) and C11 (+1)
        # So we need TWO terms? No - in tensor decomposition, each rank-1 term
        # contributes to ALL output positions where both U and W have non-zero.
        
        # Actually, I need to think about this differently.
        # In a rank-r decomposition, we have r terms:
        # T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
        # For Strassen, each "multiplication" M_s corresponds to ONE column r.
        # U[*,r] encodes the linear combination of A entries
        # V[*,r] encodes the linear combination of B entries
        # W[*,r] encodes which C entries get this product and with what coefficient
        
        # M1: (A00+A11)(B00+B11)
        # U: a00=1, a11=1
        # V: b00=1, b11=1
        # W: c00=1, c11=1
        # This is ONE term (one column in U,V,W)
        
        # Let me restart the Strassen with correct single-term encoding.
        pass  # This function will be replaced
    
    # I need to completely redo this. Let me use a cleaner approach.
    
    # Reset
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    
    def add_rank1(u_entries, v_entries, w_entries):
        """Add one rank-1 term. u_entries, v_entries, w_entries are dicts {index: value}"""
        nonlocal col
        for idx, val in u_entries.items():
            U[idx][col] += val
        for idx, val in v_entries.items():
            V[idx][col] += val
        for idx, val in w_entries.items():
            W[idx][col] += val
        col += 1

    def add_strassen(k_start):
        """Strassen for A[0:2,0:2] * B[0:2, k_start:k_start+2] -> C[0:2, k_start:k_start+2]"""
        a00, a01, a10, a11 = 0, 1, 3, 4
        b00 = 0 * 7 + k_start
        b01 = 0 * 7 + k_start + 1
        b10 = 1 * 7 + k_start
        b11 = 1 * 7 + k_start + 1
        c00 = k_start * 2 + 0
        c01 = (k_start + 1) * 2 + 0
        c10 = k_start * 2 + 1
        c11 = (k_start + 1) * 2 + 1
        
        # M1 = (A00+A11)(B00+B11)
        # C00 += M1, C11 += M1
        add_rank1({a00: 1, a11: 1}, {b00: 1, b11: 1}, {c00: 1, c11: 1})
        
        # M2 = (A10+A11)(B00)
        # C10 += M2, C11 -= M2
        add_rank1({a10: 1, a11: 1}, {b00: 1}, {c10: 1, c11: -1})
        
        # M3 = (A00)(B01-B11)
        # C01 += M3, C11 += M3
        add_rank1({a00: 1}, {b01: 1, b11: -1}, {c01: 1, c11: 1})
        
        # M4 = (A11)(B10-B00)
        # C00 += M4, C10 += M4
        add_rank1({a11: 1}, {b10: 1, b00: -1}, {c00: 1, c10: 1})
        
        # M5 = (A00+A01)(B11)
        # C00 -= M5, C01 += M5
        add_rank1({a00: 1, a01: 1}, {b11: 1}, {c00: -1, c01: 1})
        
        # M6 = (A10-A00)(B00+B01)
        # C11 += M6
        add_rank1({a10: 1, a00: -1}, {b00: 1, b01: 1}, {c11: 1})
        
        # M7 = (A01-A11)(B10+B11)
        # C00 += M7
        add_rank1({a01: 1, a11: -1}, {b10: 1, b11: 1}, {c00: 1})

    # 3 Strassen blocks for column pairs (0,1), (2,3), (4,5)
    add_strassen(0)
    add_strassen(2)
    add_strassen(4)
    
    # Column 6 of B, rows 0-1: naive 2x2 x 2x1
    b06 = 0 * 7 + 6
    b16 = 1 * 7 + 6
    c06 = 6 * 2 + 0
    c16 = 6 * 2 + 1
    
    # A00 * B06 -> C06
    add_rank1({0: 1}, {b06: 1}, {c06: 1})
    # A01 * B16 -> C06
    add_rank1({1: 1}, {b16: 1}, {c06: 1})
    # A10 * B06 -> C16
    add_rank1({3: 1}, {b06: 1}, {c16: 1})
    # A11 * B16 -> C16
    add_rank1({4: 1}, {b16: 1}, {c16: 1})
    
    # A[2,:] (row 2 of A, which is A[0,2] and A[1,2]) times B[2,:] (row 2 of B)
    # A[0,2] index = 0*3+2 = 2
    # A[1,2] index = 1*3+2 = 5
    a02 = 2
    a12 = 5
    
    for k in range(7):
        b2k = 2 * 7 + k
        c0k = k * 2 + 0
        c1k = k * 2 + 1
        
        # A[0,2] * B[2,k] -> C[0,k]
        add_rank1({a02: 1}, {b2k: 1}, {c0k: 1})
        # A[1,2] * B[2,k] -> C[1,k]
        add_rank1({a12: 1}, {b2k: 1}, {c1k: 1})
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
