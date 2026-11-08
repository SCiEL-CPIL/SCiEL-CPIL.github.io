"""Matrix multiplication 2x3 by 3x9: rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a decomposition of the matrix multiplication tensor for
    multiplying a 2x3 matrix by a 3x9 matrix.
    
    Strategy: Decompose the 2x3 * 3x9 multiplication into three independent
    2x3 * 3x3 block multiplications (since 9 = 3*3). For each 2x3 * 3x3 block,
    use a rank-17 decomposition (better than schoolbook rank 18) by splitting
    the 3 columns of A into 2+1:
      - 2x2 * 2x3 part: rank 11 (using Strassen-like for 2x2x2 + schoolbook for 2x1x3)
      - 2x1 * 1x3 part: rank 6 (schoolbook)
    Total rank per block: 17. Total rank: 3 * 17 = 51.
    
    The output factor follows W[k*n+i, r] convention where k in [0,9), i in [0,2).
    U has shape (n*m, rank) = (6, rank) with U[i*m+j, r]
    V has shape (m*p, rank) = (27, rank) with V[j*p+k, r]
    W has shape (p*n, rank) = (18, rank) with W[k*n+i, r]
    """
    n, m, p = DIMENSIONS  # n=2, m=3, p=9
    
    # We'll build the decomposition block by block.
    # For each of the 3 blocks of B (columns 0-2, 3-5, 6-8), we decompose 2x3 * 3x3.
    
    total_rank = 0
    U_rows = n * m  # 6
    V_rows = m * p  # 27
    W_rows = p * n  # 18
    
    # We'll accumulate the decomposition terms.
    # Each term is a triple (u_vec, v_vec, w_vec) where:
    #   u_vec has length 6 (indexed by i*m+j)
    #   v_vec has length 27 (indexed by j*p+k)
    #   w_vec has length 18 (indexed by k*n+i)
    
    terms = []
    
    def add_term(u, v, w):
        """Add a rank-1 term. u, v, w are dicts mapping index to value."""
        terms.append((u, v, w))
    
    # For a 2x3 * 3x3 block at column offset col_off (0, 3, or 6):
    # A is 2x3, B_block is 3x3 (columns col_off to col_off+2 of B)
    # C_block is 2x3
    # 
    # Split A's columns into first 2 and last 1:
    #   A_left = A[:, 0:2] (2x2), A_right = A[:, 2] (2x1)
    #   B_top = B_block[0:2, :] (2x3), B_bot = B_block[2, :] (1x3)
    #   C_block = A_left * B_top + A_right * B_bot
    #
    # For A_left * B_top (2x2 * 2x3):
    #   Further split: this is like multiplying 2x2 by 2x3.
    #   We can view 2x2 * 2x3 as two 2x2 * 2x1 multiplications? No.
    #   Actually, let's use a known rank-11 decomposition for 2x2 * 2x3.
    #
    # For 2x2 * 2x3 (rank 11):
    #   Let A2 be 2x2 with entries a00,a01,a10,a11
    #   Let B2 be 2x3 with entries b00,b01,b02,b10,b11,b12
    #   C = A2 * B2 is 2x3 with entries c00,c01,c02,c10,c11,c12
    #
    #   We can use:
    #   - Strassen for the first 2x2 block of B (columns 0,1): rank 7
    #   - Schoolbook for column 2 of B: rank 4
    #   Total: 11
    #
    # Strassen for 2x2 * 2x2 (columns 0,1 of B2):
    #   P1 = (a00+a11)(b00+b11)
    #   P2 = (a10+a11)b00
    #   P3 = a00(b01-b11)
    #   P4 = a11(b10-b00)
    #   P5 = (a00+a01)b11
    #   P6 = (a10-a00)(b00+b01)
    #   P7 = (a01-a11)(b10+b11)
    #   c00 = P1 - P4 + P6 - P7
    #   c01 = P3 + P1 - P5 + P7
    #   c10 = P2 + P4 - P6 + P5
    #   c11 = P1 - P3 + P2 - P4
    #
    # Schoolbook for column 2 of B2:
    #   c02 = a00*b02 + a01*b12
    #   c12 = a10*b02 + a11*b12
    #   4 terms: (a00,b02,c02), (a01,b12,c02), (a10,b02,c12), (a11,b12,c12)
    
    # For A_right * B_bot (2x1 * 1x3):
    #   C_block[:, j] = A_right[:, 0] * B_bot[0, j] for j=0,1,2
    #   6 terms: (a_i, b_j, c_{i,j}) for i=0,1 and j=0,1,2
    
    for col_off in range(3):
        # B_block columns are col_off, col_off+1, col_off+2
        # In V, B entries are at j*p + (col_off + k) for j in 0..2, k in 0..2
        
        # --- Strassen part for A_left (cols 0,1) * B_top (rows 0,1) ---
        # A_left entries: A[0,0], A[0,1], A[1,0], A[1,1]
        # In U: indices are i*m+j = i*3+j
        #   a00 -> U[0*3+0] = U[0]
        #   a01 -> U[0*3+1] = U[1]
        #   a10 -> U[1*3+0] = U[3]
        #   a11 -> U[1*3+1] = U[4]
        
        # B_top entries: B[0, col_off+k], B[1, col_off+k] for k=0,1
        # In V: indices are j*p + (col_off+k)
        #   b00 -> V[0*9 + col_off+0] = V[col_off]
        #   b01 -> V[0*9 + col_off+1] = V[col_off+1]
        #   b10 -> V[1*9 + col_off+0] = V[9+col_off]
        #   b11 -> V[1*9 + col_off+1] = V[9+col_off+1]
        
        # C_block entries for this part: C[0, col_off+0], C[0, col_off+1], C[1, col_off+0], C[1, col_off+1]
        # In W: indices are k*n+i where k is the column of C (0..8) and i is the row (0..1)
        #   c00 -> W[(col_off+0)*2 + 0] = W[2*col_off]
        #   c01 -> W[(col_off+1)*2 + 0] = W[2*col_off+2]
        #   c10 -> W[(col_off+0)*2 + 1] = W[2*col_off+1]
        #   c11 -> W[(col_off+1)*2 + 1] = W[2*col_off+3]
        
        # Strassen terms:
        # P1 = (a00+a11)(b00+b11)
        add_term({0: 1, 4: 1}, {col_off: 1, 9+col_off+1: 1}, {2*col_off: 1, 2*col_off+2: 1, 2*col_off+1: 1, 2*col_off+3: 1})
        # Wait, I need to be more careful. Each Strassen term is a single rank-1 term.
        # Let me redo this properly.
        
        # Actually, let me restart the term construction more carefully.
    
    # Let me rebuild this more carefully.
    terms = []
    
    for col_off in range(3):
        # Indices for A_left (cols 0,1 of A):
        a00 = 0  # U[0*3+0]
        a01 = 1  # U[0*3+1]
        a10 = 3  # U[1*3+0]
        a11 = 4  # U[1*3+1]
        
        # Indices for B_top (rows 0,1 of B, cols col_off, col_off+1):
        b00 = 0*9 + (col_off+0)      # V[0*9 + col_off]
        b01 = 0*9 + (col_off+1)      # V[0*9 + col_off+1]
        b10 = 1*9 + (col_off+0)      # V[9 + col_off]
        b11 = 1*9 + (col_off+1)      # V[9 + col_off+1]
        
        # Indices for C (rows 0,1, cols col_off, col_off+1):
        c00 = (col_off+0)*2 + 0      # W[col_off*2 + 0]
        c01 = (col_off+1)*2 + 0      # W[(col_off+1)*2 + 0]
        c10 = (col_off+0)*2 + 1      # W[col_off*2 + 1]
        c11 = (col_off+1)*2 + 1      # W[(col_off+1)*2 + 1]
        
        # Strassen 2x2x2 for the 2x2 block (A_left * B_top[:, 0:2]):
        # P1 = (a00+a11)(b00+b11)
        terms.append(({a00: 1, a11: 1}, {b00: 1, b11: 1}, {c00: 1, c01: 1, c10: 1, c11: 1}))
        # P2 = (a10+a11)b00
        terms.append(({a10: 1, a11: 1}, {b00: 1}, {c10: 1, c11: 1}))
        # P3 = a00(b01-b11)
        terms.append(({a00: 1}, {b01: 1, b11: -1}, {c01: 1, c10: -1}))
        # P4 = a11(b10-b00)
        terms.append(({a11: 1}, {b10: 1, b00: -1}, {c00: -1, c10: 1}))
        # P5 = (a00+a01)b11
        terms.append(({a00: 1, a01: 1}, {b11: 1}, {c01: -1, c10: 1}))
        # P6 = (a10-a00)(b00+b01)
        terms.append(({a10: 1, a00: -1}, {b00: 1, b01: 1}, {c00: 1, c10: -1}))
        # P7 = (a01-a11)(b10+b11)
        terms.append(({a01: 1, a11: -1}, {b10: 1, b11: 1}, {c01: 1, c11: -1}))
        
        # Now combine to get c00, c01, c10, c11:
        # c00 = P1 - P4 + P6 - P7
        # c01 = P3 + P1 - P5 + P7
        # c10 = P2 + P4 - P6 + P5
        # c11 = P1 - P3 + P2 - P4
        
        # Wait, I'm confusing myself. The Strassen algorithm gives us 7 products,
        # and then we combine them to get the 4 output entries. But in the tensor
        # decomposition, each "product" is a rank-1 term, and the combination
        # happens via the W matrix coefficients.
        
        # Let me redo this. The correct approach:
        # For Strassen 2x2x2, we have 7 rank-1 terms. Each term contributes
        # to specific output entries with specific coefficients.
        
        # Let me clear and redo.
    
    terms = []
    
    for col_off in range(3):
        a00 = 0
        a01 = 1
        a10 = 3
        a11 = 4
        
        b00 = 0*9 + (col_off+0)
        b01 = 0*9 + (col_off+1)
        b10 = 1*9 + (col_off+0)
        b11 = 1*9 + (col_off+1)
        
        c00 = (col_off+0)*2 + 0
        c01 = (col_off+1)*2 + 0
        c10 = (col_off+0)*2 + 1
        c11 = (col_off+1)*2 + 1
        
        # Strassen 7 terms for 2x2 * 2x2:
        # Term 1: (a00+a11)(b00+b11) -> contributes to c00, c01, c10, c11
        #   c00 += 1, c01 += 1, c10 += 1, c11 += 1
        terms.append(({a00: 1, a11: 1}, {b00: 1, b11: 1}, {c00: 1, c01: 1, c10: 1, c11: 1}))
        
        # Term 2: (a10+a11)b00 -> contributes to c10, c11
        #   c10 += 1, c11 += 1
        terms.append(({a10: 1, a11: 1}, {b00: 1}, {c10: 1, c11: 1}))
        
        # Term 3: a00(b01-b11) -> contributes to c01, c10
        #   c01 += 1, c10 -= 1
        terms.append(({a00: 1}, {b01: 1, b11: -1}, {c01: 1, c10: -1}))
        
        # Term 4: a11(b10-b00) -> contributes to c00, c10
        #   c00 -= 1, c10 += 1
        terms.append(({a11: 1}, {b10: 1, b00: -1}, {c00: -1, c10: 1}))
        
        # Term 5: (a00+a01)b11 -> contributes to c01, c10
        #   c01 -= 1, c10 += 1
        terms.append(({a00: 1, a01: 1}, {b11: 1}, {c01: -1, c10: 1}))
        
        # Term 6: (a10-a00)(b00+b01) -> contributes to c00, c10
        #   c00 += 1, c10 -= 1
        terms.append(({a10: 1, a00: -1}, {b00: 1, b01: 1}, {c00: 1, c10: -1}))
        
        # Term 7: (a01-a11)(b10+b11) -> contributes to c01, c11
        #   c01 += 1, c11 -= 1
        terms.append(({a01: 1, a11: -1}, {b10: 1, b11: 1}, {c01: 1, c11: -1}))
        
        # Now for column 2 of the 2x3 block (B_top[:, 2] = B[:, col_off+2]):
        # Schoolbook: c02 = a00*b02 + a01*b12, c12 = a10*b02 + a11*b12
        b02 = 0*9 + (col_off+2)
        b12 = 1*9 + (col_off+2)
        c02 = (col_off+2)*2 + 0
        c12 = (col_off+2)*2 + 1
        
        # Term 8: a00 * b02 -> c02
        terms.append(({a00: 1}, {b02: 1}, {c02: 1}))
        # Term 9: a01 * b12 -> c02
        terms.append(({a01: 1}, {b12: 1}, {c02: 1}))
        # Term 10: a10 * b02 -> c12
        terms.append(({a10: 1}, {b02: 1}, {c12: 1}))
        # Term 11: a11 * b12 -> c12
        terms.append(({a11: 1}, {b12: 1}, {c12: 1}))
        
        # Now for A_right (col 2 of A) * B_bot (row 2 of B):
        # A_right = A[:, 2] -> U[0*3+2]=U[2], U[1*3+2]=U[5]
        # B_bot = B[2, :] -> V[2*9 + col_off+k] for k=0,1,2
        a20 = 2  # U[0*3+2]
        a21 = 5  # U[1*3+2]
        
        bb0 = 2*9 + (col_off+0)
        bb1 = 2*9 + (col_off+1)
        bb2 = 2*9 + (col_off+2)
        
        # C entries for this part: same c00, c01, c02, c10, c11, c12
        # c00 += a20 * bb0
        terms.append(({a20: 1}, {bb0: 1}, {c00: 1}))
        # c01 += a20 * bb1
        terms.append(({a20: 1}, {bb1: 1}, {c01: 1}))
        # c02 += a20 * bb2
        terms.append(({a20: 1}, {bb2: 1}, {c02: 1}))
        # c10 += a21 * bb0
        terms.append(({a21: 1}, {bb0: 1}, {c10: 1}))
        # c11 += a21 * bb1
        terms.append(({a21: 1}, {bb1: 1}, {c11: 1}))
        # c12 += a21 * bb2
        terms.append(({a21: 1}, {bb2: 1}, {c12: 1}))
    
    # Now build U, V, W matrices from the terms
    rank = len(terms)
    U = [[0] * rank for _ in range(U_rows)]
    V = [[0] * rank for _ in range(V_rows)]
    W = [[0] * rank for _ in range(W_rows)]
    
    for r, (u_dict, v_dict, w_dict) in enumerate(terms):
        for idx, val in u_dict.items():
            U[idx][r] = val
        for idx, val in v_dict.items():
            V[idx][r] = val
        for idx, val in w_dict.items():
            W[idx][r] = val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

