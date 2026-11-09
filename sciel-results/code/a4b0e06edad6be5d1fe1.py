import time
import sys

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # n=3, m=4, p=7
    # U: 12 x rank
    # V: 28 x rank
    # W: 21 x rank
    
    # We will collect terms as (u_indices, v_indices, w_indices)
    # u_indices: list of (idx, coeff)
    # v_indices: list of (idx, coeff)
    # w_indices: list of (idx, coeff)
    
    terms = []
    
    def add_strassen_block(a_row_offset, a_col_offset, b_row_offset, b_col_offset):
        # A block is 2x2 at (a_row_offset, a_col_offset)
        # B block is 2x2 at (b_row_offset, b_col_offset)
        # C block is 2x2 at (a_row_offset, b_col_offset)
        
        # A indices
        a00 = a_row_offset * m + a_col_offset
        a01 = a_row_offset * m + a_col_offset + 1
        a10 = (a_row_offset + 1) * m + a_col_offset
        a11 = (a_row_offset + 1) * m + a_col_offset + 1
        
        # B indices
        b00 = b_row_offset * p + b_col_offset
        b01 = b_row_offset * p + b_col_offset + 1
        b10 = (b_row_offset + 1) * p + b_col_offset
        b11 = (b_row_offset + 1) * p + b_col_offset + 1
        
        # C indices (W[k*n + i])
        c00 = b_col_offset * n + a_row_offset
        c01 = (b_col_offset + 1) * n + a_row_offset
        c10 = b_col_offset * n + (a_row_offset + 1)
        c11 = (b_col_offset + 1) * n + (a_row_offset + 1)
        
        # Strassen 7 multiplications
        # m1 = (a00 + a11)(b00 + b11)
        terms.append(([ (a00, 1), (a11, 1) ], [ (b00, 1), (b11, 1) ], [ (c00, 1), (c01, -1), (c10, -1), (c11, 1) ]))
        # Wait, the combination for C is:
        # c00 = m1 - m5 + m7
        # c01 = m3 + m5
        # c10 = m2 + m4
        # c11 = m1 - m2 + m6
        
        # Let's define the 7 terms properly.
        # m1: U: a00, a11. V: b00, b11. W: c00(+1), c01(-1), c10(-1), c11(+1)
        # m2: U: a10, a11. V: b00. W: c10(+1), c11(-1)
        # m3: U: a00. V: b01, b11. W: c01(+1), c11(-1) -> No, m3 = a00(b01-b11). W: c01(+1), c11(-1)?
        # Let's re-verify Strassen output formulas.
        # c00 = m1 - m5 + m7
        # c01 = m3 + m5
        # c10 = m2 + m4
        # c11 = m1 - m2 + m6
        
        # m1 = (a00+a11)(b00+b11)
        # m2 = (a10+a11)b00
        # m3 = a00(b01-b11)
        # m4 = a11(b10-b00)
        # m5 = (a00+a01)b11
        # m6 = (a10-a00)(b00+b01)
        # m7 = (a01-a11)(b10+b11)
        
        # I will clear the previous append and do it correctly.
        terms.pop() # Remove the incorrect one
        
        # m1
        terms.append(([ (a00, 1), (a11, 1) ], [ (b00, 1), (b11, 1) ], [ (c00, 1), (c01, 0), (c10, 0), (c11, 0) ])) # Placeholder, will fix W coeffs
        # Actually, it's easier to just add the 7 terms with their specific U, V, W contributions.
        
        # Let's restart the term addition logic inside the function properly.
        
    # Redefining add_strassen_block to be correct
    def add_strassen_block_correct(a_row_offset, a_col_offset, b_row_offset, b_col_offset):
        a00 = a_row_offset * m + a_col_offset
        a01 = a_row_offset * m + a_col_offset + 1
        a10 = (a_row_offset + 1) * m + a_col_offset
        a11 = (a_row_offset + 1) * m + a_col_offset + 1
        
        b00 = b_row_offset * p + b_col_offset
        b01 = b_row_offset * p + b_col_offset + 1
        b10 = (b_row_offset + 1) * p + b_col_offset
        b11 = (b_row_offset + 1) * p + b_col_offset + 1
        
        c00 = b_col_offset * n + a_row_offset
        c01 = (b_col_offset + 1) * n + a_row_offset
        c10 = b_col_offset * n + (a_row_offset + 1)
        c11 = (b_col_offset + 1) * n + (a_row_offset + 1)
        
        # m1 = (a00 + a11)(b00 + b11)
        # Contribution to C: c00 += m1, c11 += m1
        terms.append(([ (a00, 1), (a11, 1) ], [ (b00, 1), (b11, 1) ], [ (c00, 1), (c11, 1) ]))
        
        # m2 = (a10 + a11)b00
        # Contribution to C: c10 += m2, c11 -= m2
        terms.append(([ (a10, 1), (a11, 1) ], [ (b00, 1) ], [ (c10, 1), (c11, -1) ]))
        
        # m3 = a00(b01 - b11)
        # Contribution to C: c01 += m3, c11 -= m3
        terms.append(([ (a00, 1) ], [ (b01, 1), (b11, -1) ], [ (c01, 1), (c11, -1) ]))
        
        # m4 = a11(b10 - b00)
        # Contribution to C: c10 += m4, c11 -= m4? No.
        # c10 = m2 + m4. c11 = m1 - m2 + m6.
        # Wait, m4 is in c10 only?
        # Let's check:
        # c10 = (a10+a11)b00 + a11(b10-b00) = a10b00 + a11b00 + a11b10 - a11b00 = a10b00 + a11b10. Correct.
        # So m4 contributes to c10 with +1.
        terms.append(([ (a11, 1) ], [ (b10, 1), (b00, -1) ], [ (c10, 1) ]))
        
        # m5 = (a00 + a01)b11
        # Contribution to C: c00 -= m5, c01 += m5
        terms.append(([ (a00, 1), (a01, 1) ], [ (b11, 1) ], [ (c00, -1), (c01, 1) ]))
        
        # m6 = (a10 - a00)(b00 + b01)
        # Contribution to C: c11 += m6
        terms.append(([ (a10, 1), (a00, -1) ], [ (b00, 1), (b01, 1) ], [ (c11, 1) ]))
        
        # m7 = (a01 - a11)(b10 + b11)
        # Contribution to C: c00 += m7
        terms.append(([ (a01, 1), (a11, -1) ], [ (b10, 1), (b11, 1) ], [ (c00, 1) ]))
        
    def add_schoolbook_221(a_row_offset, a_col_offset, b_row_offset, b_col_offset):
        # A is 2x2, B is 2x1.
        a00 = a_row_offset * m + a_col_offset
        a01 = a_row_offset * m + a_col_offset + 1
        a10 = (a_row_offset + 1) * m + a_col_offset
        a11 = (a_row_offset + 1) * m + a_col_offset + 1
        
        b00 = b_row_offset * p + b_col_offset
        b10 = (b_row_offset + 1) * p + b_col_offset
        
        c00 = b_col_offset * n + a_row_offset
        c10 = b_col_offset * n + (a_row_offset + 1)
        
        # m1 = a00 b00 -> c00
        terms.append(([ (a00, 1) ], [ (b00, 1) ], [ (c00, 1) ]))
        # m2 = a01 b10 -> c00
        terms.append(([ (a01, 1) ], [ (b10, 1) ], [ (c00, 1) ]))
        # m3 = a10 b00 -> c10
        terms.append(([ (a10, 1) ], [ (b00, 1) ], [ (c10, 1) ]))
        # m4 = a11 b10 -> c10
        terms.append(([ (a11, 1) ], [ (b10, 1) ], [ (c10, 1) ]))
        
    def add_1x2x7(a_row_offset, a_col_offset, b_row_offset, b_col_offset):
        # A is 1x2, B is 2x7.
        # a00, a01
        # b00..b06, b10..b16
        # c0..c6
        a00 = a_row_offset * m + a_col_offset
        a01 = a_row_offset * m + a_col_offset + 1
        
        for k in range(7):
            b0k = b_row_offset * p + b_col_offset + k
            b1k = (b_row_offset + 1) * p + b_col_offset + k
            ck = (b_col_offset + k) * n + a_row_offset
            
            # m1 = a00 b0k -> ck
            terms.append(([ (a00, 1) ], [ (b0k, 1) ], [ (ck, 1) ]))
            # m2 = a01 b1k -> ck
            terms.append(([ (a01, 1) ], [ (b1k, 1) ], [ (ck, 1) ]))
            
    # Part 1: A_L B_T
    # A_L: rows 0-2, cols 0-1
    # B_T: rows 0-1, cols 0-6
    
    # 1a: A[0:2, 0:2] * B[0:2, 0:6]
    # Split B cols into 0-1, 2-3, 4-5, 6-6
    add_strassen_block_correct(0, 0, 0, 0) # A[0:2,0:2] * B[0:2,0:2]
    add_strassen_block_correct(0, 0, 0, 2) # A[0:2,0:2] * B[0:2,2:4]
    add_strassen_block_correct(0, 0, 0, 4) # A[0:2,0:2] * B[0:2,4:6]
    add_schoolbook_221(0, 0, 0, 6)         # A[0:2,0:2] * B[0:2,6:7]
    
    # 1c: A[2:3, 0:2] * B[0:2, 0:7]
    add_1x2x7(2, 0, 0, 0)
    
    # Part 2: A_R B_B
    # A_R: rows 0-2, cols 2-3
    # B_B: rows 2-3, cols 0-6
    
    # 2a: A[0:2, 2:4] * B[2:4, 0:6]
    add_strassen_block_correct(0, 2, 2, 0) # A[0:2,2:4] * B[2:4,0:2]
    add_strassen_block_correct(0, 2, 2, 2) # A[0:2,2:4] * B[2:4,2:4]
    add_strassen_block_correct(0, 2, 2, 4) # A[0:2,2:4] * B[2:4,4:6]
    add_schoolbook_221(0, 2, 2, 6)         # A[0:2,2:4] * B[2:4,6:7]
    
    # 2c: A[2:3, 2:4] * B[2:4, 0:7]
    add_1x2x7(2, 2, 2, 0)
    
    rank = len(terms)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col_idx, (u_list, v_list, w_list) in enumerate(terms):
        for idx, coeff in u_list:
            U[idx][col_idx] = coeff
        for idx, coeff in v_list:
            V[idx][col_idx] = coeff
        for idx, coeff in w_list:
            W[idx][col_idx] = coeff
            
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print(f"Rank: {len(res['U'][0])}")
