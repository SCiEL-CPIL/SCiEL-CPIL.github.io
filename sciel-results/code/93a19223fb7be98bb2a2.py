# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS  # 2, 4, 4
    
    rank = 28
    U = [[0] * rank for _ in range(n * m)]  # (8, 28)
    V = [[0] * rank for _ in range(m * p)]  # (16, 28)
    W = [[0] * rank for _ in range(p * n)]  # (8, 28)
    
    # Strassen decomposition for 2x2x2
    # Inputs: a[4], b[4]
    # Outputs: c[4]
    # 7 multiplications
    # p1 = (a0 + a3)(b0 + b3)
    # p2 = (a2 + a3)b0
    # p3 = a0(b2 - b3)
    # p4 = a3(b1 + b3)
    # p5 = (a0 + a1)b3
    # p6 = (a2 - a0)(b0 + b1)
    # p7 = (a1 - a3)(b2 + b3)
    
    # c0 = p1 - p4 + p5 - p7  (Wait, let's verify this mapping)
    # Standard Strassen:
    # c00 = p1 - p4 + p5 - p7 ? No.
    # Let's use the verified formulas:
    # c00 = p1 + p4 - p5 + p7 is wrong.
    # Correct:
    # c00 = p1 - p4 + p5 - p7 ?
    
    # Let's just use the standard set:
    # p1 = (a00+a11)(b00+b11)
    # p2 = (a10+a11)b00
    # p3 = a00(b10-b11)
    # p4 = a11(b01+b11)
    # p5 = (a00+a01)b11
    # p6 = (a10-a00)(b00+b01)
    # p7 = (a01-a11)(b10+b11)
    
    # c00 = p1 - p4 + p5 - p7  -> Check:
    # p1 = a00b00 + a00b11 + a11b00 + a11b11
    # -p4 = -a11b01 - a11b11
    # +p5 = a00b11 + a01b11
    # -p7 = -a01b10 - a01b11 + a11b10 + a11b11
    # Sum: a00b00 + (a00b11-a00b11) + a11b00 + (a11b11-a11b11+a11b11) - a11b01 + a01b11 - a01b10 - a01b11 + a11b10
    # = a00b00 + a11b00 + a11b11 - a11b01 + a11b10 - a01b10
    # This is not a00b00 + a01b10.
    
    # Okay, I will hardcode the correct Strassen coefficients.
    # From Wikipedia:
    # p1 = (a11+a22)(b11+b22)
    # p2 = (a21+a22)b11
    # p3 = a11(b21-b22)
    # p4 = a22(b12+b22)
    # p5 = (a11+a12)b22
    # p6 = (a21-a11)(b11+b12)
    # p7 = (a12-a22)(b21+b22)
    
    # c11 = p1 + p4 - p5 + p7
    # c12 = p3 + p5
    # c21 = p2 + p4
    # c22 = p1 - p2 + p3 + p6
    
    # Mapping 1-based to 0-based:
    # a11->a00, a12->a01, a21->a10, a22->a11
    # b11->b00, b12->b01, b21->b10, b22->b11
    # c11->c00, c12->c01, c21->c10, c22->c11
    
    # p1 = (a00+a11)(b00+b11)
    # p2 = (a10+a11)b00
    # p3 = a00(b10-b11)
    # p4 = a11(b01+b11)
    # p5 = (a00+a01)b11
    # p6 = (a10-a00)(b00+b01)
    # p7 = (a01-a11)(b10+b11)
    
    # c00 = p1 + p4 - p5 + p7
    # c01 = p3 + p5
    # c10 = p2 + p4
    # c11 = p1 - p2 + p3 + p6
    
    def add_strassen_block(a_idx, b_idx, c_idx, col_offset):
        # a_idx: list of 4 global indices for A
        # b_idx: list of 4 global indices for B
        # c_idx: list of 4 global indices for C
        # col_offset: starting column in U, V, W
        
        # p1
        U[a_idx[0]][col_offset] += 1
        U[a_idx[3]][col_offset] += 1
        V[b_idx[0]][col_offset] += 1
        V[b_idx[3]][col_offset] += 1
        W[c_idx[0]][col_offset] += 1
        W[c_idx[3]][col_offset] += 1
        
        # p2
        U[a_idx[2]][col_offset+1] += 1
        U[a_idx[3]][col_offset+1] += 1
        V[b_idx[0]][col_offset+1] += 1
        W[c_idx[2]][col_offset+1] += 1
        W[c_idx[3]][col_offset+1] -= 1
        
        # p3
        U[a_idx[0]][col_offset+2] += 1
        V[b_idx[2]][col_offset+2] += 1
        V[b_idx[3]][col_offset+2] -= 1
        W[c_idx[1]][col_offset+2] += 1
        W[c_idx[3]][col_offset+2] += 1
        
        # p4
        U[a_idx[3]][col_offset+3] += 1
        V[b_idx[1]][col_offset+3] += 1
        V[b_idx[3]][col_offset+3] += 1
        W[c_idx[0]][col_offset+3] += 1
        W[c_idx[2]][col_offset+3] += 1
        
        # p5
        U[a_idx[0]][col_offset+4] += 1
        U[a_idx[1]][col_offset+4] += 1
        V[b_idx[3]][col_offset+4] += 1
        W[c_idx[0]][col_offset+4] -= 1
        W[c_idx[1]][col_offset+4] += 1
        
        # p6
        U[a_idx[2]][col_offset+5] += 1
        U[a_idx[0]][col_offset+5] -= 1
        V[b_idx[0]][col_offset+5] += 1
        V[b_idx[1]][col_offset+5] += 1
        W[c_idx[3]][col_offset+5] += 1
        
        # p7
        U[a_idx[1]][col_offset+6] += 1
        U[a_idx[3]][col_offset+6] -= 1
        V[b_idx[2]][col_offset+6] += 1
        V[b_idx[3]][col_offset+6] += 1
        W[c_idx[0]][col_offset+6] += 1

    # Global indices:
    # A: i*4 + j
    # B: j*4 + k
    # C: k*2 + i (Task: W[k*n+i, r])
    
    # Block 1: A1 (cols 0,1) x B1a (rows 0,1, cols 0,1) -> C (cols 0,1, rows 0,1)
    # A1: a00, a01, a10, a11 -> indices 0, 1, 4, 5
    # B1a: b00, b01, b10, b11 -> indices 0, 1, 4, 5
    # C: c00, c01, c10, c11 -> indices 0, 1, 2, 3 (k*2+i)
    # c00: k=0, i=0 -> 0
    # c01: k=1, i=0 -> 2? No.
    # Task: W[k*n+i, r]. n=2.
    # c00: k=0, i=0 -> 0*2+0=0
    # c01: k=1, i=0 -> 1*2+0=2
    # c10: k=0, i=1 -> 0*2+1=1
    # c11: k=1, i=1 -> 1*2+1=3
    
    # Wait, C is 2x4.
    # C[i,k].
    # c00 is C[0,0]. k=0, i=0. Index 0.
    # c01 is C[0,1]. k=1, i=0. Index 2.
    # c10 is C[1,0]. k=0, i=1. Index 1.
    # c11 is C[1,1]. k=1, i=1. Index 3.
    
    # So c_idx for block 1 (cols 0,1) is [0, 2, 1, 3].
    
    a1 = [0, 1, 4, 5]
    b1a = [0, 1, 4, 5]
    c1 = [0, 2, 1, 3]
    add_strassen_block(a1, b1a, c1, 0)
    
    # Block 2: A1 x B1b (rows 0,1, cols 2,3) -> C (cols 2,3)
    # B1b: b02, b03, b12, b13 -> indices 2, 3, 6, 7
    # C: c02, c03, c12, c13
    # c02: k=2, i=0 -> 4
    # c03: k=3, i=0 -> 6
    # c12: k=2, i=1 -> 5
    # c13: k=3, i=1 -> 7
    # c_idx: [4, 6, 5, 7]
    
    b1b = [2, 3, 6, 7]
    c2 = [4, 6, 5, 7]
    add_strassen_block(a1, b1b, c2, 7)
    
    # Block 3: A2 (cols 2,3) x B2a (rows 2,3, cols 0,1) -> C (cols 0,1)
    # A2: a02, a03, a12, a13 -> indices 2, 3, 6, 7
    # B2a: b20, b21, b30, b31 -> indices 8, 9, 12, 13
    # C: c00, c01, c10, c11 -> indices 0, 2, 1, 3
    
    a2 = [2, 3, 6, 7]
    b2a = [8, 9, 12, 13]
    c3 = [0, 2, 1, 3]
    add_strassen_block(a2, b2a, c3, 14)
    
    # Block 4: A2 x B2b (rows 2,3, cols 2,3) -> C (cols 2,3)
    # B2b: b22, b23, b32, b33 -> indices 10, 11, 14, 15
    # C: c02, c03, c12, c13 -> indices 4, 6, 5, 7
    
    b2b = [10, 11, 14, 15]
    c4 = [4, 6, 5, 7]
    add_strassen_block(a2, b2b, c4, 21)
    
    # Verify
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for r in range(rank):
                    total += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(total - 1) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
