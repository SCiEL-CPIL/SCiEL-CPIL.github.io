"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 33
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # 1. Core parts: 3 blocks of 2x2 by 2x2 using Strassen
    # A indices: i in {0,1}, j in {0,1}
    # B indices: j in {0,1}, k in {2l, 2l+1} for l in {0,1,2}
    # C indices: k in {2l, 2l+1}, i in {0,1}
    
    strassen_coeffs = [
        # M1: (a00 + a11)(b00 + b11)
        # U: a00, a11 -> 1, 1
        # V: b00, b11 -> 1, 1
        # W: c00 (1), c11 (1), c01 (-1), c10 (-1) ? No, W is for output.
        # We need to map M_r to C_k,i.
        # c00 = M1 - M4 + M6
        # c01 = M3 + M4
        # c10 = M2 + M7
        # c11 = M1 - M3 + M5
        # So for each M_r, we add to W with coefficients.
        
        # M1
        {'u': [(0, 1), (3, 1)], 'v': [(0, 1), (6, 1)], 'w': [(0, 1), (2, 1), (1, -1), (3, -1)]}, # c00=k0,i0; c11=k1,i1; c01=k0,i1; c10=k1,i0
        # Wait, W index is k*n + i.
        # c00: k=0, i=0 -> 0
        # c01: k=0, i=1 -> 1
        # c10: k=1, i=0 -> 2
        # c11: k=1, i=1 -> 3
        
        # M1 contributes to c00 (1), c11 (1), c01 (-1), c10 (-1)
        # M2: (a10 + a11)b00 -> U: a10(1), a11(1); V: b00(1); W: c10(1)
        {'u': [(3, 1), (4, 1)], 'v': [(0, 1)], 'w': [(2, 1)]},
        # M3: a00(b10 + b11) -> U: a00(1); V: b10(1), b11(1); W: c01(1), c11(-1)
        {'u': [(0, 1)], 'v': [(6, 1), (7, 1)], 'w': [(1, 1), (3, -1)]},
        # M4: a11 b10 -> U: a11(1); V: b10(1); W: c00(-1), c01(1)
        {'u': [(4, 1)], 'v': [(6, 1)], 'w': [(0, -1), (1, 1)]},
        # M5: (a00 + a01)b11 -> U: a00(1), a01(1); V: b11(1); W: c11(1)
        {'u': [(0, 1), (1, 1)], 'v': [(7, 1)], 'w': [(3, 1)]},
        # M6: (a10 - a00)(b00 + b10) -> U: a10(1), a00(-1); V: b00(1), b10(1); W: c00(1)
        {'u': [(3, 1), (0, -1)], 'v': [(0, 1), (6, 1)], 'w': [(0, 1)]},
        # M7: (a01 - a11)(b01 + b11) -> U: a01(1), a11(-1); V: b01(1), b11(1); W: c10(1)
        {'u': [(1, 1), (4, -1)], 'v': [(1, 1), (7, 1)], 'w': [(2, 1)]},
    ]
    
    # Note: U index for A[i,j] is i*m+j = 3i+j.
    # a00: 0, a01: 1, a10: 3, a11: 4.
    # V index for B[j,k] is j*p+k = 6j+k.
    # b00: 0, b01: 1, b10: 6, b11: 7.
    # W index for C[k,i] is k*n+i = 2k+i.
    
    for l in range(3):
        # Offset for B columns: 2l
        # b00 -> 2l, b01 -> 2l+1
        # b10 -> 6+2l, b11 -> 7+2l
        # C rows: 2l, 2l+1
        # c00 -> 2*(2l)+0 = 4l
        # c01 -> 2*(2l)+1 = 4l+1
        # c10 -> 2*(2l+1)+0 = 4l+2
        # c11 -> 2*(2l+1)+1 = 4l+3
        
        for term in strassen_coeffs:
            for idx, val in term['u']:
                U[idx][col] = val
            for idx, val in term['v']:
                # Adjust B index by 2l
                # Original idx 0 (b00) -> 2l
                # Original idx 1 (b01) -> 2l+1
                # Original idx 6 (b10) -> 6+2l
                # Original idx 7 (b11) -> 7+2l
                new_idx = idx + 2*l if idx < 2 else idx + 2*l
                V[new_idx][col] = val
            for idx, val in term['w']:
                # Adjust C index by 4l
                new_idx = idx + 4*l
                W[new_idx][col] = val
            col += 1
            
    # 2. Extra part: 2x1 by 1x6
    # A indices: i in {0,1}, j=2 -> 3i+2 (2, 5)
    # B indices: j=2, k in 0..5 -> 12+k (12..17)
    # C indices: k in 0..5, i in {0,1} -> 2k+i
    
    for k in range(6):
        for i in range(2):
            U[3*i + 2][col] = 1
            V[12 + k][col] = 1
            W[2*k + i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
