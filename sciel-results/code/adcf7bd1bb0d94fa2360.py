import time

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen decomposition for 2x2 matrices
    # X, Y are 2x2. Z = X Y.
    # M1 = (X00 - X10)(Y10 + Y11)
    # M2 = (X00 + X10)Y11
    # M3 = (X00 + X01)Y00
    # M4 = (X11 - X00)(Y00 + Y01)
    # M5 = (X00 + X11)(Y11 - Y01)
    # M6 = (X01 - X11)(Y00 + Y10)
    # M7 = (X10 - X00)(Y01 + Y11)
    
    # Z00 = M1 + M4 - M5 + M7
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M2 + M3 + M6
    
    # Each term t adds to W:
    # t=0 (M1): Z00(+1), Z11(+1)
    # t=1 (M2): Z10(+1), Z11(-1)
    # t=2 (M3): Z01(+1), Z11(+1)
    # t=3 (M4): Z00(+1), Z10(+1)
    # t=4 (M5): Z00(-1), Z01(+1)
    # t=5 (M6): Z11(+1)
    # t=6 (M7): Z00(+1)
    
    # Wait, let's re-verify the Z mapping.
    # Z00: M1, M4, -M5, M7
    # Z01: M3, M5
    # Z10: M2, M4
    # Z11: M1, -M2, M3, M6
    
    # So for each t, we know which Z entries it affects and with what coefficient.
    
    strassen_terms = [
        # (u_idx, u_coeff, v_idx, v_coeff, w_idx, w_coeff)
        # t=0: M1
        # U: X00(1), X10(-1)
        # V: Y10(1), Y11(1)
        # W: Z00(1), Z11(1)
        [(0, 1, 2, 1, 0, 1, 3, 1)], # This is not a tuple structure, let's use lists of (idx, coeff)
        # Actually, easier to just define the 7 terms explicitly.
    ]
    
    # Let's define the 7 terms as lists of (local_row, local_col, coeff) for X, Y, Z
    # X local indices: 00, 01, 10, 11
    # Y local indices: 00, 01, 10, 11
    # Z local indices: 00, 01, 10, 11
    
    terms = []
    # M1
    terms.append({
        'X': [(0, 0, 1), (1, 0, -1)],
        'Y': [(1, 0, 1), (1, 1, 1)],
        'Z': [(0, 0, 1), (1, 1, 1)]
    })
    # M2
    terms.append({
        'X': [(0, 0, 1), (1, 0, 1)],
        'Y': [(1, 1, 1)],
        'Z': [(1, 0, 1), (1, 1, -1)]
    })
    # M3
    terms.append({
        'X': [(0, 0, 1), (0, 1, 1)],
        'Y': [(0, 0, 1)],
        'Z': [(0, 1, 1), (1, 1, 1)]
    })
    # M4
    terms.append({
        'X': [(1, 1, 1), (0, 0, -1)],
        'Y': [(0, 0, 1), (0, 1, 1)],
        'Z': [(0, 0, 1), (1, 0, 1)]
    })
    # M5
    terms.append({
        'X': [(0, 0, 1), (1, 1, 1)],
        'Y': [(1, 1, 1), (0, 1, -1)],
        'Z': [(0, 0, -1), (0, 1, 1)]
    })
    # M6
    terms.append({
        'X': [(0, 1, 1), (1, 1, -1)],
        'Y': [(0, 0, 1), (1, 0, 1)],
        'Z': [(1, 1, 1)]
    })
    # M7
    terms.append({
        'X': [(1, 0, 1), (0, 0, -1)],
        'Y': [(0, 1, 1), (1, 1, 1)],
        'Z': [(0, 0, 1)]
    })
    
    for c in range(3):
        # Block c corresponds to columns 2c, 2c+1 of B and C
        k_start = 2 * c
        
        # Part 1: A_L (cols 0,1) * B_{c,T} (rows 0,1)
        # X = A_L: i in 0..1, j in 0..1
        # Y = B_{c,T}: j in 0..1, k in k_start..k_start+1
        # Z: i in 0..1, k in k_start..k_start+1
        
        for t in range(7):
            # U (X)
            for (ri, ci, coeff) in terms[t]['X']:
                # Global A index: i * 4 + j
                # ri is i, ci is j
                u_idx = ri * 4 + ci
                U[u_idx][col] += coeff
                
            # V (Y)
            for (ri, ci, coeff) in terms[t]['Y']:
                # Global B index: j * 6 + k
                # ri is j, ci is k - k_start
                v_idx = ri * 6 + (k_start + ci)
                V[v_idx][col] += coeff
                
            # W (Z)
            for (ri, ci, coeff) in terms[t]['Z']:
                # Global C index: k * 2 + i
                # ri is k - k_start, ci is i
                w_idx = (k_start + ri) * 2 + ci
                W[w_idx][col] += coeff
                
            col += 1
            
        # Part 2: A_R (cols 2,3) * B_{c,B} (rows 2,3)
        # X = A_R: i in 0..1, j in 2..3
        # Y = B_{c,B}: j in 2..3, k in k_start..k_start+1
        # Z: i in 0..1, k in k_start..k_start+1
        
        for t in range(7):
            # U (X)
            for (ri, ci, coeff) in terms[t]['X']:
                # Global A index: i * 4 + j
                # ri is i, ci is j - 2
                u_idx = ri * 4 + (2 + ci)
                U[u_idx][col] += coeff
                
            # V (Y)
            for (ri, ci, coeff) in terms[t]['Y']:
                # Global B index: j * 6 + k
                # ri is j - 2, ci is k - k_start
                v_idx = (2 + ri) * 6 + (k_start + ci)
                V[v_idx][col] += coeff
                
            # W (Z)
            for (ri, ci, coeff) in terms[t]['Z']:
                # Global C index: k * 2 + i
                # ri is k - k_start, ci is i
                w_idx = (k_start + ri) * 2 + ci
                W[w_idx][col] += coeff
                
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
