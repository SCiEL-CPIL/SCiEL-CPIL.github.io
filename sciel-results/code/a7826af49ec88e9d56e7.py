"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 49
    
    def A_idx(i, j):
        return i * m + j
    
    def B_idx(i, j):
        return i * m + j
    
    def C_idx(i, k):
        return k * n + i  # W[k*n+i] for C[i,k]
    
    def make_lf(idx, coeff=1):
        return {idx: coeff}
    
    def add_lf(lf1, lf2):
        result = dict(lf1)
        for k, v in lf2.items():
            result[k] = result.get(k, 0) + v
        return result
    
    def sub_lf(lf1, lf2):
        result = dict(lf1)
        for k, v in lf2.items():
            result[k] = result.get(k, 0) - v
        return result
    
    # A block elements
    A11_00 = make_lf(A_idx(0,0)); A11_01 = make_lf(A_idx(0,1))
    A11_10 = make_lf(A_idx(1,0)); A11_11 = make_lf(A_idx(1,1))
    A12_00 = make_lf(A_idx(0,2)); A12_01 = make_lf(A_idx(0,3))
    A12_10 = make_lf(A_idx(1,2)); A12_11 = make_lf(A_idx(1,3))
    A21_00 = make_lf(A_idx(2,0)); A21_01 = make_lf(A_idx(2,1))
    A21_10 = make_lf(A_idx(3,0)); A21_11 = make_lf(A_idx(3,1))
    A22_00 = make_lf(A_idx(2,2)); A22_01 = make_lf(A_idx(2,3))
    A22_10 = make_lf(A_idx(3,2)); A22_11 = make_lf(A_idx(3,3))
    
    B11_00 = make_lf(B_idx(0,0)); B11_01 = make_lf(B_idx(0,1))
    B11_10 = make_lf(B_idx(1,0)); B11_11 = make_lf(B_idx(1,1))
    B12_00 = make_lf(B_idx(0,2)); B12_01 = make_lf(B_idx(0,3))
    B12_10 = make_lf(B_idx(1,2)); B12_11 = make_lf(B_idx(1,3))
    B21_00 = make_lf(B_idx(2,0)); B21_01 = make_lf(B_idx(2,1))
    B21_10 = make_lf(B_idx(3,0)); B21_11 = make_lf(B_idx(3,1))
    B22_00 = make_lf(B_idx(2,2)); B22_01 = make_lf(B_idx(2,3))
    B22_10 = make_lf(B_idx(3,2)); B22_11 = make_lf(B_idx(3,3))
    
    X1 = [[add_lf(A11_00, A22_00), add_lf(A11_01, A22_01)],
           [add_lf(A11_10, A22_10), add_lf(A11_11, A22_11)]]
    Y1 = [[add_lf(B11_00, B22_00), add_lf(B11_01, B22_01)],
           [add_lf(B11_10, B22_10), add_lf(B11_11, B22_11)]]
    
    X2 = [[add_lf(A21_00, A22_00), add_lf(A21_01, A22_01)],
           [add_lf(A21_10, A22_10), add_lf(A21_11, A22_11)]]
    Y2 = [[B11_00, B11_01], [B11_10, B11_11]]
    
    X3 = [[A11_00, A11_01], [A11_10, A11_11]]
    Y3 = [[sub_lf(B12_00, B22_00), sub_lf(B12_01, B22_01)],
           [sub_lf(B12_10, B22_10), sub_lf(B12_11, B22_11)]]
    
    X4 = [[A22_00, A22_01], [A22_10, A22_11]]
    Y4 = [[sub_lf(B21_00, B11_00), sub_lf(B21_01, B11_01)],
           [sub_lf(B21_10, B11_10), sub_lf(B21_11, B11_11)]]
    
    X5 = [[add_lf(A11_00, A12_00), add_lf(A11_01, A12_01)],
           [add_lf(A11_10, A12_10), add_lf(A11_11, A12_11)]]
    Y5 = [[B22_00, B22_01], [B22_10, B22_11]]
    
    X6 = [[sub_lf(A21_00, A11_00), sub_lf(A21_01, A11_01)],
           [sub_lf(A21_10, A11_10), sub_lf(A21_11, A11_11)]]
    Y6 = [[add_lf(B11_00, B12_00), add_lf(B11_01, B12_01)],
           [add_lf(B11_10, B12_10), add_lf(B11_11, B12_11)]]
    
    X7 = [[sub_lf(A12_00, A22_00), sub_lf(A12_01, A22_01)],
           [sub_lf(A12_10, A22_10), sub_lf(A12_11, A22_11)]]
    Y7 = [[add_lf(B21_00, B22_00), add_lf(B21_01, B22_01)],
           [add_lf(B21_10, B22_10), add_lf(B21_11, B22_11)]]
    
    block_products = [(X1,Y1),(X2,Y2),(X3,Y3),(X4,Y4),(X5,Y5),(X6,Y6),(X7,Y7)]
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    for bi, (X, Y) in enumerate(block_products):
        x_s1 = add_lf(X[0][0], X[1][1]); y_s1 = add_lf(Y[0][0], Y[1][1])
        x_s2 = add_lf(X[1][0], X[1][1]); y_s2 = Y[0][0]
        x_s3 = X[0][0]; y_s3 = sub_lf(Y[0][1], Y[1][1])
        x_s4 = X[1][1]; y_s4 = sub_lf(Y[1][0], Y[0][0])
        x_s5 = add_lf(X[0][0], X[0][1]); y_s5 = Y[1][1]
        x_s6 = sub_lf(X[1][0], X[0][0]); y_s6 = add_lf(Y[0][0], Y[0][1])
        x_s7 = sub_lf(X[0][1], X[1][1]); y_s7 = add_lf(Y[1][0], Y[1][1])
        
        sub_products = [(x_s1,y_s1),(x_s2,y_s2),(x_s3,y_s3),(x_s4,y_s4),
                        (x_s5,y_s5),(x_s6,y_s6),(x_s7,y_s7)]
        
        for si, (x_lf, y_lf) in enumerate(sub_products):
            # Mj block entry coefficients for sub-product si
            if si == 0:   # S1
                be = {(0,0):1, (1,1):1}
            elif si == 1: # S2
                be = {(1,0):1, (1,1):-1}
            elif si == 2: # S3
                be = {(0,1):1, (1,1):1}
            elif si == 3: # S4
                be = {(0,0):1, (1,0):1}
            elif si == 4: # S5
                be = {(0,0):-1, (0,1):1}
            elif si == 5: # S6
                be = {(1,1):1}
            else:         # S7
                be = {(0,0):1}
            
            # C block contributions for block product bi
            if bi == 0:   # M1
                cb = {'C11':1, 'C22':1}
            elif bi == 1: # M2
                cb = {'C21':1, 'C22':-1}
            elif bi == 2: # M3
                cb = {'C12':1, 'C22':1}
            elif bi == 3: # M4
                cb = {'C11':1, 'C21':1}
            elif bi == 4: # M5
                cb = {'C11':-1, 'C12':1}
            elif bi == 5: # M6
                cb = {'C22':1}
            else:         # M7
                cb = {'C11':1}
            
            # Set U and V for this column (raw linear forms)
            for a_idx, a_coeff in x_lf.items():
                U[a_idx][col] = a_coeff
            for b_idx, b_coeff in y_lf.items():
                V[b_idx][col] = b_coeff
            
            # Set W: for each Mj block entry and C block, compute C position
            for (mr, mc), m_coeff in be.items():
                for c_block, c_coeff in cb.items():
                    total = m_coeff * c_coeff
                    if c_block == 'C11':
                        i, k = mr, mc
                    elif c_block == 'C12':
                        i, k = mr, 2+mc
                    elif c_block == 'C21':
                        i, k = 2+mr, mc
                    else:
                        i, k = 2+mr, 2+mc
                    W[C_idx(i, k)][col] += total
            
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
