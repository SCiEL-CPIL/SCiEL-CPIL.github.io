import time

DIMENSIONS = [2, 2, 11]

def construct():
    n, m, p = DIMENSIONS
    # Rank 39
    R = 39
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B00)
    # M4 = A11(B01-B11)
    # M5 = (A00+A10)B11
    # M6 = (A11-A01)(B00+B10)
    # M7 = (A01-A10)(B01+B11)
    
    # A factors (indices in U: i*m+j)
    # A00: 0, A01: 1, A10: 2, A11: 3
    a_factors = [
        [(0,1), (3,1)],      # M1: A00+A11
        [(2,1), (3,1)],      # M2: A10+A11
        [(0,1)],             # M3: A00
        [(3,1)],             # M4: A11
        [(0,1), (2,1)],      # M5: A00+A10
        [(3,1), (1,-1)],     # M6: A11-A01
        [(1,1), (2,-1)]      # M7: A01-A10
    ]
    
    # B factors for a block (indices in V: j*p+k)
    # B00: 0, B01: 1, B10: 2, B11: 3 (relative to block start)
    # V index: j*p + k.
    # Block k starts at col 2k.
    # B00 is j=0, k=2k. V[0*p + 2k]
    # B01 is j=0, k=2k+1. V[0*p + 2k+1]
    # B10 is j=1, k=2k. V[1*p + 2k]
    # B11 is j=1, k=2k+1. V[1*p + 2k+1]
    
    b_factors_rel = [
        [(0, 0, 1), (1, 1, 1)],    # M1: B00+B11 (j=0,k=0; j=1,k=1)
        [(0, 0, 1)],               # M2: B00
        [(1, 0, 1), (0, 0, -1)],   # M3: B10-B00 (j=1,k=0; j=0,k=0)
        [(0, 1, 1), (1, 1, -1)],   # M4: B01-B11 (j=0,k=1; j=1,k=1)
        [(1, 1, 1)],               # M5: B11
        [(0, 0, 1), (1, 0, 1)],    # M6: B00+B10
        [(0, 1, 1), (1, 1, 1)]     # M7: B01+B11
    ]
    
    # W reconstruction
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    w_coeffs = [
        [1, 0, 1, 0],  # M1: C00=1, C01=0, C10=0, C11=1 ? No.
        # Let's map:
        # M1: C00+=1, C11+=1
        # M2: C10+=1, C11+=-1
        # M3: C01+=1, C11+=1
        # M4: C00+=1, C10+=1
        # M5: C00+=-1, C01+=1
        # M6: C11+=1
        # M7: C00+=1
    ]
    
    # Correct mapping:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    w_map = [
        [1, 0, 0, 1], # M1: C00, C01, C10, C11
        [0, 0, 1, -1], # M2
        [0, 1, 0, 1],  # M3
        [1, 0, 1, 0],  # M4
        [-1, 1, 0, 0], # M5
        [0, 0, 0, 1],  # M6
        [1, 0, 0, 0]   # M7
    ]
    
    col = 0
    for k in range(5):
        # 5 Strassen blocks
        for m_idx in range(7):
            # U
            for (u_idx, coeff) in a_factors[m_idx]:
                U[u_idx][col] += coeff
            
            # V
            for (j, k_rel, coeff) in b_factors_rel[m_idx]:
                v_idx = j * p + (2*k + k_rel)
                V[v_idx][col] += coeff
            
            # W
            # Output cols: 2k, 2k+1
            # W index: k_out * n + i
            # C00 -> i=0, k=2k -> W[2k*2 + 0]
            # C01 -> i=0, k=2k+1 -> W[(2k+1)*2 + 0]
            # C10 -> i=1, k=2k -> W[2k*2 + 1]
            # C11 -> i=1, k=2k+1 -> W[(2k+1)*2 + 1]
            
            w_indices = [
                (2*k)*2 + 0,
                (2*k+1)*2 + 0,
                (2*k)*2 + 1,
                (2*k+1)*2 + 1
            ]
            
            for w_idx, coeff in zip(w_indices, w_map[m_idx]):
                if coeff != 0:
                    W[w_idx][col] += coeff
            
            col += 1
            
    # Last block: 2x1
    # C0,10 = A00 B0,10 + A01 B1,10
    # C1,10 = A10 B0,10 + A11 B1,10
    
    # N1: A00 * B0,10
    U[0][col] = 1
    V[0*p + 10][col] = 1
    W[10*2 + 0][col] = 1 # C0,10
    col += 1
    
    # N2: A01 * B1,10
    U[1][col] = 1
    V[1*p + 10][col] = 1
    W[10*2 + 0][col] = 1 # C0,10
    col += 1
    
    # N3: A10 * B0,10
    U[2][col] = 1
    V[0*p + 10][col] = 1
    W[10*2 + 1][col] = 1 # C1,10
    col += 1
    
    # N4: A11 * B1,10
    U[3][col] = 1
    V[1*p + 10][col] = 1
    W[10*2 + 1][col] = 1 # C1,10
    col += 1
    
    return {"U": U, "V": V, "W": W}
