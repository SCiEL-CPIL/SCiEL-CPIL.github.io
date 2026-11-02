"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    # Strassen-based decomposition: 2x2 * 2x14 = 7 blocks of 2x2 * 2x2
    # Each 2x2*2x2 uses Strassen (rank 7), total rank = 7*7 = 49
    n, m, p = DIMENSIONS  # n=2, m=2, p=14
    num_blocks = p // 2  # 7 blocks of 2 columns
    rank = num_blocks * 7  # 49
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen products for C = A*B (2x2 * 2x2):
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # M1=(a+d)(e+h), M2=(c+d)e, M3=a(f-h), M4=d(g-e), M5=(a+b)h, M6=(c-a)(e+f), M7=(b-d)(g+h)
    # C11=M1+M4-M5+M7, C12=M3+M5, C21=M2+M4, C22=M1-M2+M3+M6
    
    # For each product: u_coeffs on A, v_coeffs on B, w_coeffs on C
    # A indices: 0=a, 1=b, 2=c, 3=d
    # B local: 0=e, 1=f, 2=g, 3=h
    # C local: 0=C11, 1=C12, 2=C21, 3=C22
    
    strassen = [
        # M1
        {'u': {0:1, 3:1}, 'v': {0:1, 3:1}, 'w': {0:1, 3:1}},
        # M2
        {'u': {2:1, 3:1}, 'v': {0:1}, 'w': {2:1, 3:-1}},
        # M3
        {'u': {0:1}, 'v': {1:1, 3:-1}, 'w': {1:1, 3:1}},
        # M4
        {'u': {3:1}, 'v': {2:1, 0:-1}, 'w': {0:1, 2:1}},
        # M5
        {'u': {0:1, 1:1}, 'v': {3:1}, 'w': {0:-1, 1:1}},
        # M6
        {'u': {2:1, 0:-1}, 'v': {0:1, 1:1}, 'w': {3:1}},
        # M7
        {'u': {1:1, 3:-1}, 'v': {2:1, 3:1}, 'w': {0:1}},
    ]
    
    col = 0
    for block in range(num_blocks):
        for prod in strassen:
            # Fill U: A index = i*m + j (row i, col j of 2x2 A)
            for a_idx, coeff in prod['u'].items():
                U[a_idx][col] += coeff
            
            # Fill V: B index = j*p + k (row j, col k of 2x14 B)
            # Local B: e=(0,2b), f=(0,2b+1), g=(1,2b), h=(1,2b+1)
            b_map = {0: (0, 2*block), 1: (0, 2*block+1), 2: (1, 2*block), 3: (1, 2*block+1)}
            for b_idx, coeff in prod['v'].items():
                br, bc = b_map[b_idx]
                V[br * p + bc][col] += coeff
            
            # Fill W: C index = k*n + i (col k, row i of 2x14 C)
            # Local C: C11=(0,2b), C12=(0,2b+1), C21=(1,2b), C22=(1,2b+1)
            c_map = {0: (0, 2*block), 1: (0, 2*block+1), 2: (1, 2*block), 3: (1, 2*block+1)}
            for c_idx, coeff in prod['w'].items():
                cr, cc = c_map[c_idx]
                W[cc * n + cr][col] += coeff
            
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
