import time
from fractions import Fraction

def verify(U, V, W, n=2, m=2, p=8):
    """Verify that U, V, W form an exact decomposition. Returns True/False."""
    nm, mp, pn = n*m, m*p, p*n
    rank = len(U[0]) if U else 0
    # Check shapes
    if len(U) != nm or len(V) != mp or len(W) != pn:
        return False
    for row in U:
        if len(row) != rank:
            return False
    for row in V:
        if len(row) != rank:
            return False
    for row in W:
        if len(row) != rank:
            return False
    
    # Check the 32 required entries equal 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                s = sum(U[a][r] * V[b][r] * W[c][r] for r in range(rank))
                if s != 1:
                    return False
    
    # Check that all other entries are 0
    valid = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                valid.add((i*m+j, j*p+k, k*n+i))
                
    for a in range(nm):
        for b in range(mp):
            for c in range(pn):
                if (a, b, c) not in valid:
                    s = sum(U[a][r] * V[b][r] * W[c][r] for r in range(rank))
                    if s != 0:
                        return False
    return True

def strassen_decomposition_228():
    """Construct a rank-28 decomposition using Strassen's algorithm on 2x2 blocks."""
    n, m, p = 2, 2, 8
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen vectors for 2x2x2
    # U vectors (4x1) for A indices [00, 01, 10, 11]
    U_strassen = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0], # M6
        [0, 1, 0, -1]  # M7
    ]
    
    # V vectors (4x1) for B indices [0k0, 0k1, 1k0, 1k1] (local)
    # Global mapping: local 0->b0, 1->b1, 2->b2, 3->b3
    V_strassen = [
        [1, 0, 0, 1], # M1: b0k0 + b1k1
        [1, 0, 0, 0], # M2: b0k0
        [0, 1, 0, -1], # M3: b0k1 - b1k1
        [-1, 0, 1, 0], # M4: b1k0 - b0k0
        [0, 0, 0, 1], # M5: b1k1
        [1, 1, 0, 0], # M6: b0k0 + b0k1
        [0, 0, 1, 1]  # M7: b1k0 + b1k1
    ]
    
    # W vectors (4x1) for C indices [0k0, 0k1, 1k0, 1k1] (local)
    W_strassen = [
        [1, 0, 0, 1], # M1: c0k0 + c1k1
        [0, 0, 1, -1], # M2: c1k0 - c1k1
        [0, 1, 0, 1], # M3: c0k1 + c1k1
        [1, 0, 1, 0], # M4: c0k0 + c1k0
        [-1, 1, 0, 0], # M5: -c0k0 + c0k1
        [0, 0, 0, 1], # M6: c1k1
        [1, 0, 0, 0]  # M7: c0k0
    ]
    
    # 4 blocks of 2 columns each
    for k_block in range(4):
        k0 = 2 * k_block
        k1 = 2 * k_block + 1
        
        # Global indices for B (rows 0,1; cols k0,k1)
        # b = j*p + k
        b_idx = [
            0 * p + k0, # j=0, k=k0
            0 * p + k1, # j=0, k=k1
            1 * p + k0, # j=1, k=k0
            1 * p + k1  # j=1, k=k1
        ]
        
        # Global indices for C (rows 0,1; cols k0,k1)
        # c = k*n + i
        c_idx = [
            k0 * n + 0, # k=k0, i=0
            k1 * n + 0, # k=k1, i=0
            k0 * n + 1, # k=k0, i=1
            k1 * n + 1  # k=k1, i=1
        ]
        
        for s in range(7):
            r = 7 * k_block + s
            
            # Fill U
            for i in range(4):
                U[i][r] = U_strassen[s][i]
                
            # Fill V
            for local_i, global_b in enumerate(b_idx):
                V[global_b][r] = V_strassen[s][local_i]
                
            # Fill W
            for local_i, global_c in enumerate(c_idx):
                W[global_c][r] = W_strassen[s][local_i]
                
    return U, V, W

def construct():
    """Entry point: return the best decomposition found."""
    U, V, W = strassen_decomposition_228()
    
    # Verify to be safe
    if verify(U, V, W):
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to schoolbook if verification fails (should not happen)
        n, m, p = 2, 2, 8
        rank = n * m * p
        U_fb = [[0] * rank for _ in range(n * m)]
        V_fb = [[0] * rank for _ in range(m * p)]
        W_fb = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j][col] = 1
                    V_fb[j * p + k][col] = 1
                    W_fb[k * n + i][col] = 1
                    col += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
