import time

DIMENSIONS = [2, 2, 8]

def construct():
    n, m, p = DIMENSIONS
    # Rank is 4 * 7 = 28
    rank = 28
    
    # Initialize U, V, W with zeros
    # U: (n*m, rank) = (4, 28)
    # V: (m*p, rank) = (16, 28)
    # W: (p*n, rank) = (16, 28)
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Strassen products for a single 2x2x2 block
    # A = [[a, b], [c, d]] -> U indices: 0:a, 1:b, 2:c, 3:d
    # B = [[e, g], [f, h]] -> V indices for block: 0:e, 1:f, 2:g, 3:h (local)
    # C = [[C00, C01], [C10, C11]] -> W indices for block: 0:C00, 1:C01, 2:C10, 3:C11 (local)
    
    # Local U vectors for a, b, c, d
    # Local V vectors for e, f, g, h
    # Local W vectors for C00, C01, C10, C11
    
    # M1 = (a+d)(e+h)
    u1 = [1, 0, 0, 1]
    v1 = [1, 1, 0, 0] # e+h
    w1 = [1, -1, 1, -1] # This is a guess, need correct reconstruction
    
    # Correct Reconstruction Coefficients:
    # C00 = M1 - M4 + M5 + M7
    # C01 = M4 - M5 + M6
    # C10 = M1 + M2 - M3 - M7
    # C11 = -M1 + M3 + M2 + M6
    
    # M1
    u_1 = [1, 0, 0, 1]
    v_1 = [1, 1, 0, 0]
    w_1 = [1, 0, 1, -1] # Coeffs for M1 in C00, C01, C10, C11
    
    # M2 = (c+d)(g+h)
    u_2 = [0, 0, 1, 1]
    v_2 = [0, 0, 1, 1]
    w_2 = [0, 0, 1, 1] # Coeffs for M2
    
    # M3 = a(f-h)
    u_3 = [1, 0, 0, 0]
    v_3 = [0, 1, 0, -1]
    w_3 = [0, 0, -1, 1] # Coeffs for M3
    
    # M4 = d(e-f)
    u_4 = [0, 0, 0, 1]
    v_4 = [1, -1, 0, 0]
    w_4 = [-1, 1, 0, 0] # Coeffs for M4
    
    # M5 = (a+b)h
    u_5 = [1, 1, 0, 0]
    v_5 = [0, 0, 0, 1]
    w_5 = [1, -1, 0, 0] # Coeffs for M5
    
    # M6 = (c-d)g
    u_6 = [0, 0, 1, -1]
    v_6 = [0, 0, 1, 0]
    w_6 = [0, 1, 0, 1] # Coeffs for M6
    
    # M7 = (b-d)(f+g)
    u_7 = [0, 1, 0, -1]
    v_7 = [0, 1, 1, 0]
    w_7 = [1, 0, -1, 0] # Coeffs for M7
    
    # List of (u, v, w) tuples
    strassen_tensors = [
        (u_1, v_1, w_1),
        (u_2, v_2, w_2),
        (u_3, v_3, w_3),
        (u_4, v_4, w_4),
        (u_5, v_5, w_5),
        (u_6, v_6, w_6),
        (u_7, v_7, w_7)
    ]
    
    # Map to global indices for 4 blocks
    for l in range(4):
        base_col = l * 7
        for t_idx in range(7):
            col = base_col + t_idx
            u, v, w = strassen_tensors[t_idx]
            
            # U indices: i*m + j
            # u[0] is a (i=0,j=0), u[1] is b (i=0,j=1), u[2] is c (i=1,j=0), u[3] is d (i=1,j=1)
            U[0][col] += u[0]
            U[1][col] += u[1]
            U[2][col] += u[2]
            U[3][col] += u[3]
            
            # V indices: j*p + k
            # v[0] is e (j=0, k=2l), v[1] is f (j=1, k=2l), v[2] is g (j=0, k=2l+1), v[3] is h (j=1, k=2l+1)
            V[0 * p + 2*l][col] += v[0]
            V[1 * p + 2*l][col] += v[1]
            V[0 * p + 2*l + 1][col] += v[2]
            V[1 * p + 2*l + 1][col] += v[3]
            
            # W indices: k*n + i
            # w[0] is C00 (k=2l, i=0), w[1] is C01 (k=2l+1, i=0), w[2] is C10 (k=2l, i=1), w[3] is C11 (k=2l+1, i=1)
            W[2*l * n + 0][col] += w[0]
            W[(2*l+1) * n + 0][col] += w[1]
            W[2*l * n + 1][col] += w[2]
            W[(2*l+1) * n + 1][col] += w[3]
            
    return {"U": U, "V": V, "W": W}
