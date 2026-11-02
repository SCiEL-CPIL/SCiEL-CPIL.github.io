"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns the rank-minimal tensor decomposition for 2x2 matrix multiplication.
    Uses Strassen's algorithm to achieve rank 7.
    
    The decomposition is defined by:
    C[i,k] = sum_c W[k*n+i, c] * (sum_a U[i*m+a, c] * A[i,a]) * (sum_b V[a*p+b, c] * B[a,b])
    
    For 2x2 matrices:
    A = [[a, b], [c, d]]
    B = [[e, f], [g, h]]
    C = [[ae+bg, af+bh], [ce+dg, cf+dh]]
    
    We define 7 intermediate products M_c such that C entries are linear combinations of M_c.
    U, V, W encode the linear forms for A, B, and the summation for C respectively.
    """
    n, m, p = DIMENSIONS
    rank = 7
    
    # Initialize factor matrices with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define the linear forms for each of the 7 Strassen products.
    # Each entry is a list of (index, coefficient) pairs.
    # Index corresponds to the flattened index of the input matrix (row * dim + col).
    # For U (left factor, size 2x2), indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # For V (right factor, size 2x2), indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # For W (output factor, size 2x2), indices: 0:(0,0), 1:(1,0), 2:(0,1), 3:(1,1) 
    # Note: The task specifies W[k*n+i, r]. So row index is k*n + i.
    # For 2x2:
    # (i=0, k=0) -> row 0 (C[0,0])
    # (i=1, k=0) -> row 1 (C[1,0])
    # (i=0, k=1) -> row 2 (C[0,1])
    # (i=1, k=1) -> row 3 (C[1,1])
    
    # Strassen's 7 products:
    # M0 = (a + d) * (e + h)
    # M1 = (c + d) * e
    # M2 = a * (f - h)
    # M3 = d * (g - f)
    # M4 = (a + b) * h
    # M5 = (c - a) * (e + f)
    # M6 = (b - d) * (g + h)
    
    # Output formulas:
    # C[0,0] = M0 - M1 + M3 + M4
    # C[1,0] = M1 + M3
    # C[0,1] = M2 + M4
    # C[1,1] = M0 - M2 + M5 - M6
    
    strassen_forms = [
        # M0: (a+d)(e+h)
        {
            'u': [(0, 1), (3, 1)],   # a + d
            'v': [(0, 1), (3, 1)],   # e + h
            'w': [(0, 1), (3, -1)]   # C00 += 1, C11 -= 1
        },
        # M1: (c+d)e
        {
            'u': [(2, 1), (3, 1)],   # c + d
            'v': [(0, 1)],           # e
            'w': [(0, -1), (1, 1)]   # C00 -= 1, C10 += 1
        },
        # M2: a(f-h)
        {
            'u': [(0, 1)],           # a
            'v': [(1, 1), (3, -1)],  # f - h
            'w': [(2, 1), (3, -1)]   # C01 += 1, C11 -= 1
        },
        # M3: d(g-f)
        {
            'u': [(3, 1)],           # d
            'v': [(2, 1), (1, -1)],  # g - f
            'w': [(0, 1), (1, 1)]    # C00 += 1, C10 += 1
        },
        # M4: (a+b)h
        {
            'u': [(0, 1), (1, 1)],   # a + b
            'v': [(3, 1)],           # h
            'w': [(0, 1), (2, 1)]    # C00 += 1, C01 += 1
        },
        # M5: (c-a)(e+f)
        {
            'u': [(2, 1), (0, -1)],  # c - a
            'v': [(0, 1), (1, 1)],   # e + f
            'w': [(3, 1)]            # C11 += 1
        },
        # M6: (b-d)(g+h)
        {
            'u': [(1, 1), (3, -1)],  # b - d
            'v': [(2, 1), (3, 1)],   # g + h
            'w': [(3, -1)]           # C11 -= 1
        }
    ]
    
    for c in range(rank):
        form = strassen_forms[c]
        
        # Fill U
        for idx, coeff in form['u']:
            U[idx][c] = coeff
            
        # Fill V
        for idx, coeff in form['v']:
            V[idx][c] = coeff
            
        # Fill W
        for idx, coeff in form['w']:
            W[idx][c] = coeff

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
