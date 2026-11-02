"""
Optimized Strassen decomposition for 2x2 matrix multiplication.
Rank: 7 (Theoretical minimum).
"""

def construct():
    """
    Returns the rank-minimal (rank 7) exact tensor decomposition for 2x2 matrix multiplication.
    
    The decomposition uses Strassen's algorithm with integer entries.
    
    Interface:
        U: shape (n*m, rank) = (4, 7). Index i*m + j corresponds to A[i, j].
        V: shape (m*p, rank) = (4, 7). Index j*p + k corresponds to B[j, k].
        W: shape (p*n, rank) = (4, 7). Index k*n + i corresponds to C[i, k].
        
    The tensor equation satisfied is:
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i, j, k.
    """
    n, m, p = 2, 2, 2
    rank = 7
    
    # Initialize factor matrices as lists of lists of zeros
    # Shapes: U (4, 7), V (4, 7), W (4, 7)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen's seven products:
    # Let A = [[a00, a01], [a10, a11]]
    #     B = [[b00, b01], [b10, b11]]
    #
    # m0 = (a00 + a11) * (b00 + b11)
    # m1 = (a10 + a11) * b00
    # m2 = a00 * (b01 - b11)
    # m3 = a11 * (b10 - b00)
    # m4 = (a00 + a01) * b11
    # m5 = (a10 - a00) * (b00 + b01)
    # m6 = (a01 - a11) * (b10 + b11)
    #
    # c00 = m0 + m3 - m4 + m6
    # c01 = m2 + m4
    # c10 = m1 + m3
    # c11 = m0 - m1 + m2 + m5
    
    # Index mapping:
    # U: row = i*2 + j (a_ij)
    #   a00: 0, a01: 1, a10: 2, a11: 3
    # V: row = j*2 + k (b_jk)
    #   b00: 0, b01: 1, b10: 2, b11: 3
    # W: row = k*2 + i (c_ik) -> Note: prompt says W[k*n+i, r], n=2
    #   c00 (i=0, k=0): 0*2+0 = 0
    #   c10 (i=1, k=0): 0*2+1 = 1
    #   c01 (i=0, k=1): 1*2+0 = 2
    #   c11 (i=1, k=1): 1*2+1 = 3

    # --- Product 0: m0 = (a00 + a11)(b00 + b11) ---
    # A parts: a00 (0), a11 (3)
    U[0][0] = 1
    U[3][0] = 1
    # B parts: b00 (0), b11 (3)
    V[0][0] = 1
    V[3][0] = 1
    # C parts: c00 (0), c11 (3)
    W[0][0] = 1
    W[3][0] = 1

    # --- Product 1: m1 = (a10 + a11)b00 ---
    # A parts: a10 (2), a11 (3)
    U[2][1] = 1
    U[3][1] = 1
    # B parts: b00 (0)
    V[0][1] = 1
    # C parts: c10 (1), c11 (3)
    W[1][1] = 1
    W[3][1] = -1

    # --- Product 2: m2 = a00(b01 - b11) ---
    # A parts: a00 (0)
    U[0][2] = 1
    # B parts: b01 (1), b11 (3)
    V[1][2] = 1
    V[3][2] = -1
    # C parts: c01 (2), c11 (3)
    W[2][2] = 1
    W[3][2] = 1

    # --- Product 3: m3 = a11(b10 - b00) ---
    # A parts: a11 (3)
    U[3][3] = 1
    # B parts: b10 (2), b00 (0)
    V[2][3] = 1
    V[0][3] = -1
    # C parts: c00 (0), c10 (1)
    W[0][3] = 1
    W[1][3] = 1

    # --- Product 4: m4 = (a00 + a01)b11 ---
    # A parts: a00 (0), a01 (1)
    U[0][4] = 1
    U[1][4] = 1
    # B parts: b11 (3)
    V[3][4] = 1
    # C parts: c00 (0), c01 (2)
    W[0][4] = -1
    W[2][4] = 1

    # --- Product 5: m5 = (a10 - a00)(b00 + b01) ---
    # A parts: a10 (2), a00 (0)
    U[2][5] = 1
    U[0][5] = -1
    # B parts: b00 (0), b01 (1)
    V[0][5] = 1
    V[1][5] = 1
    # C parts: c11 (3)
    W[3][5] = 1

    # --- Product 6: m6 = (a01 - a11)(b10 + b11) ---
    # A parts: a01 (1), a11 (3)
    U[1][6] = 1
    U[3][6] = -1
    # B parts: b10 (2), b11 (3)
    V[2][6] = 1
    V[3][6] = 1
    # C parts: c00 (0)
    W[0][6] = 1

    return {"U": U, "V": V, "W": W}
