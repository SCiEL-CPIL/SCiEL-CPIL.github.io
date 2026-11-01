"""Strassen-based rank-21 decomposition for 2x2 * 2x6 matrix multiplication."""
DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=6
    rank = 21
    # U: (n*m, rank) = (4, 21), index [i*m+j, c]
    # V: (m*p, rank) = (12, 21), index [j*p+k, c]
    # W: (p*n, rank) = (12, 21), index [k*n+i, c]
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen products (corrected m4 = (d-c) * g to be a valid bilinear form):
    # m1 = (a+d) * (e+f)
    # m2 = (c+d) * e
    # m3 = a * (f-h)
    # m4 = (d-c) * g
    # m5 = (a+b) * h
    # m6 = (c-a) * (e+f)
    # m7 = (b-d) * (g+h)
    # Reconstruction:
    # C[0,0] = m1 + m4 - m5 - m7
    # C[0,1] = m3 + m5
    # C[1,0] = m2 + m4
    # C[1,1] = m1 - m2 + m3 + m7

    # Each Strassen product c corresponds to 3 columns of the rank-21 tensor,
    # one for each of the 6 columns k of B (hence of C).
    # For a given c and a given k, the A-side and B-side coefficient of the
    # product is the same for every k; only the W-side C-index depends on k.

    # For each Strassen product c in 0..6, define:
    #   U entries: A-row/col coefficients
    #   V entries: B-row/col coefficients
    #   For each k in 0..5, W entries: C-row/col coefficient

    # U coefficients per (i,j) for each Strassen product c:
    # c=0 (m1): a+d -> U[0,0]=1, U[1,1]=1
    # c=1 (m2): c+d -> U[1,0]=1, U[1,1]=1
    # c=2 (m3): a   -> U[0,0]=1
    # c=3 (m4): d-c -> U[1,1]=1, U[1,0]=-1
    # c=4 (m5): a+b -> U[0,0]=1, U[0,1]=1
    # c=5 (m6): c-a -> U[1,0]=1, U[0,0]=-1
    # c=6 (m7): b-d -> U[0,1]=1, U[1,1]=-1
    U_coef = [
        {(0,0): 1, (1,1): 1},       # c=0
        {(1,0): 1, (1,1): 1},       # c=1
        {(0,0): 1},                  # c=2
        {(1,1): 1, (1,0): -1},      # c=3
        {(0,0): 1, (0,1): 1},       # c=4
        {(1,0): 1, (0,0): -1},      # c=5
        {(0,1): 1, (1,1): -1},      # c=6
    ]
    # V coefficients per (j,k) for each Strassen product c:
    # c=0 (m1): (e+f) -> V[0,0]=1, V[0,1]=1
    # c=1 (m2): e     -> V[0,0]=1
    # c=2 (m3): (f-h) -> V[0,1]=1, V[1,1]=-1
    # c=3 (m4): g     -> V[1,0]=1
    # c=4 (m5): h     -> V[1,1]=1
    # c=5 (m6): (e+f) -> V[0,0]=1, V[0,1]=1
    # c=6 (m7): (g+h) -> V[1,0]=1, V[1,1]=1
    V_coef = [
        {(0,0): 1, (0,1): 1},       # c=0
        {(0,0): 1},                  # c=1
        {(0,1): 1, (1,1): -1},      # c=2
        {(1,0): 1},                  # c=3
        {(1,1): 1},                  # c=4
        {(0,0): 1, (0,1): 1},       # c=5
        {(1,0): 1, (1,1): 1},       # c=6
    ]
    # W coefficients per (k,i) for each Strassen product c:
    # C[0,0] = m1 + m4 - m5 - m7
    # C[0,1] = m3 + m5
    # C[1,0] = m2 + m4
    # C[1,1] = m1 - m2 + m3 + m7
    # For a given C-row i and C-col k:
    #   C[0,0]: m1=+1, m4=+1, m5=-1, m7=-1
    #   C[0,1]: m3=+1, m5=+1
    #   C[1,0]: m2=+1, m4=+1
    #   C[1,1]: m1=+1, m2=-1, m3=+1, m7=+1
    # W[k*n+i] with n=2, so W[0]=C[0,0], W[1]=C[0,1], W[2]=C[1,0], W[3]=C[1,1], etc.
    W_coef = [
        {(0,0): 1, (0,1): 1, (1,0): -1, (1,1): -1},   # c=0
        {(1,0): 1, (1,1): -1},                          # c=1
        {(0,1): 1, (1,1): 1},                           # c=2
        {(0,0): 1, (1,0): 1},                           # c=3
        {(0,0): -1, (0,1): 1},                          # c=4
        {},                                              # c=5
        {(0,0): -1, (1,1): 1},                          # c=6
    ]

    for c in range(7):
        for k in range(p):
            col = c * 3 + k
            # Fill U
            for (i, j), val in U_coef[c].items():
                U[i * m + j][col] += val
            # Fill V
            for (jj, kk), val in V_coef[c].items():
                V[jj * p + kk][col] += val
            # Fill W (only columns where C-col = k)
            for (kk, ii), val in W_coef[c].items():
                if kk == k:
                    W[kk * n + ii][col] += val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
