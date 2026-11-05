"""Matrix multiplication 3x4x8 rank-minimal decomposition."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 8
    
    # Strassen's 7-term decomposition for 2x2x2 blocks.
    # Each term is (u, v, w) where u,v,w are 4-element vectors for the local 2x2 blocks.
    # Local A index: a*2+b (a=row, b=col), similarly for B and C.
    strassen_7 = [
        # M1 = (A11+A22)(B11+B22): C11+=M1, C22+=M1
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # M2 = (A21+A22)B11: C12+=M2
        ([0, 1, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0]),
        # M3 = A11(B21-B22): C21+=M3, C22-=M3
        ([1, 0, 0, 0], [0, 0, 1, -1], [0, 0, 1, -1]),
        # M4 = A22(B12-B11): C11-=M4, C12+=M4
        ([0, 0, 0, 1], [-1, 1, 0, 0], [-1, 1, 0, 0]),
        # M5 = (A11+A12)B22: C21+=M5, C22+=M5
        ([1, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 1]),
        # M6 = (A21-A11)(B11+B12): C11+=M6, C22+=M6
        ([-1, 1, 0, 0], [1, 1, 0, 0], [1, 0, 0, 1]),
        # M7 = (A12-A22)(B21+B22): C11+=M7, C22+=M7
        ([1, 0, -1, 0], [0, 0, 1, 1], [1, 0, 0, 1]),
    ]
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    # 2x2x2 Strassen blocks: i in {0,1}, j pairs (0,1) and (2,3), k pairs (0,1),(2,3),(4,5),(6,7)
    for i0 in [0]:
        for j0 in [0, 2]:
            for k0 in [0, 2, 4, 6]:
                for (u, v, w) in strassen_7:
                    U_col = [0] * (n * m)
                    V_col = [0] * (m * p)
                    W_col = [0] * (p * n)
                    
                    for a in range(2):
                        for b in range(2):
                            U_col[(i0 + a) * m + (j0 + b)] = u[a * 2 + b]
                    
                    for a in range(2):
                        for b in range(2):
                            V_col[(j0 + a) * p + (k0 + b)] = v[a * 2 + b]
                    
                    for a in range(2):
                        for b in range(2):
                            W_col[(k0 + a) * n + (i0 + b)] = w[a * 2 + b]
                    
                    U_cols.append(U_col)
                    V_cols.append(V_col)
                    W_cols.append(W_col)
    
    # Schoolbook for i=2 (1x4x8 block): rank 4*8 = 32
    for j in range(m):
        for k in range(p):
            U_col = [0] * (n * m)
            V_col = [0] * (m * p)
            W_col = [0] * (p * n)
            
            U_col[2 * m + j] = 1
            V_col[j * p + k] = 1
            W_col[k * n + 2] = 1
            
            U_cols.append(U_col)
            V_cols.append(V_col)
            W_cols.append(W_col)
    
    rank = len(U_cols)
    
    # Transpose to (n*m, rank) format
    U = [[U_cols[c][r] for c in range(rank)] for r in range(n * m)]
    V = [[V_cols[c][r] for c in range(rank)] for r in range(m * p)]
    W = [[W_cols[c][r] for c in range(rank)] for r in range(p * n)]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
