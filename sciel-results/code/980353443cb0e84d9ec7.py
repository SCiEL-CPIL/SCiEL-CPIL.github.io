import numpy as np

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    # Block decomposition strategy:
    # Treat A as column of n x 1 blocks (actually 2x1)
    # Treat B as row of 1 x p blocks (actually 1x7)
    # Iterate over the m blocks
    for j in range(m):
        for i in range(n):
            for k in range(p):
                # Indices for U (A part)
                # A is n x m. Index a = i * m + j
                a = i * m + j
                # Indices for V (B part)
                # B is m x p. Index b = j * p + k
                b = j * p + k
                # Indices for W (C part)
                # C is n x p. Task says W[k*n+i, r]. 
                # Note: Standard C index is i*p + k or similar.
                # Task specifies W[k*n+i, r].
                # Let's verify the size. p*n = 7*2 = 14.
                # k is 0..6, i is 0..1.
                # c = k * n + i = 2k + i.
                # This maps (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3 ... (6,0)->12, (6,1)->13.
                # This is a valid bijection to 0..13.
                c = k * n + i
                
                U[a, col] = 1
                V[b, col] = 1
                W[c, col] = 1
                col += 1
                
    U_list = [list(map(int, row)) for row in U]
    V_list = [list(map(int, row)) for row in V]
    W_list = [list(map(int, row)) for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
