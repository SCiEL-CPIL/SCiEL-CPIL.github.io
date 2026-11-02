"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=4, m=5, p=5
    
    # We decompose C = A * B
    # A is 4x5, B is 5x5, C is 4x5
    # Split m=5 into m1=4, m2=1
    # A = [A1 | A2] where A1 is 4x4, A2 is 4x1
    # B = [B1; B2] where B1 is 4x5, B2 is 1x5
    # C = A1*B1 + A2*B2
    
    # Split B1 (4x5) into B11 (4x4) and B12 (4x1)
    # B1 = [B11 | B12]
    # A1*B1 = A1*B11 + A1*B12
    
    # Term 1: A1*B11 (4x4 * 4x4)
    # Term 2: A1*B12 (4x4 * 4x1)
    # Term 3: A2*B2 (4x1 * 1x5)
    
    rank_total = 0
    U_list = []
    V_list = []
    W_list = []
    
    def add_decomposition(U_local, V_local, W_local, i_offset, j_offset, k_offset):
        # U_local: (n_local * m_local, r)
        # V_local: (m_local * p_local, r)
        # W_local: (p_local * n_local, r)
        # i_offset: offset for i in U and W
        # j_offset: offset for j in U and V
        # k_offset: offset for k in V and W
        pass

    # We will build the global U, V, W directly.
    # U size: 4*5 = 20
    # V size: 5*5 = 25
    # W size: 5*4 = 20
    
    # Initialize with zeros
    U = [[0] * 52 for _ in range(20)]
    V = [[0] * 52 for _ in range(25)]
    W = [[0] * 52 for _ in range(20)]
    
    col = 0
    
    # Term 1: A1*B11 (4x4 * 4x4)
    # i in [0,3], j in [0,3], k in [0,3]
    # Global U index: i*5 + j
    # Global V index: j*5 + k
    # Global W index: k*4 + i
    for i in range(4):
        for j in range(4):
            for k in range(4):
                U[i*5 + j][col] = 1
                V[j*5 + k][col] = 1
                W[k*4 + i][col] = 1
                col += 1
    # col is now 16
    
    # Term 2: A1*B12 (4x4 * 4x1)
    # i in [0,3], j in [0,3], k in [0,0] (local p=1)
    # Global U index: i*5 + j
    # Global V index: j*5 + 4 (since k=4 in global p=5)
    # Global W index: 4*4 + i (since k=4 in global)
    for i in range(4):
        for j in range(4):
            # k is always 4 in global
            U[i*5 + j][col] = 1
            V[j*5 + 4][col] = 1
            W[4*4 + i][col] = 1
            col += 1
    # col is now 16 + 16 = 32
    
    # Term 3: A2*B2 (4x1 * 1x5)
    # i in [0,3], j in [0,0] (local m=1), k in [0,4]
    # Global U index: i*5 + 4 (since j=4 in global m=5)
    # Global V index: 4*5 + k
    # Global W index: k*4 + i
    for i in range(4):
        # j is always 4 in global
        for k in range(5):
            U[i*5 + 4][col] = 1
            V[4*5 + k][col] = 1
            W[k*4 + i][col] = 1
            col += 1
    # col is now 32 + 20 = 52
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
