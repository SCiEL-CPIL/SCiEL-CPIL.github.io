import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We will tile the tensor with 2x2x2 Strassen blocks and 1x1x1 schoolbook blocks.
    # This ensures a valid decomposition.
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    # Define Strassen 2x2x2 decomposition
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # U vectors (for A)
    # A00=1, A11=1 -> [1, 0, 0, 1] (flattened 2x2)
    # A10=1, A11=1 -> [0, 1, 0, 1]
    # A00=1 -> [1, 0, 0, 0]
    # A11=1 -> [0, 0, 0, 1]
    # A00=1, A01=1 -> [1, 1, 0, 0]
    # A10=1, A00=-1 -> [-1, 1, 0, 0]
    # A01=1, A11=-1 -> [0, 1, 0, -1]
    
    u_strassen = [
        [1, 0, 0, 1],
        [0, 1, 0, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 1, 0, 0],
        [0, 1, 0, -1]
    ]
    
    # V vectors (for B)
    # B00=1, B11=1 -> [1, 0, 0, 1]
    # B00=1 -> [1, 0, 0, 0]
    # B01=1, B11=-1 -> [0, 1, 0, -1]
    # B10=1, B00=-1 -> [-1, 1, 0, 0]
    # B11=1 -> [0, 0, 0, 1]
    # B00=1, B01=1 -> [1, 1, 0, 0]
    # B10=1, B11=1 -> [0, 1, 1, 0]
    
    v_strassen = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 1, 1, 0]
    ]
    
    # W vectors (for C)
    # C00 = M1 - M4 + M6
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M7
    
    # M1: [1, 0, 0, 1]
    # M2: [0, 0, 1, 0]
    # M3: [0, 1, 0, 0]
    # M4: [0, 0, 0, 1]
    # M5: [0, 0, 0, 1] -> Wait, C11 uses M5?
    # C00 = M1 - M4 + M6
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M7
    
    # Let's map to indices 0,1,2,3 for C00,C01,C10,C11
    w_strassen = [
        [1, 0, 0, 0], # M1 for C00
        [0, 0, 1, 0], # M2 for C10
        [0, 1, 0, 0], # M3 for C01
        [0, 0, 0, 1], # M4 for C00 (-1) and C10 (+1)
        [0, 1, 0, 0], # M5 for C01
        [1, 0, 0, 0], # M6 for C00
        [0, 0, 0, 1]  # M7 for C11
    ]
    # Correcting W signs based on formulas:
    # C00 = M1 - M4 + M6
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M7
    
    # We need to combine these into 7 vectors for the 7 products.
    # Prod 1 (M1): contributes to C00 (+1), C11 (+1) -> [1, 0, 0, 1]
    # Prod 2 (M2): contributes to C10 (+1) -> [0, 0, 1, 0]
    # Prod 3 (M3): contributes to C01 (+1), C11 (-1) -> [0, 1, 0, -1]
    # Prod 4 (M4): contributes to C00 (-1), C10 (+1) -> [-1, 0, 1, 0]
    # Prod 5 (M5): contributes to C01 (+1) -> [0, 1, 0, 0]
    # Prod 6 (M6): contributes to C00 (+1) -> [1, 0, 0, 0]
    # Prod 7 (M7): contributes to C11 (+1) -> [0, 0, 0, 1]
    
    w_strassen = [
        [1, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1]
    ]

    def add_block(i_start, j_start, k_start, u_base, v_base, w_base):
        # u_base is list of vectors of length n_block * m_block
        # We need to map these to the global U, V, W matrices.
        pass

    # This approach is too complex to implement dynamically in the short time.
    # I will use the simple tiling with pre-defined 2x2x2 blocks and 1x1x1 blocks.
    
    U = [[0] * 105 for _ in range(15)] # n*m = 15
    V = [[0] * 105 for _ in range(35)] # m*p = 35
    W = [[0] * 105 for _ in range(21)] # p*n = 21
    
    rank = 0
    
    # Define Strassen 2x2x2
    # Indices in local 2x2x2 block:
    # u: 0,1,2,3 (i,j)
    # v: 0,1,2,3 (j,k)
    # w: 0,1,2,3 (k,i)
    
    strassen_u = [
        [1, 0, 0, 1],
        [0, 1, 0, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 1, 0, 0],
        [0, 1, 0, -1]
    ]
    strassen_v = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 1, 1, 0]
    ]
    strassen_w = [
        [1, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1]
    ]
    
    def add_strassen_block(i0, j0, k0):
        nonlocal rank
        for t in range(7):
            u_vec = strassen_u[t]
            v_vec = strassen_v[t]
            w_vec = strassen_w[t]
            
            for idx, val in enumerate(u_vec):
                if val != 0:
                    i_local = idx // 2
                    j_local = idx % 2
                    i_global = i0 + i_local
                    j_global = j0 + j_local
                    U[i_global * 5 + j_global][rank] = val
            
            for idx, val in enumerate(v_vec):
                if val != 0:
                    j_local = idx // 2
                    k_local = idx % 2
                    j_global = j0 + j_local
                    k_global = k0 + k_local
                    V[j_global * 7 + k_global][rank] = val
            
            for idx, val in enumerate(w_vec):
                if val != 0:
                    k_local = idx // 2
                    i_local = idx % 2
                    k_global = k0 + k_local
                    i_global = i0 + i_local
                    W[k_global * 3 + i_global][rank] = val
            
            rank += 1

    def add_schoolbook_block(i0, j0, k0):
        nonlocal rank
        i_global = i0
        j_global = j0
        k_global = k0
        U[i_global * 5 + j_global][rank] = 1
        V[j_global * 7 + k_global][rank] = 1
        W[k_global * 3 + i_global][rank] = 1
        rank += 1

    # Tiling
    # I: 0,1 (2), 2 (1)
    # J: 0,1 (2), 2,3 (2), 4 (1)
    # K: 0,1 (2), 2,3 (2), 4,5 (2), 6 (1)
    
    for i0 in [0, 2]:
        i_size = 2 if i0 == 0 else 1
        for j0 in [0, 2, 4]:
            j_size = 2 if j0 < 4 else 1
            for k0 in [0, 2, 4, 6]:
                k_size = 2 if k0 < 6 else 1
                
                if i_size == 2 and j_size == 2 and k_size == 2:
                    add_strassen_block(i0, j0, k0)
                else:
                    for i in range(i_size):
                        for j in range(j_size):
                            for k in range(k_size):
                                add_schoolbook_block(i0+i, j0+j, k0+k)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
