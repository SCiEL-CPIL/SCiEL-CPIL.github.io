"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 4, 4
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a local decomposition
    def add_block(i_start, j_start, k_start, local_Us, local_Vs, local_Ws, local_n, local_m, local_p):
        nonlocal col
        for r_idx in range(len(local_Us)):
            u_vec = local_Us[r_idx]
            v_vec = local_Vs[r_idx]
            w_vec = local_Ws[r_idx]
            
            # Map local U to global U
            for li in range(local_n):
                for lj in range(local_m):
                    val = u_vec[li * local_m + lj]
                    if val != 0:
                        gi = i_start + li
                        gj = j_start + lj
                        U[gi * m + gj][col] = val
                        
            # Map local V to global V
            for lj in range(local_m):
                for lk in range(local_p):
                    val = v_vec[lj * local_p + lk]
                    if val != 0:
                        gj = j_start + lj
                        gk = k_start + lk
                        V[gj * p + gk][col] = val
                        
            # Map local W to global W
            for lk in range(local_p):
                for li in range(local_n):
                    val = w_vec[lk * local_n + li]
                    if val != 0:
                        gk = k_start + lk
                        gi = i_start + li
                        W[gk * n + gi][col] = val
            col += 1

    # Define Strassen [2, 2, 2] vectors
    # Local n=2, m=2, p=2
    # U size 4, V size 4, W size 4
    strassen_Us = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0], # M6
        [0, 1, 0, -1]  # M7
    ]
    strassen_Vs = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1], # M3
        [-1, 0, 1, 0], # M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    strassen_Ws = [
        [1, 0, 0, 1], # M1
        [0, 1, 0, -1], # M2
        [0, 0, 1, 1], # M3
        [1, 1, 0, 0], # M4
        [-1, 0, 1, 0], # M5
        [0, 0, 0, 1], # M6
        [1, 0, 0, 0]  # M7
    ]

    # Define [1, 2, 2] vectors
    # Local n=1, m=2, p=2
    # U size 2, V size 4, W size 2
    one22_Us = [
        [1, 0], # M1
        [0, 1], # M2
        [1, 0], # M3
        [0, 1]  # M4
    ]
    one22_Vs = [
        [1, 0, 0, 0], # M1
        [0, 0, 1, 0], # M2
        [0, 1, 0, 0], # M3
        [0, 0, 0, 1]  # M4
    ]
    one22_Ws = [
        [1, 0], # M1
        [1, 0], # M2
        [0, 1], # M3
        [0, 1]  # M4
    ]

    # Block 1.1.1: i=0, j=0:2, k=0:2
    add_block(0, 0, 0, one22_Us, one22_Vs, one22_Ws, 1, 2, 2)
    
    # Block 1.1.2: i=0, j=2:4, k=0:2
    add_block(0, 2, 0, one22_Us, one22_Vs, one22_Ws, 1, 2, 2)
    
    # Block 1.2.1: i=0, j=0:2, k=2:4
    add_block(0, 0, 2, one22_Us, one22_Vs, one22_Ws, 1, 2, 2)
    
    # Block 1.2.2: i=0, j=2:4, k=2:4
    add_block(0, 2, 2, one22_Us, one22_Vs, one22_Ws, 1, 2, 2)
    
    # Block 2.1.1: i=1:3, j=0:2, k=0:2
    add_block(1, 0, 0, strassen_Us, strassen_Vs, strassen_Ws, 2, 2, 2)
    
    # Block 2.1.2: i=1:3, j=2:4, k=0:2
    add_block(1, 2, 0, strassen_Us, strassen_Vs, strassen_Ws, 2, 2, 2)
    
    # Block 2.2.1: i=1:3, j=0:2, k=2:4
    add_block(1, 0, 2, strassen_Us, strassen_Vs, strassen_Ws, 2, 2, 2)
    
    # Block 2.2.2: i=1:3, j=2:4, k=2:4
    add_block(1, 2, 2, strassen_Us, strassen_Vs, strassen_Ws, 2, 2, 2)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
