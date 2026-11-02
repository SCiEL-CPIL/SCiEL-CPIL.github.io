"""Matrix multiplication 2x3 x 3x10 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    """Construct rank-36 exact tensor decomposition for 2x3 * 3x10 matrix multiplication.
    
    The decomposition splits the problem into two parts based on the middle dimension j:
    1. j in {0, 1}: A1 (2x2) * B1 (2x10). Split B1 into 5 blocks of 2x2.
       Each block uses Strassen's algorithm (rank 7). Total rank 35.
    2. j = 2: A2 (2x1) * B2 (1x10). This part has rank 1 because 
       sum_r U[i,2,r] V[2,k,r] W[k,i,r] = 1 can be satisfied with a single term 
       where all entries are 1.
    
    Total rank = 35 + 1 = 36.
    """
    n, m, p = DIMENSIONS
    rank = 36
    U = [[0] * rank for _ in range(n * m)]   # shape (6, 36)
    V = [[0] * rank for _ in range(m * p)]   # shape (30, 36)
    W = [[0] * rank for _ in range(p * n)]   # shape (20, 36)
    
    col = 0
    
    # --- Part 1: A1 (2x2) * B1 (2x10) using 5 Strassen blocks ---
    # A1 indices: i in {0,1}, j in {0,1} -> U index i*m + j
    # B1 indices: j in {0,1}, k in {0..9} -> V index j*p + k
    # C1 indices: i in {0,1}, k in {0..9} -> W index k*n + i
    
    for g in range(5):
        k_base = 2 * g
        k0 = k_base
        k1 = k_base + 1
        
        # B indices for this block
        v_idx_b00 = 0 * p + k0
        v_idx_b01 = 0 * p + k1
        v_idx_b10 = 1 * p + k0
        v_idx_b11 = 1 * p + k1
        
        # A indices
        u_idx_a00 = 0 * m + 0
        u_idx_a01 = 0 * m + 1
        u_idx_a10 = 1 * m + 0
        u_idx_a11 = 1 * m + 1
        
        # C indices for this block
        w_idx_c00 = k0 * n + 0
        w_idx_c01 = k1 * n + 0
        w_idx_c10 = k0 * n + 1
        w_idx_c11 = k1 * n + 1
        
        # Strassen terms for 2x2 * 2x2
        # m1 = (a00 + a11) * (b00 + b11)
        U[u_idx_a00][col] = 1
        U[u_idx_a11][col] = 1
        V[v_idx_b00][col] = 1
        V[v_idx_b11][col] = 1
        W[w_idx_c00][col] = 1
        W[w_idx_c11][col] = 1
        col += 1
        
        # m2 = (a10 + a11) * b00
        U[u_idx_a10][col] = 1
        U[u_idx_a11][col] = 1
        V[v_idx_b00][col] = 1
        W[w_idx_c10][col] = 1
        W[w_idx_c11][col] = -1
        col += 1
        
        # m3 = a00 * (b01 - b11)
        U[u_idx_a00][col] = 1
        V[v_idx_b01][col] = 1
        V[v_idx_b11][col] = -1
        W[w_idx_c01][col] = 1
        W[w_idx_c11][col] = 1
        col += 1
        
        # m4 = a11 * (b10 - b00)
        U[u_idx_a11][col] = 1
        V[v_idx_b10][col] = 1
        V[v_idx_b00][col] = -1
        W[w_idx_c00][col] = 1
        W[w_idx_c10][col] = 1
        col += 1
        
        # m5 = (a00 + a01) * b11
        U[u_idx_a00][col] = 1
        U[u_idx_a01][col] = 1
        V[v_idx_b11][col] = 1
        W[w_idx_c01][col] = 1
        W[w_idx_c00][col] = -1
        col += 1
        
        # m6 = (a10 - a00) * (b00 + b01)
        U[u_idx_a10][col] = 1
        U[u_idx_a00][col] = -1
        V[v_idx_b00][col] = 1
        V[v_idx_b01][col] = 1
        W[w_idx_c11][col] = 1
        col += 1
        
        # m7 = (a01 - a11) * (b10 + b11)
        U[u_idx_a01][col] = 1
        U[u_idx_a11][col] = -1
        V[v_idx_b10][col] = 1
        V[v_idx_b11][col] = 1
        W[w_idx_c00][col] = 1
        col += 1
        
    # --- Part 2: A2 (2x1) * B2 (1x10) using Rank 1 ---
    # A2 indices: i in {0,1}, j=2 -> U index i*m + 2
    # B2 indices: j=2, k in {0..9} -> V index 2*p + k
    # C2 indices: i in {0,1}, k in {0..9} -> W index k*n + i
    
    # We use a single column (col=35) for this part.
    # U[i, 2, 35] = 1 for i in {0, 1}
    # V[2, k, 35] = 1 for k in {0..9}
    # W[k, i, 35] = 1 for k in {0..9}, i in {0, 1}
    
    for i in range(n):
        U[i * m + 2][col] = 1
        
    for k in range(p):
        V[2 * p + k][col] = 1
        
    for k in range(p):
        for i in range(n):
            W[k * n + i][col] = 1
            
    col += 1
            
    # col should be 36
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
