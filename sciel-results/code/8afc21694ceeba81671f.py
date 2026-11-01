"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-7 tensor decomposition for 2x2 matrix multiplication.
    This implements Strassen's algorithm, which is known to be optimal 
    for 2x2 matrix multiplication over most fields.
    
    The decomposition is:
    T = sum_{r=0}^{6} u_r \otimes v_r \otimes w_r
    
    Where:
    U has shape (n*m, rank) -> (4, 7)
    V has shape (m*p, rank) -> (4, 7)
    W has shape (p*n, rank) -> (4, 7)
    
    Indexing conventions:
    U[i*m + j, r]
    V[j*p + k, r]
    W[k*n + i, r]  (Note: W follows the trace convention W[k*n+i])
    """
    n, m, p = 2, 2, 2
    rank = 7
    
    # Initialize matrices with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to get flat indices
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i
    
    # Strassen's 7 scalar products:
    # m0 = (a00 + a11) * (b00 + b11)
    # m1 = (a10 + a11) * b00
    # m2 = a00 * (b01 - b11)
    # m3 = a11 * (b10 - b00)
    # m4 = (a00 + a01) * b11
    # m5 = (a10 - a00) * (b00 + b01)
    # m6 = (a01 - a11) * (b10 + b11)
    
    # Component 0: m0
    U[u_idx(0,0)][0] = 1
    U[u_idx(1,1)][0] = 1
    V[v_idx(0,0)][0] = 1
    V[v_idx(1,1)][0] = 1
    W[w_idx(0,0)][0] = 1  # C00
    W[w_idx(1,1)][0] = 1  # C11
    
    # Component 1: m1
    U[u_idx(1,0)][1] = 1
    U[u_idx(1,1)][1] = 1
    V[v_idx(0,0)][1] = 1
    W[w_idx(0,1)][1] = 1  # C10
    W[w_idx(1,1)][1] = -1 # C11
    
    # Component 2: m2
    U[u_idx(0,0)][2] = 1
    V[v_idx(0,1)][2] = 1
    V[v_idx(1,1)][2] = -1
    W[w_idx(1,0)][2] = 1  # C01
    W[w_idx(1,1)][2] = 1  # C11
    
    # Component 3: m3
    U[u_idx(1,1)][3] = 1
    V[v_idx(1,0)][3] = 1
    V[v_idx(0,0)][3] = -1
    W[w_idx(0,0)][3] = 1  # C00
    W[w_idx(0,1)][3] = 1  # C10
    
    # Component 4: m4
    U[u_idx(0,0)][4] = 1
    U[u_idx(0,1)][4] = 1
    V[v_idx(1,1)][4] = 1
    W[w_idx(0,0)][4] = -1 # C00
    W[w_idx(1,0)][4] = 1  # C01
    
    # Component 5: m5
    U[u_idx(1,0)][5] = 1
    U[u_idx(0,0)][5] = -1
    V[v_idx(0,0)][5] = 1
    V[v_idx(0,1)][5] = 1
    W[w_idx(1,1)][5] = 1  # C11
    
    # Component 6: m6
    U[u_idx(0,1)][6] = 1
    U[u_idx(1,1)][6] = -1
    V[v_idx(1,0)][6] = 1
    V[v_idx(1,1)][6] = 1
    W[w_idx(0,0)][6] = 1  # C00

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
