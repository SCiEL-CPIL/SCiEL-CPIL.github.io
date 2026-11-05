import time

DIMENSIONS = [5, 5, 5]

def _schoolbook_factors(n, m, p):
    """Returns U, V, W for n x m times m x p using schoolbook algorithm."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def _concat_factors(factors_list):
    """Concatenates a list of (U, V, W) tuples."""
    if not factors_list:
        return [], [], []
    total_rank = sum(len(U) for U, V, W in factors_list) # Wait, rank is number of columns
    total_rank = sum(len(U[0]) if U else 0 for U, V, W in factors_list)
    
    # Determine shapes
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    
    # We need to know the dimensions of the first factor to initialize
    first_U, first_V, first_W = factors_list[0]
    n_m = len(first_U)
    m_p = len(first_V)
    p_n = len(first_W)
    
    U = [[0] * total_rank for _ in range(n_m)]
    V = [[0] * total_rank for _ in range(m_p)]
    W = [[0] * total_rank for _ in range(p_n)]
    
    offset = 0
    for U_f, V_f, W_f in factors_list:
        r = len(U_f[0]) if U_f else 0
        for i in range(n_m):
            for j in range(r):
                U[i][offset + j] = U_f[i][j]
        for i in range(m_p):
            for j in range(r):
                V[i][offset + j] = V_f[i][j]
        for i in range(p_n):
            for j in range(r):
                W[i][offset + j] = W_f[i][j]
        offset += r
        
    return U, V, W

def construct():
    start_time = time.time()
    deadline = 120.0
    
    # 2+3 block decomposition
    # A = [[A22, A23], [A32, A33]]
    # B = [[B22, B23], [B32, B33]]
    
    # Blocks:
    # A22: 2x2, A23: 2x3, A32: 3x2, A33: 3x3
    # B22: 2x2, B23: 2x3, B32: 3x2, B33: 3x3
    
    # Products:
    # C22 = A22*B22 + A23*B32
    # C23 = A22*B23 + A23*B33
    # C32 = A32*B22 + A33*B32
    # C33 = A32*B23 + A33*B33
    
    # We need to map the global indices to block indices.
    # Global A is 5x5.
    # Indices 0,1 correspond to first block row/col.
    # Indices 2,3,4 correspond to second block row/col.
    
    # We will generate factors for each sub-product and sum them.
    # Since the tensor decomposition is linear, we can just concatenate the factors
    # if we ensure the U, V, W matrices are zero-padded to the full 5x5 dimensions.
    
    # Helper to create a factor for a sub-block multiplication embedded in 5x5.
    def _embedded_factor(n_sub, m_sub, p_sub, i_offset, j_offset, k_offset):
        """
        Generates U, V, W for multiplying a n_sub x m_sub matrix by m_sub x p_sub matrix.
        The result is embedded in the 5x5 output at (i_offset, j_offset) relative to the block structure?
        No, the indices are global.
        U has shape (25, rank). Non-zero only at rows corresponding to A's block.
        V has shape (25, rank). Non-zero only at cols corresponding to B's block.
        W has shape (25, rank). Non-zero only at rows corresponding to C's block.
        """
        # Get schoolbook factors for the sub-problem
        U_sub, V_sub, W_sub = _schoolbook_factors(n_sub, m_sub, p_sub)
        rank = len(U_sub[0])
        
        U = [[0] * rank for _ in range(25)]
        V = [[0] * rank for _ in range(25)]
        W = [[0] * rank for _ in range(25)]
        
        # Map sub-indices to global indices
        # A block is at rows i_offset..i_offset+n_sub-1, cols j_offset..j_offset+m_sub-1?
        # Wait, A is n x m.
        # In our case:
        # A22: n=2, m=2. i_offset=0, j_offset=0 (col of A is row of B? No)
        # A is 5x5.
        # A22 is top-left 2x2.
        # A23 is top-right 2x3.
        # A32 is bottom-left 3x2.
        # A33 is bottom-right 3x3.
        
        # For U (corresponding to A):
        # Row index in U is i*5 + j.
        # For A22: i in 0..1, j in 0..1.
        
        # For V (corresponding to B):
        # Row index in V is j*5 + k.
        # For B22: j in 0..1, k in 0..1.
        
        # For W (corresponding to C):
        # Row index in W is k*5 + i.
        # For C22: k in 0..1, i in 0..1.
        
        # Let's define the mapping for each block.
        
        # We pass the specific offsets.
        # i_offset: start row of A block
        # j_offset: start col of A block (which is start row of B block)
        # k_offset: start col of B block (which is start row of C block? No, W is k*n+i)
        
        # Actually, W index is k*5 + i.
        # i is row of C, k is col of C.
        
        # Let's just use the sub-factors and place them.
        
        for r in range(rank):
            for u_i in range(n_sub * m_sub):
                if U_sub[u_i][r] != 0:
                    # u_i = i_sub * m_sub + j_sub
                    i_sub = u_i // m_sub
                    j_sub = u_i % m_sub
                    global_i = i_offset + i_sub
                    global_j = j_offset + j_sub
                    U[global_i * 5 + global_j][r] = U_sub[u_i][r]
            
            for v_i in range(m_sub * p_sub):
                if V_sub[v_i][r] != 0:
                    # v_i = j_sub * p_sub + k_sub
                    j_sub = v_i // p_sub
                    k_sub = v_i % p_sub
                    global_j = j_offset + j_sub
                    global_k = k_offset + k_sub
                    V[global_j * 5 + global_k][r] = V_sub[v_i][r]
            
            for w_i in range(p_sub * n_sub):
                if W_sub[w_i][r] != 0:
                    # w_i = k_sub * n_sub + i_sub
                    k_sub = w_i // n_sub
                    i_sub = w_i % n_sub
                    global_k = k_offset + k_sub
                    global_i = i_offset + i_sub
                    W[global_k * 5 + global_i][r] = W_sub[w_i][r]
                    
        return U, V, W

    factors = []
    
    # C22 = A22*B22 + A23*B32
    # A22: 2x2 at (0,0). B22: 2x2 at (0,0). C22: 2x2 at (0,0).
    factors.append(_embedded_factor(2, 2, 2, 0, 0, 0))
    
    # A23: 2x3 at (0,2). B32: 3x2 at (2,0). C22: 2x2 at (0,0).
    # A23 is rows 0-1, cols 2-4.
    # B32 is rows 2-4, cols 0-1.
    # Product is 2x2 at rows 0-1, cols 0-1.
    factors.append(_embedded_factor(2, 3, 2, 0, 2, 0))
    
    # C23 = A22*B23 + A23*B33
    # A22: 2x2 at (0,0). B23: 2x3 at (0,2). C23: 2x3 at (0,2).
    factors.append(_embedded_factor(2, 2, 3, 0, 0, 2))
    
    # A23: 2x3 at (0,2). B33: 3x3 at (2,2). C23: 2x3 at (0,2).
    factors.append(_embedded_factor(2, 3, 3, 0, 2, 2))
    
    # C32 = A32*B22 + A33*B32
    # A32: 3x2 at (2,0). B22: 2x2 at (0,0). C32: 3x2 at (2,0).
    factors.append(_embedded_factor(3, 2, 2, 2, 0, 0))
    
    # A33: 3x3 at (2,2). B32: 3x2 at (2,0). C32: 3x2 at (2,0).
    factors.append(_embedded_factor(3, 3, 2, 2, 2, 0))
    
    # C33 = A32*B23 + A33*B33
    # A32: 3x2 at (2,0). B23: 2x3 at (0,2). C33: 3x3 at (2,2).
    factors.append(_embedded_factor(3, 2, 3, 2, 0, 2))
    
    # A33: 3x3 at (2,2). B33: 3x3 at (2,2). C33: 3x3 at (2,2).
    factors.append(_embedded_factor(3, 3, 3, 2, 2, 2))
    
    U, V, W = _concat_factors(factors)
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    construct()
