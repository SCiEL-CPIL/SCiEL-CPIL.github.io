"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
import time

def construct():
    n, m, p = DIMENSIONS
    # Total dimensions for U, V, W rows
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    
    # Strassen blocks defined by index sets
    # We use 4 indices for I, J, K in sets {0,1} and {2,3}
    idx_sets = [ (0, 1), (2, 3) ]
    
    # List of columns for U, V, W
    cols_U = []
    cols_V = []
    cols_W = []
    
    # Helper to add a column
    def add_col(u_vals, v_vals, w_vals):
        cols_U.append(u_vals)
        cols_V.append(v_vals)
        cols_W.append(w_vals)
        
    # Initialize vectors with zeros
    def zero_vec(dim):
        return [0] * dim
        
    # Strassen 2x2 logic
    # Local indices 0,1 map to global sets
    # A_local[0,0] = A[I0, J0]
    # A_local[0,1] = A[I0, J1]
    # A_local[1,0] = A[I1, J0]
    # A_local[1,1] = A[I1, J1]
    
    # B_local[0,0] = B[J0, K0]
    # B_local[0,1] = B[J0, K1]
    # B_local[1,0] = B[J1, K0]
    # B_local[1,1] = B[J1, K1]
    
    # C_prime[0,0] = C_global[K0, I0] -> W idx K0*n + I0
    # C_prime[0,1] = C_global[K1, I0] -> W idx K1*n + I0
    # C_prime[1,0] = C_global[K0, I1] -> W idx K0*n + I1
    # C_prime[1,1] = C_global[K1, I1] -> W idx K1*n + I1
    
    def add_strassen_block(I_set, J_set, K_set):
        I0, I1 = I_set
        J0, J1 = J_set
        K0, K1 = K_set
        
        # Indices in U (row I, col J)
        u_idx_a = I0 * m + J0
        u_idx_b = I0 * m + J1
        u_idx_c = I1 * m + J0
        u_idx_d = I1 * m + J1
        
        # Indices in V (row J, col K)
        v_idx_e = J0 * p + K0
        v_idx_f = J0 * p + K1
        v_idx_g = J1 * p + K0
        v_idx_h = J1 * p + K1
        
        # Indices in W (row K, col I) -> idx K * n + I
        w_idx_11 = K0 * n + I0
        w_idx_12 = K1 * n + I0
        w_idx_21 = K0 * n + I1
        w_idx_22 = K1 * n + I1
        
        # z1 = (a+d)(e+h)
        # U: a(1), d(1). V: e(1), h(1). W: 11(1), 12(-1), 21(-1), 22(1) ?
        # C11 = z1 + z4 - z5 + z7
        # C12 = z3 + z5
        # C21 = z2 + z4
        # C22 = z1 - z2 + z3 + z6
        
        # Let's define each z as a column contribution
        # We need to construct U, V, W columns for each z
        
        # z1
        u = zero_vec(dim_U); v = zero_vec(dim_V); w = zero_vec(dim_W)
        u[u_idx_a] = 1; u[u_idx_d] = 1
        v[v_idx_e] = 1; v[v_idx_h] = 1
        w[w_idx_11] = 1; w[w_idx_12] = -1; w[w_idx_21] = -1; w[w_idx_22] = 1
        add_col(u, v, w)
        
        # z2 = (c+d)e
        u = zero_vec(dim_U); v = zero_vec(dim_V); w = zero_vec(dim_W)
        u[u_idx_c] = 1; u[u_idx_d] = 1
        v[v_idx_e] = 1
        w[w_idx_21] = 1; w[w_idx_22] = -1
        add_col(u, v, w)
        
        # z3 = a(f-h)
        u = zero_vec(dim_U); v = zero_vec(dim_V); w = zero_vec(dim_W)
        u[u_idx_a] = 1
        v[v_idx_f] = 1; v[v_idx_h] = -1
        w[w_idx_12] = 1; w[w_idx_22] = 1
        add_col(u, v, w)
        
        # z4 = d(g-e)
        u = zero_vec(dim_U); v = zero_vec(dim_V); w = zero_vec(dim_W)
        u[u_idx_d] = 1
        v[v_idx_g] = 1; v[v_idx_e] = -1
        w[w_idx_11] = 1; w[w_idx_21] = 1
        add_col(u, v, w)
        
        # z5 = (a+b)h
        u = zero_vec(dim_U); v = zero_vec(dim_V); w = zero_vec(dim_W)
        u[u_idx_a] = 1; u[u_idx_b] = 1
        v[v_idx_h] = 1
        w[w_idx_11] = -1; w[w_idx_12] = 1
        add_col(u, v, w)
        
        # z6 = (c-a)(e+f)
        u = zero_vec(dim_U); v = zero_vec(dim_V); w = zero_vec(dim_W)
        u[u_idx_c] = 1; u[u_idx_a] = -1
        v[v_idx_e] = 1; v[v_idx_f] = 1
        w[w_idx_22] = 1
        add_col(u, v, w)
        
        # z7 = (b-d)(g+h)
        u = zero_vec(dim_U); v = zero_vec(dim_V); w = zero_vec(dim_W)
        u[u_idx_b] = 1; u[u_idx_d] = -1
        v[v_idx_g] = 1; v[v_idx_h] = 1
        w[w_idx_11] = 1; w[w_idx_22] = -1
        add_col(u, v, w)
        
    # Add all 8 blocks
    for I_set in idx_sets:
        for J_set in idx_sets:
            for K_set in idx_sets:
                add_strassen_block(I_set, J_set, K_set)
                
    # Add schoolbook for remaining
    # Covered I: {0,1,2,3}, J: {0,1,2,3}, K: {0,1,2,3}
    # Uncovered if i==4 or j==4 or k==4
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i == 4 or j == 4 or k == 4:
                    u = zero_vec(dim_U); v = zero_vec(dim_V); w = zero_vec(dim_W)
                    u[i * m + j] = 1
                    v[j * p + k] = 1
                    w[k * n + i] = 1
                    add_col(u, v, w)
                    
    rank = len(cols_U)
    U = [cols_U[c] for c in range(rank)] # Transpose to list of rows
    # Wait, construct returns dict with lists. The prompt example has U as list of rows.
    # "U": U, U shape (n*m, rank). So U[row][col].
    # My cols_U is list of columns. 
    # I need to transpose it back to row-major list of lists.
    
    U_final = [[0]*rank for _ in range(dim_U)]
    V_final = [[0]*rank for _ in range(dim_V)]
    W_final = [[0]*rank for _ in range(dim_W)]
    
    for c in range(rank):
        for r in range(dim_U):
            U_final[r][c] = cols_U[c][r]
        for r in range(dim_V):
            V_final[r][c] = cols_V[c][r]
        for r in range(dim_W):
            W_final[r][c] = cols_W[c][r]
            
    return {"U": U_final, "V": V_final, "W": W_final}
# EVOLVE-BLOCK-END
