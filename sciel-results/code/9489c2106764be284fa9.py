"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Strassen coefficients for 2x2x2 multiplication
    # U: A indices (i*m+j), V: B indices (j*p+k), W: C indices (k*n+i)
    # For a single 2x2 block, local indices are 0..3 for U, V, W
    
    # M1 = (A00+A11)(B00+B11)
    u1 = [1, 0, 0, 1]
    v1 = [1, 0, 0, 1]
    w1 = [1, 0, 0, 1] # C00 += 1, C11 += 1
    
    # M2 = (A10+A11)B00
    u2 = [0, 0, 1, 1]
    v2 = [1, 0, 0, 0]
    w2 = [0, 0, 1, 0] # C10 += 1
    
    # M3 = A00(B10-B11)
    u3 = [1, 0, 0, 0]
    v3 = [0, 0, 1, -1]
    w3 = [0, 1, 0, -1] # C01 += 1, C11 -= 1
    
    # M4 = A11(B01-B00)
    u4 = [0, 0, 0, 1]
    v4 = [0, 1, -1, 0]
    w4 = [-1, 0, 1, 0] # C00 -= 1, C10 += 1
    
    # M5 = (A00+A10)B11
    u5 = [1, 0, 1, 0]
    v5 = [0, 0, 0, 1]
    w5 = [0, 1, 0, 0] # C01 += 1
    
    # M6 = (A11-A00)(B00+B10)
    u6 = [-1, 0, 0, 1]
    v6 = [1, 0, 1, 0]
    w6 = [-1, 0, 0, 1] # C00 -= 1, C11 += 1
    
    # M7 = (A01-A11)(B11+B01)
    u7 = [0, 1, 0, -1]
    v7 = [0, 1, 0, 1]
    w7 = [1, 0, 0, -1] # C00 += 1, C11 -= 1
    
    U_vecs = [u1, u2, u3, u4, u5, u6, u7]
    V_vecs = [v1, v2, v3, v4, v5, v6, v7]
    W_vecs = [w1, w2, w3, w4, w5, w6, w7]
    
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for q in range(4):
        for k in range(7):
            col = 7 * q + k
            
            # Fill U
            for i in range(4):
                U[i][col] = U_vecs[k][i]
                
            # Fill V
            # Local indices 0,1,2,3 correspond to B00, B01, B10, B11
            # Global indices for block q:
            # B00 -> 0*p + 2q = 2q
            # B01 -> 0*p + 2q+1 = 2q+1
            # B10 -> 1*p + 2q = 8+2q
            # B11 -> 1*p + 2q+1 = 8+2q+1
            v_indices = [2*q, 2*q+1, 8+2*q, 8+2*q+1]
            for i in range(4):
                V[v_indices[i]][col] = V_vecs[k][i]
                
            # Fill W
            # Local indices 0,1,2,3 correspond to C00, C01, C10, C11
            # Global indices for block q:
            # C00 -> 2q*n + 0 = 4q
            # C01 -> 2q*n + 1 = 4q+1
            # C10 -> (2q+1)*n + 0 = 4q+2
            # C11 -> (2q+1)*n + 1 = 4q+3
            w_indices = [4*q, 4*q+1, 4*q+2, 4*q+3]
            for i in range(4):
                W[w_indices[i]][col] = W_vecs[k][i]
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
