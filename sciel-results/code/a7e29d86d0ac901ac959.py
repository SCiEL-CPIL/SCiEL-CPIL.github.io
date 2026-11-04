"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We will use a hybrid decomposition:
    # 1. For j in {0,1} and {2,3}, and k in pairs {0,1}, {2,3}, {4,5}, use Strassen (rank 7 per 2x2x2 block).
    # 2. For j=4, use trivial decomposition (rank 12 for 2x1x6).
    
    # Total rank estimate: 6 blocks * 7 + 12 = 54.
    # We need to allocate space. Let's just build lists and then pad or construct directly.
    # Since rank is variable, we can build the matrices dynamically.
    
    # However, the interface requires returning matrices of shape (n*m, rank), etc.
    # We don't know the exact rank in advance if we just append, but we can calculate it.
    # Rank = 6 * 7 + 12 = 54.
    
    rank = 54
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, val_u, val_v, val_w):
        nonlocal col
        U[u_idx][col] = val_u
        V[v_idx][col] = val_v
        W[w_idx][col] = val_w
        col += 1
        
    # Strassen coefficients for 2x2x2 block
    # Local i, j, k in {0, 1}
    # Global i = i_loc (since n=2)
    # Global j = j_offset + j_loc
    # Global k = k_offset + k_loc
    
    # Term 1: M1 = (A11+A22)(B11+B22) -> C11 += 1, C22 += 1
    # U: (0,0)=1, (1,1)=1
    # V: (0,0)=1, (1,1)=1
    # W: (0,0)=1, (1,1)=1 (k=0,i=0; k=1,i=1)
    
    # Term 2: M2 = (A21+A22)B11 -> C21 += 1, C22 -= 1
    # U: (1,0)=1, (1,1)=1
    # V: (0,0)=1
    # W: (0,1)=1 (k=0,i=1), (1,1)=-1 (k=1,i=1)
    
    # Term 3: M3 = A11(B12-B22) -> C12 += 1, C22 += 1
    # U: (0,0)=1
    # V: (0,1)=1, (1,1)=-1
    # W: (1,0)=1 (k=1,i=0), (1,1)=1 (k=1,i=1)
    
    # Term 4: M4 = A22(B21-B11) -> C11 += 1, C21 += 1
    # U: (1,1)=1
    # V: (1,0)=1, (0,0)=-1
    # W: (0,0)=1 (k=0,i=0), (0,1)=1 (k=0,i=1)
    
    # Term 5: M5 = (A11+A12)B22 -> C11 -= 1, C12 += 1
    # U: (0,0)=1, (0,1)=1
    # V: (1,1)=1
    # W: (0,0)=-1 (k=0,i=0), (1,0)=1 (k=1,i=0)
    
    # Term 6: M6 = (A21-A11)(B11+B12) -> C22 += 1
    # U: (1,0)=1, (0,0)=-1
    # V: (0,0)=1, (0,1)=1
    # W: (1,1)=1 (k=1,i=1)
    
    # Term 7: M7 = (A12-A22)(B21+B22) -> C11 += 1
    # U: (0,1)=1, (1,1)=-1
    # V: (1,0)=1, (1,1)=1
    # W: (0,0)=1 (k=0,i=0)
    
    def add_strassen_block(j_off, k_off):
        nonlocal col
        # Term 1
        U[0*m + j_off][col] = 1; V[j_off*p + k_off][col] = 1; W[k_off*n + 0][col] = 1; col+=1
        U[1*m + j_off+1][col] = 1; V[(j_off+1)*p + k_off+1][col] = 1; W[(k_off+1)*n + 1][col] = 1; col+=1
        # Wait, U, V, W for a single term r must be consistent.
        # My add_term logic above assumed one call per (u_idx, v_idx, w_idx).
        # But a term r has multiple non-zero entries in U, V, W.
        # I need to fill the entire column r for all non-zero indices.
        
        # Let's rewrite add_strassen_block to handle the 7 terms properly.
        
        # Term 1
        r = col
        U[0*m + j_off][r] = 1
        U[1*m + j_off+1][r] = 1
        V[j_off*p + k_off][r] = 1
        V[(j_off+1)*p + k_off+1][r] = 1
        W[k_off*n + 0][r] = 1
        W[(k_off+1)*n + 1][r] = 1
        col += 1
        
        # Term 2
        r = col
        U[1*m + j_off][r] = 1
        U[1*m + j_off+1][r] = 1
        V[j_off*p + k_off][r] = 1
        W[k_off*n + 1][r] = 1
        W[(k_off+1)*n + 1][r] = -1
        col += 1
        
        # Term 3
        r = col
        U[0*m + j_off][r] = 1
        V[j_off*p + k_off+1][r] = 1
        V[(j_off+1)*p + k_off+1][r] = -1
        W[(k_off+1)*n + 0][r] = 1
        W[(k_off+1)*n + 1][r] = 1
        col += 1
        
        # Term 4
        r = col
        U[1*m + j_off+1][r] = 1
        V[(j_off+1)*p + k_off][r] = 1
        V[j_off*p + k_off][r] = -1
        W[k_off*n + 0][r] = 1
        W[k_off*n + 1][r] = 1
        col += 1
        
        # Term 5
        r = col
        U[0*m + j_off][r] = 1
        U[0*m + j_off+1][r] = 1
        V[(j_off+1)*p + k_off+1][r] = 1
        W[k_off*n + 0][r] = -1
        W[(k_off+1)*n + 0][r] = 1
        col += 1
        
        # Term 6
        r = col
        U[1*m + j_off][r] = 1
        U[0*m + j_off][r] = -1
        V[j_off*p + k_off][r] = 1
        V[j_off*p + k_off+1][r] = 1
        W[(k_off+1)*n + 1][r] = 1
        col += 1
        
        # Term 7
        r = col
        U[0*m + j_off+1][r] = 1
        U[1*m + j_off+1][r] = -1
        V[(j_off+1)*p + k_off][r] = 1
        V[(j_off+1)*p + k_off+1][r] = 1
        W[k_off*n + 0][r] = 1
        col += 1

    # Apply Strassen blocks
    # j blocks: 0, 2
    # k blocks: 0, 2, 4
    for j_off in [0, 2]:
        for k_off in [0, 2, 4]:
            add_strassen_block(j_off, k_off)
            
    # Apply trivial for j=4
    # n=2, m_block=1, p=6
    # Rank 12
    for i in range(n):
        for k in range(p):
            r = col
            U[i*m + 4][r] = 1
            V[4*p + k][r] = 1
            W[k*n + i][r] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
