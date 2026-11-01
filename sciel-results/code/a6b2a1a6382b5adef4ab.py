"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    # Strassen-based rank-35 decomposition for 2x2 * 2x10.
    # The tensor is block-diagonal: 5 independent 2x2x2 blocks (one per column pair).
    # Strassen's algorithm gives rank 7 per 2x2x2 block, so total rank = 5*7 = 35.
    n, m, p = DIMENSIONS
    # n=2, m=2, p=10
    # 5 blocks, each covering 2 columns of the right matrix
    num_blocks = p // 2  # 5
    
    # Strassen's 7 products for a 2x2 * 2x2 multiplication:
    # Let A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    # C = A*B
    #
    # Strassen:
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b12-b22)
    # M4 = a22(b21-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    #
    # C11 = M1 - M4 + M6
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M3 + M7
    #
    # For each block b (0..4), the left matrix is the same 2x2 A (shared),
    # but the right matrix is a different 2x2 B_b.
    #
    # U has shape (n*m, rank) = (4, 35)
    # V has shape (m*p, rank) = (20, 35)
    # W has shape (p*n, rank) = (20, 35)
    #
    # Indexing:
    # U[i*m + j, c] where i in 0..1, j in 0..1
    # V[j*p + k, c] where j in 0..1, k in 0..9
    # W[k*n + i, c] where k in 0..9, i in 0..1
    #
    # For block b (columns 2b, 2b+1 of the right matrix):
    # B_b = [[B_{0,2b}, B_{0,2b+1}], [B_{1,2b}, B_{1,2b+1}]]
    # In V indexing: V[j*p + k, c]
    #   j=0, k=2b -> B_{0,2b}
    #   j=0, k=2b+1 -> B_{0,2b+1}
    #   j=1, k=2b -> B_{1,2b}
    #   j=1, k=2b+1 -> B_{1,2b+1}
    
    rank = 7 * num_blocks  # 35
    U = [[0] * rank for _ in range(n * m)]  # (4, 35)
    V = [[0] * rank for _ in range(m * p)]  # (20, 35)
    W = [[0] * rank for _ in range(p * n)]  # (20, 35)
    
    # For each block, we have 7 Strassen terms.
    # For block b, the column offset in U,V,W is b*7 + s where s is 0..6.
    
    for b in range(num_blocks):
        col_base = b * 7
        c1, c2 = 2 * b, 2 * b + 1  # column indices in the right matrix
        
        # M1 = (a11+a22)(b11+b22)
        # U: a11 -> U[0, col], a22 -> U[3, col]
        # V: b11 -> V[0*p+c1, col], b22 -> V[1*p+c2, col]
        col = col_base + 0
        U[0][col] += 1   # a11
        U[3][col] += 1   # a22
        V[0 * p + c1][col] += 1  # b11
        V[1 * p + c2][col] += 1  # b22
        # W: C11 gets +M1, C22 gets +M1
        # C11 -> W[c1*n + 0, col] = W[0, col]
        # C22 -> W[c2*n + 1, col] = W[1, col]
        W[c1 * n + 0][col] += 1
        W[c2 * n + 1][col] += 1
        
        # M2 = (a21+a22)b11
        col = col_base + 1
        U[2][col] += 1   # a21
        U[3][col] += 1   # a22
        V[0 * p + c1][col] += 1  # b11
        # C21 gets +M2 -> W[c1*n + 1, col] = W[1, col]
        W[c1 * n + 1][col] += 1
        
        # M3 = a11(b12-b22)
        col = col_base + 2
        U[0][col] += 1   # a11
        V[0 * p + c2][col] += 1  # b12
        V[1 * p + c2][col] += -1  # -b22
        # C12 gets +M3 -> W[c2*n + 0, col] = W[0, col]
        # C22 gets -M3 -> W[c2*n + 1, col] = W[1, col]
        W[c2 * n + 0][col] += 1
        W[c2 * n + 1][col] += -1
        
        # M4 = a22(b21-b11)
        col = col_base + 3
        U[3][col] += 1   # a22
        V[1 * p + c1][col] += 1  # b21
        V[0 * p + c1][col] += -1  # -b11
        # C11 gets -M4 -> W[c1*n + 0, col] = W[0, col]
        # C21 gets +M4 -> W[c1*n + 1, col] = W[1, col]
        W[c1 * n + 0][col] += -1
        W[c1 * n + 1][col] += 1
        
        # M5 = (a11+a12)b22
        col = col_base + 4
        U[0][col] += 1   # a11
        U[1][col] += 1   # a12
        V[1 * p + c2][col] += 1  # b22
        # C12 gets +M5 -> W[c2*n + 0, col] = W[0, col]
        W[c2 * n + 0][col] += 1
        
        # M6 = (a21-a11)(b11+b12)
        col = col_base + 5
        U[2][col] += 1   # a21
        U[0][col] += -1  # -a11
        V[0 * p + c1][col] += 1  # b11
        V[0 * p + c2][col] += 1  # b12
        # C11 gets +M6 -> W[c1*n + 0, col] = W[0, col]
        W[c1 * n + 0][col] += 1
        
        # M7 = (a12-a22)(b21+b22)
        col = col_base + 6
        U[1][col] += 1   # a12
        U[3][col] += -1  # -a22
        V[1 * p + c1][col] += 1  # b21
        V[1 * p + c2][col] += 1  # b22
        # C22 gets +M7 -> W[c2*n + 1, col] = W[1, col]
        W[c2 * n + 1][col] += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
