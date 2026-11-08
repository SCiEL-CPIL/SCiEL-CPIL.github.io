"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-minimal (or near-minimal) decomposition for 2x5 x 5x6 matrix multiplication.
    Uses Kronecker product structure to break 5 = 2 + 3, solving 2x2x6 and 2x3x6
    independently, then combining the factorizations.
    
    The decomposition exploits:
    - 2x2x2 Strassen for the 2x2 blocks
    - Known low-rank decompositions for 2x2x6 and 2x3x6
    """
    n, m, p = 2, 5, 6
    
    # Build Strassen 2x2x2 decomposition
    # For multiplying 2x2 by 2x2: rank 7
    # U indices: (i,j) in [0,1]^2 -> 4 entries
    # V indices: (j,k) in [0,1]^2 -> 4 entries  
    # W indices: (k,i) in [0,1]^2 -> 4 entries
    
    def strassen_2x2():
        """Return (U, V, W) for 2x2x2 matrix mult with rank 7."""
        # Standard Strassen
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        # U[i*m+j, c] = U[i*2+j, c]
        # V[j*p+k, c] = V[j*2+k, c]
        # W[k*n+i, c] = W[k*2+i, c]
        
        # Strassen variables:
        # M1 = (a00+a11)(b00+b11)
        # M2 = (a10+a11)b00
        # M3 = a00(b01-b11)
        # M4 = a11(b10-b00)
        # M5 = (a00+a01)b11
        # M6 = (a10-a00)(b00+b01)
        # M7 = (a01-a11)(b10+b11)
        
        # U row index: i*2+j
        # V row index: j*2+k
        # W row index: k*2+i
        
        # M1
        U[0][0] = 1  # (0,0)
        U[3][0] = 1  # (1,1)
        V[0][0] = 1  # (0,0)
        V[2][0] = 1  # (1,0)
        V[3][0] = 1  # (1,1)
        W[1][0] = 1  # (0,1)
        W[0][0] = 1  # (0,0)
        W[3][0] = 1  # (1,1)
        
        # M2
        U[2][1] = 1  # (1,0)
        U[3][1] = 1  # (1,1)
        V[0][1] = 1  # (0,0)
        W[1][1] = 1  # (0,1)
        
        # M3
        U[0][2] = 1  # (0,0)
        V[1][2] = 1  # (0,1)
        V[3][2] = -1 # (1,1)
        W[0][2] = 1  # (0,0)
        
        # M4
        U[3][3] = 1  # (1,1)
        V[2][3] = 1  # (1,0)
        V[0][3] = -1 # (0,0)
        W[1][3] = 1  # (0,1)
        
        # M5
        U[0][4] = 1  # (0,0)
        U[1][4] = 1  # (0,1)
        V[3][4] = 1  # (1,1)
        W[3][4] = 1  # (1,1)
        
        # M6
        U[2][5] = 1  # (1,0)
        U[0][5] = -1 # (0,0)
        V[0][5] = 1  # (0,0)
        V[1][5] = 1  # (0,1)
        W[1][5] = 1  # (0,1)
        
        # M7
        U[1][6] = 1  # (0,1)
        U[3][6] = -1 # (1,1)
        V[2][6] = 1  # (1,0)
        V[3][6] = 1  # (1,1)
        W[3][6] = 1  # (1,1)
        
        return U, V, W
    
    # Now handle 2x2x6: split the p=6 into three 2-column blocks
    def mult_2x2_2x6():
        """2x2 times 2x6. Split into three 2x2x2 problems."""
        U2x2, V2x2, W2x2 = strassen_2x2()
        # For p=6, we have 3 independent 2x2x2 blocks
        rank = 7 * 3
        U = [[0]*rank for _ in range(4)]
        V = [[0]*rank for _ in range(12)]  # 2*6
        W = [[0]*rank for _ in range(12)]  # 6*2
        for blk in range(3):
            for c in range(7):
                global_c = blk*7 + c
                for i in range(2):
                    for j in range(2):
                        idx_u = i*2 + j
                        U[idx_u][global_c] = U2x2[idx_u][c]
                    for k in range(2):
                        idx_v = j*2 + k
                        # V row: j*p+k = j*6+k
                        V[j*6 + k][global_c] = V2x2[idx_v][c]
                    for kk in range(2):
                        # W row: k*n+i where k is in [0,2), so kk = k
                        idx_w = kk*2 + i
                        # W[k*n+i, c] = W[kk*2+i, c]
                        # But k = kk + blk*2, so row index = (kk+blk*2)*2 + i
                        W[(kk + blk*2)*2 + i][global_c] = W2x2[idx_w][c]
        return U, V, W, rank
    
    # Handle 2x3x2: use known rank 8 decomposition (or fallback)
    # For 2x3 times 3x2, rank is at most 6 (schoolbook), can we do better?
    # Actually for m=3, n=2, p=2: rank <= 8 via some constructions
    # Let's use a simple rank-6 schoolbook or try to find better
    def mult_2x3_2x3():
        """2x3 times 3x2. Use rank 6 schoolbook first."""
        n, m, p = 2, 3, 2
        rank = n*m*p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return U, V, W, rank
    
    # Handle 2x3x6: split into three 2x3x2 blocks
    def mult_2x3_2x6():
        n, m, p = 2, 3, 6
        rank_2x3_2x2 = 6
        rank = rank_2x3_2x2 * 3
        U = [[0]*rank for _ in range(n*m)]  # 6
        V = [[0]*rank for _ in range(m*p)]  # 18
        W = [[0]*rank for _ in range(p*n)]  # 12
        for blk in range(3):
            U_blk, V_blk, W_blk, _ = mult_2x3_2x3()
            for c in range(rank_2x3_2x2):
                global_c = blk*rank_2x3_2x2 + c
                for i in range(n):
                    for j in range(m):
                        U[i*m+j][global_c] = U_blk[i*m+j][c]
                    for k in range(2):
                        V[j*6+k][global_c] = V_blk[j*2+k][c]
                    for kk in range(2):
                        W[(kk+blk*2)*n+i][global_c] = W_blk[kk*n+i][c]
        return U, V, W, rank
    
    # Now combine 2x2x6 and 2x3x6 to get 2x5x6
    U_a, V_a, W_a, rank_a = mult_2x2_2x6()
    U_b, V_b, W_b, rank_b = mult_2x3_2x6()
    
    rank = rank_a + rank_b
    # U_a has 4 rows (i*2+j for i in 0,1, j in 0,1)
    # U_b has 6 rows (i*3+j for i in 0,1, j in 0,1,2)
    # Full U should have n*m = 10 rows: indices i*5+j
    # j splits into j=0,1 (from first 2x2) and j=2,3,4 (from 2x3)
    
    U = [[0]*rank for _ in range(10)]
    V = [[0]*rank for _ in range(30)]  # m*p = 5*6
    W = [[0]*rank for _ in range(12)]   # p*n = 6*2
    
    # Add 2x2x6 part (j=0,1)
    for i in range(2):
        for j in range(2):
            src = i*2+j
            dst = i*5+j
            for c in range(rank_a):
                U[dst][c] = U_a[src][c]
    
    # Add 2x3x6 part (j=2,3,4)
    for i in range(2):
        for j_local in range(3):
            src = i*3+j_local
            dst = i*5+(j_local+2)
            for c in range(rank_b):
                U[dst][rank_a + c] = U_b[src][c]
    
    # V part: j*p+k, j in 0..4, k in 0..5
    # For 2x2x6: j=0,1, k=0..5
    # For 2x3x6: j=2,3,4, k=0..5
    for j in range(2):
        for k in range(6):
            src = j*2+k
            dst = j*6+k
            for c in range(rank_a):
                V[dst][c] = V_a[src][c]
    
    for j_local in range(3):
        j = j_local + 2
        for k in range(6):
            src = j_local*2+k
            dst = j*6+k
            for c in range(rank_b):
                V[dst][rank_a + c] = V_b[src][c]
    
    # W part: k*n+i, k in 0..5, i in 0,1
    # Both parts cover same k range
    for k in range(6):
        for i in range(2):
            src = k*2+i
            dst = k*2+i
            for c in range(rank_a):
                W[dst][c] += W_a[src][c]
            for c in range(rank_b):
                W[dst][rank_a + c] += W_b[src][c]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
