"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
import time
_start_time = time.time()
_DEADLINE = _start_time + 130  # 130s safety margin

def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=15
    
    # Rank 53 decomposition: 7 Strassen blocks (rank 7 each) + 1 Naive block (rank 4)
    rank = 53
    # U shape: (n*m, rank) = (4, 53)
    # V shape: (m*p, rank) = (30, 53)
    # W shape: (p*n, rank) = (30, 53)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_strassen_block(base_r, k0, k1):
        """
        Adds a Strassen block for columns k0, k1 of B.
        k0, k1 are column indices in B.
        """
        r1, r2, r3, r4, r5, r6, r7 = base_r, base_r+1, base_r+2, base_r+3, base_r+4, base_r+5, base_r+6
        
        # M1 = (A00 + A11) * (B00 + B10)
        # U: A00=1, A11=1
        U[0][r1] = 1
        U[3][r1] = 1
        # V: B00=1, B10=1
        V[0*p+k0][r1] = 1
        V[1*p+k0][r1] = 1
        # W: C00=1, C11=1
        W[0*n+0][r1] = 1
        W[1*n+1][r1] = 1
        
        # M2 = (A10 + A11) * B00
        # U: A10=1, A11=1
        U[2][r2] = 1
        U[3][r2] = 1
        # V: B00=1
        V[0*p+k0][r2] = 1
        # W: C10=1, C11=-1
        W[0*n+1][r2] = 1
        W[1*n+1][r2] = -1
        
        # M3 = A00 * (B01 - B11)
        # U: A00=1
        U[0][r3] = 1
        # V: B01=1, B11=-1
        V[0*p+k1][r3] = 1
        V[1*p+k1][r3] = -1
        # W: C01=1, C11=1
        W[1*n+0][r3] = 1
        W[1*n+1][r3] = 1
        
        # M4 = A11 * (B11 - B01)
        # U: A11=1
        U[3][r4] = 1
        # V: B11=1, B01=-1
        V[1*p+k1][r4] = 1
        V[0*p+k1][r4] = -1
        # W: C00=1, C10=1
        W[0*n+0][r4] = 1
        W[0*n+1][r4] = 1
        
        # M5 = (A00 + A01) * B11
        # U: A00=1, A01=1
        U[0][r5] = 1
        U[1][r5] = 1
        # V: B11=1
        V[1*p+k1][r5] = 1
        # W: C00=-1, C01=1
        W[0*n+0][r5] = -1
        W[1*n+0][r5] = 1
        
        # M6 = (A10 - A00) * (B00 + B01)
        # U: A10=1, A00=-1
        U[2][r6] = 1
        U[0][r6] = -1
        # V: B00=1, B01=1
        V[0*p+k0][r6] = 1
        V[0*p+k1][r6] = 1
        # W: C11=1
        W[1*n+1][r6] = 1
        
        # M7 = (A01 - A11) * (B10 + B11)
        # U: A01=1, A11=-1
        U[1][r7] = 1
        U[3][r7] = -1
        # V: B10=1, B11=1
        V[1*p+k0][r7] = 1
        V[1*p+k1][r7] = 1
        # W: C00=1
        W[0*n+0][r7] = 1

    def add_naive_block(base_r, k):
        """
        Adds a naive multiplication block for column k of B.
        """
        r1, r2, r3, r4 = base_r, base_r+1, base_r+2, base_r+3
        
        # C00 += A00 * B00
        U[0][r1] = 1
        V[0*p+k][r1] = 1
        W[0*n+0][r1] = 1
        
        # C01 += A01 * B10
        U[1][r2] = 1
        V[1*p+k][r2] = 1
        W[1*n+0][r2] = 1
        
        # C10 += A10 * B00
        U[2][r3] = 1
        V[0*p+k][r3] = 1
        W[0*n+1][r3] = 1
        
        # C11 += A11 * B10
        U[3][r4] = 1
        V[1*p+k][r4] = 1
        W[1*n+1][r4] = 1

    # Apply 7 Strassen blocks for column pairs (0,1), (2,3), ..., (12,13)
    for i in range(7):
        k0 = 2 * i
        k1 = 2 * i + 1
        base_r = 7 * i
        add_strassen_block(base_r, k0, k1)
        
    # Apply 1 Naive block for column 14
    base_r = 49
    add_naive_block(base_r, 14)
    
    # Verify quickly (optional but good for debugging)
    # def _verify(U, V, W):
    #     rank = len(U[0])
    #     for i in range(n):
    #         for j in range(m):
    #             for k in range(p):
    #                 s = 0
    #                 for r in range(rank):
    #                     s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
    #                 if s != 1:
    #                     return False
    #     return True
    # if not _verify(U, V, W):
    #     return None

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
