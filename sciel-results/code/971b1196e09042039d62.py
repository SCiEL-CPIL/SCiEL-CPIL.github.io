import numpy as np

DIMENSIONS = [2, 2, 11]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=11
    # We use a block decomposition.
    # 5 blocks of 2 columns (rank 7 each, Strassen)
    # 1 block of 1 column (rank 4, Trivial)
    # Total rank = 5 * 7 + 4 = 39
    
    rank = 39
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    current_rank = 0
    
    def add_strassen_block(b_col_start):
        nonlocal current_rank
        b0 = b_col_start
        b1 = b_col_start + 1
        c0 = b_col_start
        c1 = b_col_start + 1
        
        # M1 = (A00 + A11) * (B00 + B11)
        r = current_rank
        U[0][r] = 1; U[3][r] = 1
        V[b0][r] = 1; V[p+b1][r] = 1
        W[c0*n][r] = 1; W[c1*n+1][r] = 1
        current_rank += 1
        
        # M2 = (A10 + A11) * B00
        r = current_rank
        U[2][r] = 1; U[3][r] = 1
        V[b0][r] = 1
        W[c0*n+1][r] = 1; W[c1*n+1][r] = -1
        current_rank += 1
        
        # M3 = A00 * (B01 - B11)
        r = current_rank
        U[0][r] = 1
        V[b1][r] = 1; V[p+b1][r] = -1
        W[c1*n][r] = 1; W[c1*n+1][r] = 1
        current_rank += 1
        
        # M4 = A11 * (B10 - B00)
        r = current_rank
        U[3][r] = 1
        V[p+b0][r] = 1; V[b0][r] = -1
        W[c0*n][r] = 1; W[c0*n+1][r] = 1
        current_rank += 1
        
        # M5 = (A00 + A01) * B11
        r = current_rank
        U[0][r] = 1; U[1][r] = 1
        V[p+b1][r] = 1
        W[c0*n][r] = -1; W[c1*n][r] = 1
        current_rank += 1
        
        # M6 = (A10 - A00) * (B00 + B01)
        r = current_rank
        U[2][r] = 1; U[0][r] = -1
        V[b0][r] = 1; V[b1][r] = 1
        W[c1*n+1][r] = 1
        current_rank += 1
        
        # M7 = (A01 - A11) * (B10 + B11)
        r = current_rank
        U[1][r] = 1; U[3][r] = -1
        V[p+b0][r] = 1; V[p+b1][r] = 1
        W[c0*n][r] = 1
        current_rank += 1

    def add_trivial_block(b_col_start):
        nonlocal current_rank
        b0 = b_col_start
        c0 = b_col_start
        
        # T1: A00 * B0 -> C0
        r = current_rank
        U[0][r] = 1; V[b0][r] = 1; W[c0*n][r] = 1
        current_rank += 1
        
        # T2: A01 * B1 -> C0
        r = current_rank
        U[1][r] = 1; V[p+b0][r] = 1; W[c0*n][r] = 1
        current_rank += 1
        
        # T3: A10 * B0 -> C1
        r = current_rank
        U[2][r] = 1; V[b0][r] = 1; W[c0*n+1][r] = 1
        current_rank += 1
        
        # T4: A11 * B1 -> C1
        r = current_rank
        U[3][r] = 1; V[p+b0][r] = 1; W[c0*n+1][r] = 1
        current_rank += 1

    # 5 Strassen blocks
    for i in range(5):
        add_strassen_block(2 * i)
        
    # 1 Trivial block
    add_trivial_block(10)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
