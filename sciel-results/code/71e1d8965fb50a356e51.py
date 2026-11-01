import time

DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin from 120s limit
    
    n, m, p = DIMENSIONS
    # n=2, m=2, p=9
    # U: (n*m, rank) = (4, rank)
    # V: (m*p, rank) = (18, rank)
    # W: (p*n, rank) = (18, rank)
    
    # Rank calculation:
    # Split p=9 into four 2x2 blocks and one 2x1 block.
    # Rank(M_2,2,2) = 7 (Strassen)
    # Rank(M_2,2,1) = 4 (Trivial)
    # Total Rank = 4 * 7 + 4 = 32.
    
    rank = 32
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a Strassen block for cols k0, k1
    def add_strassen(k0, k1):
        nonlocal col
        
        # M1: (a00+a11)(b0k0+b1k1)
        U[0][col] = 1
        U[3][col] = 1
        V[0*p + k0][col] = 1
        V[1*p + k1][col] = 1
        # C0k0: +1, C0k1: +1, C1k0: -1, C1k1: -1
        W[k0*n + 0][col] = 1
        W[k1*n + 0][col] = 1
        W[k0*n + 1][col] = -1
        W[k1*n + 1][col] = -1
        col += 1
        
        # M2: (a10+a11)b0k0
        U[2][col] = 1
        U[3][col] = 1
        V[0*p + k0][col] = 1
        # C0k0: 0, C0k1: 0, C1k0: +1, C1k1: -1
        W[k0*n + 1][col] = 1
        W[k1*n + 1][col] = -1
        col += 1
        
        # M3: a00(b0k1-b1k1)
        U[0][col] = 1
        V[0*p + k1][col] = 1
        V[1*p + k1][col] = -1
        # C0k0: 0, C0k1: +1, C1k0: 0, C1k1: +1
        W[k1*n + 0][col] = 1
        W[k1*n + 1][col] = 1
        col += 1
        
        # M4: a11(b1k0-b0k0)
        U[3][col] = 1
        V[1*p + k0][col] = 1
        V[0*p + k0][col] = -1
        # C0k0: -1, C0k1: 0, C1k0: +1, C1k1: 0
        W[k0*n + 0][col] = -1
        W[k0*n + 1][col] = 1
        col += 1
        
        # M5: (a00+a01)b1k1
        U[0][col] = 1
        U[1][col] = 1
        V[1*p + k1][col] = 1
        # C0k0: +1, C0k1: +1, C1k0: 0, C1k1: 0
        W[k0*n + 0][col] = 1
        W[k1*n + 0][col] = 1
        col += 1
        
        # M6: (a10-a00)(b0k0+b0k1)
        U[2][col] = 1
        U[0][col] = -1
        V[0*p + k0][col] = 1
        V[0*p + k1][col] = 1
        # C0k0: 0, C0k1: 0, C1k0: 0, C1k1: +1
        W[k1*n + 1][col] = 1
        col += 1
        
        # M7: (a01-a11)(b1k0+b1k1)
        U[1][col] = 1
        U[3][col] = -1
        V[1*p + k0][col] = 1
        V[1*p + k1][col] = 1
        # C0k0: -1, C0k1: 0, C1k0: 0, C1k1: 0
        W[k0*n + 0][col] = -1
        col += 1

    # Helper to add a Trivial block for col k
    def add_trivial(k):
        nonlocal col
        # 1. a00 b0k -> C0k
        U[0][col] = 1
        V[0*p + k][col] = 1
        W[k*n + 0][col] = 1
        col += 1
        
        # 2. a01 b1k -> C0k
        U[1][col] = 1
        V[1*p + k][col] = 1
        W[k*n + 0][col] = 1
        col += 1
        
        # 3. a10 b0k -> C1k
        U[2][col] = 1
        V[0*p + k][col] = 1
        W[k*n + 1][col] = 1
        col += 1
        
        # 4. a11 b1k -> C1k
        U[3][col] = 1
        V[1*p + k][col] = 1
        W[k*n + 1][col] = 1
        col += 1

    # Apply blocks
    # Cols 0-1
    add_strassen(0, 1)
    # Cols 2-3
    add_strassen(2, 3)
    # Cols 4-5
    add_strassen(4, 5)
    # Cols 6-7
    add_strassen(6, 7)
    # Col 8
    add_trivial(8)
    
    if time.time() > deadline:
        # Return what we have
        pass
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
