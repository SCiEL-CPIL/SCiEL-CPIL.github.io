"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
import time
_start_time = time.time()

def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=15
    
    # We use a block decomposition.
    # 7 pairs of columns (0-1, 2-3, ..., 12-13) using Strassen (rank 7 each)
    # 1 single column (14) using trivial (rank 4)
    # Total rank = 7 * 7 + 4 = 53
    
    rank = 53
    U = [[0] * rank for _ in range(n * m)]  # 4 x 53
    V = [[0] * rank for _ in range(m * p)]  # 30 x 53
    W = [[0] * rank for _ in range(p * n)]  # 30 x 53
    
    # A indices in U:
    # 0: A00, 1: A01, 2: A10, 3: A11
    
    # B indices in V:
    # j*p + k. j=0 -> 0..14, j=1 -> 15..29
    
    # C indices in W:
    # k*n + i. k=0 -> 0,1; k=1 -> 2,3; ... k=14 -> 28,29
    
    # Helper to set Strassen block for column pair (2k, 2k+1)
    # Base rank offset for this block
    def add_strassen_block(base_r, k0, k1):
        # k0, k1 are the column indices in B (and C)
        # e = B0k0, f = B0k1, g = B1k0, h = B1k1
        # P1 = (A00+A10)(B0k0+B1k1) -> (a+d)(e+h)
        # P2 = (A10+A11)B0k0 -> (c+d)e
        # P3 = A00(B0k1-B1k1) -> a(f-h)
        # P4 = A11(B1k0-B0k0) -> d(g-e)
        # P5 = (A00+A01)B1k1 -> (a+b)h
        # P6 = (A10-A00)(B0k0+B0k1) -> (c-a)(e+f)
        # P7 = (A01-A11)(B1k0+B1k1) -> (b-d)(g+h)
        
        r1 = base_r
        r2 = base_r + 1
        r3 = base_r + 2
        r4 = base_r + 3
        r5 = base_r + 4
        r6 = base_r + 5
        r7 = base_r + 6
        
        # P1
        U[0][r1] = 1 # A00
        U[3][r1] = 1 # A11 (Wait, d is A11. a is A00. a+d is A00+A11)
        # Correction: A = [[a,b],[c,d]]. Indices: 0:a, 1:b, 2:c, 3:d.
        # So a+d is U[0]+U[3].
        # My previous comment said A10, but index 3 is A11 (i=1, j=1).
        # Index 2 is A10 (i=1, j=0).
        # So:
        # P1: (A00+A11)
        U[0][r1] = 1
        U[3][r1] = 1
        V[0*p + k0][r1] = 1 # B0k0 (e)
        V[1*p + k1][r1] = 1 # B1k1 (h)
        
        # P2: (A10+A11) B0k0 -> (c+d)e
        U[2][r2] = 1 # A10 (c)
        U[3][r2] = 1 # A11 (d)
        V[0*p + k0][r2] = 1 # B0k0 (e)
        
        # P3: A00 (B0k1 - B1k1) -> a(f-h)
        U[0][r3] = 1 # A00 (a)
        V[0*p + k1][r3] = 1 # B0k1 (f)
        V[1*p + k1][r3] = -1 # B1k1 (h)
        
        # P4: A11 (B1k0 - B0k0) -> d(g-e)
        U[3][r4] = 1 # A11 (d)
        V[1*p + k0][r4] = 1 # B1k0 (g)
        V[0*p + k0][r4] = -1 # B0k0 (e)
        
        # P5: (A00+A01) B1k1 -> (a+b)h
        U[0][r5] = 1 # A00 (a)
        U[1][r5] = 1 # A01 (b)
        V[1*p + k1][r5] = 1 # B1k1 (h)
        
        # P6: (A10-A00) (B0k0+B0k1) -> (c-a)(e+f)
        U[2][r6] = 1 # A10 (c)
        U[0][r6] = -1 # A00 (a)
        V[0*p + k0][r6] = 1 # B0k0 (e)
        V[0*p + k1][r6] = 1 # B0k1 (f)
        
        # P7: (A01-A11) (B1k0+B1k1) -> (b-d)(g+h)
        U[1][r7] = 1 # A01 (b)
        U[3][r7] = -1 # A11 (d)
        V[1*p + k0][r7] = 1 # B1k0 (g)
        V[1*p + k1][r7] = 1 # B1k1 (h)
        
        # Reconstruction W
        # C0k0 = P1 + P4 - P5 + P7
        # C1k0 = -P1 + P2 + P5 + P7
        # C0k1 = P3 - P4 + P5 - P7
        # C1k1 = -P2 + P1 + P3 - P6
        
        # W index: k*n + i
        # k0 -> 2*k0, 2*k0+1
        # k1 -> 2*k1, 2*k1+1
        
        # C0k0 (i=0, k=k0) -> W[2*k0]
        W[2*k0][r1] = 1
        W[2*k0][r4] = 1
        W[2*k0][r5] = -1
        W[2*k0][r7] = 1
        
        # C1k0 (i=1, k=k0) -> W[2*k0+1]
        W[2*k0+1][r1] = -1
        W[2*k0+1][r2] = 1
        W[2*k0+1][r5] = 1
        W[2*k0+1][r7] = 1
        
        # C0k1 (i=0, k=k1) -> W[2*k1]
        W[2*k1][r3] = 1
        W[2*k1][r4] = -1
        W[2*k1][r5] = 1
        W[2*k1][r7] = -1
        
        # C1k1 (i=1, k=k1) -> W[2*k1+1]
        W[2*k1+1][r2] = -1
        W[2*k1+1][r1] = 1
        W[2*k1+1][r3] = 1
        W[2*k1+1][r6] = -1
        
    # Add 7 Strassen blocks for pairs (0,1), (2,3), ..., (12,13)
    for i in range(7):
        k0 = 2 * i
        k1 = 2 * i + 1
        base_r = 7 * i
        add_strassen_block(base_r, k0, k1)
        
    # Add trivial block for column 14
    base_r = 49
    r1 = base_r
    r2 = base_r + 1
    r3 = base_r + 2
    r4 = base_r + 3
    
    # Q1 = A00 B014
    U[0][r1] = 1
    V[0*p + 14][r1] = 1
    W[2*14][r1] = 1 # C014
    
    # Q2 = A01 B114
    U[1][r2] = 1
    V[1*p + 14][r2] = 1
    W[2*14][r2] = 1 # C014
    
    # Q3 = A10 B014
    U[2][r3] = 1
    V[0*p + 14][r3] = 1
    W[2*14+1][r3] = 1 # C114
    
    # Q4 = A11 B114
    U[3][r4] = 1
    V[1*p + 14][r4] = 1
    W[2*14+1][r4] = 1 # C114
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
