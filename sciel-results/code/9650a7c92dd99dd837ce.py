"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
import time
_start_time = time.time()
_DEADLINE = _start_time + 120.0  # Safety margin before 150s kill

def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=15
    
    # Block decomposition: 7 Strassen pairs (rank 49) + 1 schoolbook column (rank 4) = 53
    rank = 53
    
    # U: (n*m, rank) = (4, 53)
    # Indices: 0: A00, 1: A01, 2: A10, 3: A11
    U = [[0.0] * rank for _ in range(n * m)]
    
    # V: (m*p, rank) = (30, 53)
    # Indices: j*p + k. j=0 -> 0..14, j=1 -> 15..29
    V = [[0.0] * rank for _ in range(m * p)]
    
    # W: (p*n, rank) = (30, 53)
    # Indices: k*n + i. k=0 -> 0,1; k=1 -> 2,3; ... k=14 -> 28,29
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Helper to add a Strassen block for column pair (k0, k1)
    def add_strassen_block(base_r, k0, k1):
        # A = [[a, b], [c, d]]
        # B columns k0,k1: e=B[0,k0], f=B[0,k1], g=B[1,k0], h=B[1,k1]
        # C columns k0,k1: C[0,k0]=ae+bg, C[0,k1]=af+bh, C[1,k0]=ce+dg, C[1,k1]=cf+dh
        
        r1 = base_r
        r2 = base_r + 1
        r3 = base_r + 2
        r4 = base_r + 3
        r5 = base_r + 4
        r6 = base_r + 5
        r7 = base_r + 6
        
        # P1 = (a + d) * (e + h)
        U[0][r1] = 1.0  # a
        U[3][r1] = 1.0  # d
        V[0*p + k0][r1] = 1.0  # e
        V[1*p + k1][r1] = 1.0  # h
        W[2*k0][r1] = 1.0    # C0k0
        W[2*k1+1][r1] = 1.0  # C1k1
        
        # P2 = (c + d) * e
        U[2][r2] = 1.0  # c
        U[3][r2] = 1.0  # d
        V[0*p + k0][r2] = 1.0  # e
        W[2*k0+1][r2] = 1.0  # C1k0
        W[2*k1+1][r2] = -1.0 # C1k1
        
        # P3 = a * (f - h)
        U[0][r3] = 1.0  # a
        V[0*p + k1][r3] = 1.0  # f
        V[1*p + k1][r3] = -1.0 # h
        W[2*k1][r3] = 1.0    # C0k1
        W[2*k1+1][r3] = 1.0  # C1k1
        
        # P4 = d * (g - e)
        U[3][r4] = 1.0  # d
        V[1*p + k0][r4] = 1.0  # g
        V[0*p + k0][r4] = -1.0 # e
        W[2*k0][r4] = 1.0    # C0k0
        W[2*k0+1][r4] = 1.0  # C1k0
        
        # P5 = (a + b) * h
        U[0][r5] = 1.0  # a
        U[1][r5] = 1.0  # b
        V[1*p + k1][r5] = 1.0  # h
        W[2*k0][r5] = -1.0   # C0k0
        W[2*k1][r5] = 1.0    # C0k1
        
        # P6 = (c - a) * (e + f)
        U[2][r6] = 1.0  # c
        U[0][r6] = -1.0 # a
        V[0*p + k0][r6] = 1.0  # e
        V[0*p + k1][r6] = 1.0  # f
        W[2*k1+1][r6] = 1.0  # C1k1
        
        # P7 = (b - d) * (g + h)
        U[1][r7] = 1.0  # b
        U[3][r7] = -1.0 # d
        V[1*p + k0][r7] = 1.0  # g
        V[1*p + k1][r7] = 1.0  # h
        W[2*k0][r7] = 1.0    # C0k0

    # Add 7 Strassen blocks for pairs (0,1), (2,3), ..., (12,13)
    for i in range(7):
        if time.time() > _DEADLINE - 5.0:
            break
        k0 = 2 * i
        k1 = 2 * i + 1
        base_r = 7 * i
        add_strassen_block(base_r, k0, k1)
        
    # Add trivial schoolbook block for column 14
    # C014 = A00*B014 + A01*B114
    # C114 = A10*B014 + A11*B114
    base_r = 49
    r1 = base_r
    r2 = base_r + 1
    r3 = base_r + 2
    r4 = base_r + 3
    
    # Term 1: A00 * B014
    U[0][r1] = 1.0
    V[0*p + 14][r1] = 1.0
    W[2*14][r1] = 1.0  # C014
    
    # Term 2: A01 * B114
    U[1][r2] = 1.0
    V[1*p + 14][r2] = 1.0
    W[2*14][r2] = 1.0  # C014
    
    # Term 3: A10 * B014
    U[2][r3] = 1.0
    V[0*p + 14][r3] = 1.0
    W[2*14+1][r3] = 1.0  # C114
    
    # Term 4: A11 * B114
    U[3][r4] = 1.0
    V[1*p + 14][r4] = 1.0
    W[2*14+1][r4] = 1.0  # C114
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
