"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    # Strassen-based rank-35 decomposition for 2x2 * 2x10 multiplication.
    # The 2x10 matrix has 5 independent 2x2 column blocks.
    # For each block, apply Strassen's rank-7 algorithm.
    # Total rank = 7 * 5 = 35.
    #
    # Strassen's algorithm for C = A * B where A, B are 2x2:
    #   A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    #   C = [[a*e + b*g, a*f + b*h], [c*e + d*g, c*f + d*h]]
    #
    # Strassen's 7 products:
    #   P1 = (a + d)(e + h)
    #   P2 = (c + d)e
    #   P3 = a(f - h)
    #   P4 = d(g - e)
    #   P5 = (a + b)h
    #   P6 = (c - a)(e + f)
    #   P7 = (b - d)(g + h)
    #
    #   C[0,0] = P1 + P4 - P5 + P7
    #   C[0,1] = P3 + P5
    #   C[1,0] = P2 + P4
    #   C[1,1] = P1 - P2 + P3 + P6
    #
    # For 2x2 * 2x10:
    #   A = [[a, b], [c, d]] (2x2)
    #   B = [[b00, b01, b02, b03, b04, b05, b06, b07, b08, b09],
    #        [b10, b11, b12, b13, b14, b15, b16, b17, b18, b19]] (2x10)
    #
    # Split B into 5 column blocks of width 2:
    #   B_j = [[b0(2j), b0(2j+1)], [b1(2j), b1(2j+1)]] for j=0..4
    #
    # For each block j, compute C_j = A * B_j (2x2 result)
    # The output C is 2x10 with columns from C_0, C_1, ..., C_4.
    #
    # Trace convention: W[k*n + i, r] where k is column index of output, i is row index.
    # Output C[i, k] = sum_r U[i*m + j, r] * V[j*p + k', r] ... 
    # Wait, let me re-read the interface.
    #
    # U: (n*m, rank) = (4, rank)
    # V: (m*p, rank) = (20, rank)
    # W: (p*n, rank) = (20, rank)
    #
    # The multiplication tensor T[i, j, k] where:
    #   i indexes rows of A (0..n-1)
    #   j indexes columns of A / rows of B (0..m-1)
    #   k indexes columns of B (0..p-1)
    #
    # C[i, k] = sum_j A[i, j] * B[j, k]
    #
    # U[i*m + j, r] corresponds to A[i, j]
    # V[j*p + k, r] corresponds to B[j, k]
    # W[k*n + i, r] corresponds to C[i, k] (trace convention: k*n + i)
    #
    # So for each rank term r:
    #   sum over i,j,k: U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i,j,k]
    #
    # Strassen for block j (column block index b = 0..4):
    #   Let e = B[0, 2b], f = B[0, 2b+1], g = B[1, 2b], h = B[1, 2b+1]
    #
    #   P1 = (a + d)(e + h)
    #   P2 = (c + d)e
    #   P3 = a(f - h)
    #   P4 = d(g - e)
    #   P5 = (a + b)h
    #   P6 = (c - a)(e + f)
    #   P7 = (b - d)(g + h)
    #
    #   C[0, 2b]   = P1 + P4 - P5 + P7
    #   C[0, 2b+1] = P3 + P5
    #   C[1, 2b]   = P2 + P4
    #   C[1, 2b+1] = P1 - P2 + P3 + P6
    #
    # For each Strassen product, we have:
    #   U entries: coefficients of a, b, c, d (indices 0,1,2,3 in U)
    #   V entries: coefficients of e, f, g, h (indices in V)
    #   W entries: coefficients for C[0,2b], C[0,2b+1], C[1,2b], C[1,2b+1]
    #
    # U index: i*m + j where i is row of A, j is col of A
    #   a -> U[0*2+0] = U[0]
    #   b -> U[0*2+1] = U[1]
    #   c -> U[1*2+0] = U[2]
    #   d -> U[1*2+1] = U[3]
    #
    # V index: j*p + k where j is row of B, k is col of B
    #   e = B[0, 2b] -> V[0*10 + 2b] = V[2b]
    #   f = B[0, 2b+1] -> V[0*10 + 2b+1] = V[2b+1]
    #   g = B[1, 2b] -> V[1*10 + 2b] = V[10+2b]
    #   h = B[1, 2b+1] -> V[1*10 + 2b+1] = V[10+2b+1]
    #
    # W index: k*n + i where k is col of C, i is row of C
    #   C[0, 2b] -> W[2b*2 + 0] = W[4b]
    #   C[0, 2b+1] -> W[(2b+1)*2 + 0] = W[4b+2]
    #   C[1, 2b] -> W[2b*2 + 1] = W[4b+1]
    #   C[1, 2b+1] -> W[(2b+1)*2 + 1] = W[4b+3]
    
    n, m, p = DIMENSIONS  # n=2, m=2, p=10
    num_blocks = p // 2   # 5 blocks
    
    rank = 7 * num_blocks  # 35
    U = [[0] * rank for _ in range(n * m)]  # (4, 35)
    V = [[0] * rank for _ in range(m * p)]  # (20, 35)
    W = [[0] * rank for _ in range(p * n)]  # (20, 35)
    
    for b in range(num_blocks):
        base = 7 * b  # base column index in U, V, W
        
        # P1 = (a + d)(e + h)
        col = base
        U[0][col] = 1  # a
        U[3][col] = 1  # d
        V[2*b][col] = 1  # e
        V[10+2*b+1][col] = 1  # h
        W[4*b][col] = 1  # C[0, 2b]
        W[4*b+3][col] = 1  # C[1, 2b+1]
        
        # P2 = (c + d)e
        col = base + 1
        U[2][col] = 1  # c
        U[3][col] = 1  # d
        V[2*b][col] = 1  # e
        W[4*b+1][col] = 1  # C[1, 2b]
        W[4*b+3][col] = -1  # C[1, 2b+1]
        
        # P3 = a(f - h)
        col = base + 2
        U[0][col] = 1  # a
        V[2*b+1][col] = 1  # f
        V[10+2*b+1][col] = -1  # -h
        W[4*b+2][col] = 1  # C[0, 2b+1]
        W[4*b+3][col] = 1  # C[1, 2b+1]
        
        # P4 = d(g - e)
        col = base + 3
        U[3][col] = 1  # d
        V[10+2*b][col] = 1  # g
        V[2*b][col] = -1  # -e
        W[4*b][col] = 1  # C[0, 2b]
        W[4*b+1][col] = 1  # C[1, 2b]
        
        # P5 = (a + b)h
        col = base + 4
        U[0][col] = 1  # a
        U[1][col] = 1  # b
        V[10+2*b+1][col] = 1  # h
        W[4*b][col] = -1  # C[0, 2b]
        W[4*b+2][col] = 1  # C[0, 2b+1]
        
        # P6 = (c - a)(e + f)
        col = base + 5
        U[2][col] = 1  # c
        U[0][col] = -1  # -a
        V[2*b][col] = 1  # e
        V[2*b+1][col] = 1  # f
        W[4*b+3][col] = 1  # C[1, 2b+1]
        
        # P7 = (b - d)(g + h)
        col = base + 6
        U[1][col] = 1  # b
        U[3][col] = -1  # -d
        V[10+2*b][col] = 1  # g
        V[10+2*b+1][col] = 1  # h
        W[4*b][col] = 1  # C[0, 2b]
    
    # Verification step (optional, but good for robustness)
    # if time.time() < deadline:
    #     pass # Verification code would go here

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
