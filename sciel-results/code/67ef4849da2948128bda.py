import time

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5 second safety margin
    
    n, m, p = DIMENSIONS
    
    # Strassen-based decomposition: partition p=12 columns into 6 pairs
    # Each pair uses 7 rank-1 terms (Strassen for 2x2x2 matrix mult)
    # Total rank = 6 * 7 = 42
    
    num_blocks = p // 2  # 6 blocks
    rank = num_blocks * 7  # 42
    
    U = [[0] * rank for _ in range(n * m)]  # shape (4, 42)
    V = [[0] * rank for _ in range(m * p)]  # shape (24, 42)
    W = [[0] * rank for _ in range(p * n)]  # shape (24, 42)
    
    # Strassen 7-product decomposition for 2x2x2
    for t in range(num_blocks):
        base = t * 7
        for s in range(7):
            c = base + s
            if s == 0:
                # M1 = (A00 + A11) * (B00 + B11)
                U[0][c] = 1
                U[3][c] = 1
                V[0*p + 2*t][c] = 1
                V[1*p + 2*t+1][c] = 1
                W[2*t*n + 0][c] = 1
                W[(2*t+1)*n + 1][c] = 1
            elif s == 1:
                # M2 = (A10 + A11) * B00
                U[2][c] = 1
                U[3][c] = 1
                V[0*p + 2*t][c] = 1
                W[2*t*n + 1][c] = 1
                W[(2*t+1)*n + 1][c] = -1
            elif s == 2:
                # M3 = A00 * (B01 - B11)
                U[0][c] = 1
                V[0*p + 2*t+1][c] = 1
                V[1*p + 2*t+1][c] = -1
                W[(2*t+1)*n + 0][c] = 1
                W[(2*t+1)*n + 1][c] = 1
            elif s == 3:
                # M4 = A11 * (B10 - B00)
                U[3][c] = 1
                V[1*p + 2*t][c] = 1
                V[0*p + 2*t][c] = -1
                W[2*t*n + 0][c] = 1
                W[2*t*n + 1][c] = 1
            elif s == 4:
                # M5 = (A00 + A01) * B11
                U[0][c] = 1
                U[1][c] = 1
                V[1*p + 2*t+1][c] = 1
                W[2*t*n + 0][c] = -1
                W[(2*t+1)*n + 0][c] = 1
            elif s == 5:
                # M6 = (A10 - A00) * (B00 + B01)
                U[2][c] = 1
                U[0][c] = -1
                V[0*p + 2*t][c] = 1
                V[0*p + 2*t+1][c] = 1
                W[(2*t+1)*n + 1][c] = 1
            elif s == 6:
                # M7 = (A01 - A11) * (B10 + B11)
                U[1][c] = 1
                U[3][c] = -1
                V[1*p + 2*t][c] = 1
                V[1*p + 2*t+1][c] = 1
                W[2*t*n + 0][c] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
