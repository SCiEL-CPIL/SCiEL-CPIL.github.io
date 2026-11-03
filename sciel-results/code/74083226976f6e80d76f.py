"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-minimal exact tensor decomposition for 2x3 * 3x8 matrix multiplication.
    
    Strategy:
    1. Split m=3 into m1=1 and m2=2.
    2. Part 1: A[:, 0] (2x1) * B[0, :] (1x8) -> 2x8. Rank 16.
    3. Part 2: A[:, 1:3] (2x2) * B[1:3, :] (2x8) -> 2x8.
       Split p=8 into four blocks of 2.
       Each 2x2x2 uses Strassen (Rank 7).
       Total Rank for Part 2: 4 * 7 = 28.
    4. Total Rank: 16 + 28 = 44.
    """
    start_time = time.time()
    
    n, m, p = DIMENSIONS  # 2, 3, 8
    
    rank = 44
    U = [[0] * rank for _ in range(n * m)]  # 6 x 44
    V = [[0] * rank for _ in range(m * p)]  # 24 x 44
    W = [[0] * rank for _ in range(p * n)]  # 16 x 44

    col = 0

    # Helper to add a 2x2x2 Strassen block
    def add_strassen_2x2x2(a00, a01, a10, a11, b00, b01, b10, b11, c00, c01, c10, c11):
        nonlocal col
        
        # M1 = (A00 + A11) * (B00 + B11)
        U[a00][col] += 1; U[a11][col] += 1
        V[b00][col] += 1; V[b11][col] += 1
        W[c00][col] += 1; W[c11][col] += 1
        col += 1

        # M2 = (A10 + A11) * B00
        U[a10][col] += 1; U[a11][col] += 1
        V[b00][col] += 1
        W[c10][col] += 1; W[c11][col] -= 1
        col += 1

        # M3 = A00 * (B01 - B11)
        U[a00][col] += 1
        V[b01][col] += 1; V[b11][col] -= 1
        W[c01][col] += 1; W[c11][col] += 1
        col += 1

        # M4 = A11 * (B10 - B00)
        U[a11][col] += 1
        V[b10][col] += 1; V[b00][col] -= 1
        W[c00][col] += 1; W[c10][col] += 1
        col += 1

        # M5 = (A00 + A01) * B11
        U[a00][col] += 1; U[a01][col] += 1
        V[b11][col] += 1
        W[c00][col] -= 1; W[c01][col] += 1
        col += 1

        # M6 = (A10 - A00) * (B00 + B01)
        U[a10][col] += 1; U[a00][col] -= 1
        V[b00][col] += 1; V[b01][col] += 1
        W[c11][col] += 1
        col += 1

        # M7 = (A01 - A11) * (B10 + B11)
        U[a01][col] += 1; U[a11][col] -= 1
        V[b10][col] += 1; V[b11][col] += 1
        W[c00][col] += 1
        col += 1

    # Part 1: 2x1x8 (j=0)
    # A[:, 0] * B[0, :]
    for i in range(n):
        for k in range(p):
            u_idx = i * m + 0
            v_idx = 0 * p + k
            w_idx = k * n + i
            
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1

    # Part 2: 2x2x8 (j=1,2)
    # A[:, 1:3] * B[1:3, :]
    # A indices:
    a00 = 1 # A[0, 1]
    a01 = 2 # A[0, 2]
    a10 = 4 # A[1, 1]
    a11 = 5 # A[1, 2]

    # Split p=8 into 4 blocks of 2
    for p_block in range(4):
        p_start = p_block * 2
        
        # B indices for this 2-column block
        b00 = 1*p + p_start
        b01 = 1*p + p_start + 1
        b10 = 2*p + p_start
        b11 = 2*p + p_start + 1
        
        # C indices for this 2-column block
        c00 = p_start*n + 0
        c01 = (p_start+1)*n + 0
        c10 = p_start*n + 1
        c11 = (p_start+1)*n + 1

        add_strassen_2x2x2(a00, a01, a10, a11, b00, b01, b10, b11, c00, c01, c10, c11)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
