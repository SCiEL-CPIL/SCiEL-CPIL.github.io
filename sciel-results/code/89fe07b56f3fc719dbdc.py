"""Matrix multiplication 5x5x5 with reduced rank decomposition."""
DIMENSIONS = [5, 5, 5]

import time
_start_time = time.time()
DEADLINE = 120.0  # safety margin before 150s kill

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 5, 5, 5
    
    # Use a block decomposition:
    # Split indices into 3 + 2.
    # For the 3x3 sub-block use Laderman (rank 7), for 2x2 use Strassen (rank 7).
    # Cross-block terms use schoolbook.
    # Total rank = 7 (3x3) + 7 (2x2) + 6 (3x2) + 6 (2x3) + 6 (3x2 for C21 cross) + 6 (2x3 for C22 cross)
    # But this double counts; let's just use a clean schoolbook-based construction
    # with a known rank-49 method: Kronecker of Strassen(2x2, rank7) with 2x2 identity,
    # giving 4x4 rank 49, plus 1 extra row/col schoolbook.
    
    # Simplest valid improvement: rank 49 via embedding Strassen^2 in 5x5.
    # Actually, let's use the direct construction:
    # C = A*B where we compute via 49 rank-1 terms using Strassen on 4x4 blocks + 1.
    
    # For safety and correctness, use schoolbook rank 125 as fallback,
    # but attempt rank 49 via known construction.
    
    # Known: 5x5 matrix mult can be done with rank 49 using:
    # Pad to 8x8? No. Use 4x4 Strassen (rank 49) for top-left 4x4,
    # and handle row/col 5 with schoolbook: 4 + 4 + 1 = 9 more, total 58.
    # That's better than 125.
    
    # Let's do: top-left 4x4 via Strassen rank 49, plus cross terms.
    # A = [[A4, a5], [b5, c5]] where A4 is 4x4, a5 is 4x1, b5 is 1x4, c5 is 1x1
    # B = [[B4, d5], [e5, f5]]
    # C = [[A4*B4 + a5*e5, A4*d5 + a5*f5], [b5*B4 + c5*e5, b5*d5 + c5*f5]]
    # Terms: A4*B4 (49), a5*e5 (4), A4*d5 (16), a5*f5 (4), b5*B4 (16), c5*e5 (4), b5*d5 (4), c5*f5 (1)
    # Total: 49 + 4 + 16 + 4 + 16 + 4 + 4 + 1 = 98
    # That's rank 98, better than 125.
    
    rank = 98
    U = [[0] * rank for _ in range(n * m)]  # 25 x 98
    V = [[0] * rank for _ in range(m * p)]  # 25 x 98
    W = [[0] * rank for _ in range(p * n)]  # 25 x 98
    
    col = 0
    # Strassen 4x4: use recursive Strassen or explicit 49 terms.
    # For 4x4, use 2x2 Strassen recursively: each 2x2 is rank 7, 4 blocks = 28, plus cross = 49 total for 4x4.
    # Actually 4x4 via Strassen = rank 49 (7^2).
    
    # Let me just use schoolbook for 4x4 (rank 16) to be safe, total = 16 + 4+16+4+16+4+4+1 = 65
    # Even simpler and valid.
    
    # Reuse: top-left 4x4 schoolbook rank 16
    for i in range(4):
        for j in range(4):
            for k in range(4):
                U[i * 5 + j][col] = 1
                V[j * 5 + k][col] = 1
                W[k * 5 + i][col] = 1
                col += 1
    # a5 * e5: A[0:4, 4] * B[4, 0:4] -> C[0:4, 0:4]
    for i in range(4):
        for j in range(4):
            U[i * 5 + 4][col] = 1
            V[4 * 5 + j][col] = 1
            W[j * 5 + i][col] = 1
            col += 1
    # A4 * d5: A[0:4, 0:4] * B[0:4, 4] -> C[0:4, 4]
    for i in range(4):
        for k in range(4):
            U[i * 5 + k][col] = 1
            V[k * 5 + 4][col] = 1
            W[4 * 5 + i][col] = 1
            col += 1
    # a5 * f5: A[0:4, 4] * B[4, 4] -> C[0:4, 4]
    for i in range(4):
        U[i * 5 + 4][col] = 1
        V[4 * 5 + 4][col] = 1
        W[4 * 5 + i][col] = 1
        col += 1
    # b5 * B4: A[4, 0:4] * B[0:4, 0:4] -> C[4, 0:4]
    for j in range(4):
        for k in range(4):
            U[4 * 5 + j][col] = 1
            V[j * 5 + k][col] = 1
            W[k * 5 + 4][col] = 1
            col += 1
    # c5 * e5: A[4,4] * B[4, 0:4] -> C[4, 0:4]
    for j in range(4):
        U[4 * 5 + 4][col] = 1
        V[4 * 5 + j][col] = 1
        W[j * 5 + 4][col] = 1
        col += 1
    # b5 * d5: A[4, 0:4] * B[0:4, 4] -> C[4, 4]
    for j in range(4):
        U[4 * 5 + j][col] = 1
        V[j * 5 + 4][col] = 1
        W[4 * 5 + 4][col] = 1
        col += 1
    # c5 * f5: A[4,4] * B[4,4] -> C[4,4]
    U[4 * 5 + 4][col] = 1
    V[4 * 5 + 4][col] = 1
    W[4 * 5 + 4][col] = 1
    col += 1
    
    assert col == rank, f"col={col}, rank={rank}"
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
