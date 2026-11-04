"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen terms for 2x2 matrix multiplication
    # Format: (A_indices, A_coeffs, B_indices, B_coeffs, C_indices, C_coeffs)
    # Local indices (row, col) for 2x2 matrices
    terms = [
        # M1 = (a00+a11)(b00+b11)
        ([(0,0), (1,1)], [1, 1], [(0,0), (1,1)], [1, 1], [(0,0), (1,1)], [1, 1]),
        # M2 = (a10+a11)b00
        ([(1,0), (1,1)], [1, 1], [(0,0)], [1], [(1,0), (1,1)], [1, -1]),
        # M3 = a00(b01-b11)
        ([(0,0)], [1], [(0,1), (1,1)], [1, -1], [(0,1), (1,1)], [1, 1]),
        # M4 = a11(b10-b00)
        ([(1,1)], [1], [(1,0), (0,0)], [1, -1], [(0,0), (1,0)], [1, 1]),
        # M5 = (a00+a01)b11
        ([(0,0), (0,1)], [1, 1], [(1,1)], [1], [(0,0), (0,1)], [-1, 1]),
        # M6 = (a10-a00)(b00+b01)
        ([(1,0), (0,0)], [1, -1], [(0,0), (0,1)], [1, 1], [(1,1)], [1]),
        # M7 = (a01-a11)(b10+b11)
        ([(0,1), (1,1)], [1, -1], [(1,0), (1,1)], [1, 1], [(0,0)], [1]),
    ]
    
    col = 0
    for q in range(4):
        # Left part of A (cols 0,1) times Top part of B (rows 0,1, cols 2q, 2q+1)
        for t in terms:
            a_idx, a_val, b_idx, b_val, c_idx, c_val = t
            for (li, lj), val in zip(a_idx, a_val):
                u = li * m + lj
                U[u][col] += val
            for (lj, lk), val in zip(b_idx, b_val):
                v = lj * p + (2 * q + lk)
                V[v][col] += val
            for (li, lk), val in zip(c_idx, c_val):
                w = (2 * q + lk) * n + li
                W[w][col] += val
            col += 1
            
        # Right part of A (cols 2,3) times Bottom part of B (rows 2,3, cols 2q, 2q+1)
        for t in terms:
            a_idx, a_val, b_idx, b_val, c_idx, c_val = t
            for (li, lj), val in zip(a_idx, a_val):
                u = li * m + (2 + lj)
                U[u][col] += val
            for (lj, lk), val in zip(b_idx, b_val):
                v = (2 + lj) * p + (2 * q + lk)
                V[v][col] += val
            for (li, lk), val in zip(c_idx, c_val):
                w = (2 * q + lk) * n + li
                W[w][col] += val
            col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
