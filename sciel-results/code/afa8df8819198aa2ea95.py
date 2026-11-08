import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin
    
    n, m, p = DIMENSIONS
    
    # Use Laderman's algorithm for 3x3 matrix multiplication (rank 23)
    # and tile it 2x2 times for the 6x6 matrix.
    # Total rank = 2 * 2 * 2 * 23 = 184.
    
    block_size = 3
    blocks_per_dim = 2  # 6 / 3
    
    rank = (blocks_per_dim ** 3) * 23
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Laderman's algorithm coefficients for 3x3 matrices
    # Each entry is (u_coeffs, v_coeffs, w_coeffs)
    # u_coeffs: dict {(i, j): val} for A
    # v_coeffs: dict {(j, k): val} for B
    # w_coeffs: dict {(i, k): val} for C
    # The decomposition is:
    # C_{ik} = sum_r W_{k*n+i, r} * (sum_j U_{i*m+j, r} A_{ij}) * (sum_j V_{j*p+k, r} B_{jk})
    
    laderman_patterns = [
        # M1
        ({(0,0): 1, (2,2): 1}, {(0,0): 1, (2,2): 1}, {(0,0): 1, (2,2): 1}),
        # M2
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): -1, (1,1): 1, (2,2): 1}),
        # M3
        ({(0,0): 1, (2,2): 1}, {(1,1): 1, (2,2): 1}, {(1,1): 1, (2,2): -1}),
        # M4
        ({(0,1): 1, (1,2): 1}, {(1,1): 1}, {(0,1): 1, (1,2): 1}),
        # M5
        ({(1,0): 1, (2,1): 1}, {(1,1): 1}, {(1,0): 1, (2,1): 1}),
        # M6
        ({(1,1): 1, (2,2): 1}, {(0,0): 1, (1,1): 1}, {(1,1): -1, (2,2): 1}),
        # M7
        ({(0,0): 1, (1,1): 1}, {(1,1): 1, (2,2): 1}, {(0,0): 1, (1,1): -1}),
        # M8
        ({(0,1): 1, (1,2): 1}, {(1,1): 1, (2,2): 1}, {(0,1): -1, (1,2): 1}),
        # M9
        ({(1,0): 1, (2,1): 1}, {(0,0): 1, (1,1): 1}, {(1,0): 1, (2,1): -1}),
        # M10
        ({(0,0): 1, (2,2): 1}, {(0,1): 1, (1,2): 1}, {(0,1): 1, (1,2): 1}),
        # M11
        ({(0,0): 1, (2,2): 1}, {(1,0): 1, (2,1): 1}, {(1,0): 1, (2,1): 1}),
        # M12
        ({(1,1): 1, (2,2): 1}, {(0,1): 1, (1,2): 1}, {(0,1): -1, (1,2): 1}),
        # M13
        ({(1,1): 1, (2,2): 1}, {(1,0): 1, (2,1): 1}, {(1,0): 1, (2,1): -1}),
        # M14
        ({(0,1): 1, (1,2): 1}, {(0,0): 1, (2,2): 1}, {(0,1): 1, (1,2): 1}),
        # M15
        ({(1,0): 1, (2,1): 1}, {(0,0): 1, (2,2): 1}, {(1,0): 1, (2,1): 1}),
        # M16
        ({(0,0): 1, (1,1): 1}, {(0,1): 1, (1,2): 1}, {(0,1): -1, (1,2): 1}),
        # M17
        ({(0,0): 1, (1,1): 1}, {(1,0): 1, (2,1): 1}, {(1,0): 1, (2,1): -1}),
        # M18
        ({(0,1): 1, (1,2): 1}, {(1,0): 1, (2,1): 1}, {(1,0): -1, (2,1): 1}),
        # M19
        ({(1,0): 1, (2,1): 1}, {(0,1): 1, (1,2): 1}, {(0,1): 1, (1,2): -1}),
        # M20
        ({(0,0): 1, (2,2): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # M21
        ({(0,0): 1, (2,2): 1}, {(0,0): 1, (2,2): 1}, {(0,0): 1, (2,2): 1}),
        # M22
        ({(1,1): 1}, {(1,1): 1}, {(1,1): 1}),
        # M23
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (2,2): 1}, {(0,0): 1, (2,2): 1}),
    ]
    
    col_idx = 0
    for I in range(blocks_per_dim):
        for J in range(blocks_per_dim):
            for K in range(blocks_per_dim):
                for u_coeffs, v_coeffs, w_coeffs in laderman_patterns:
                    # Update U
                    for (pi, pj), val in u_coeffs.items():
                        r = I * block_size + pi
                        c = J * block_size + pj
                        U[r * m + c][col_idx] += val
                    
                    # Update V
                    for (pi, pj), val in v_coeffs.items():
                        r = J * block_size + pi
                        c = K * block_size + pj
                        V[r * p + c][col_idx] += val
                    
                    # Update W
                    for (pi, pj), val in w_coeffs.items():
                        # W index is k*n + i
                        # k is col of C (which is K block, pj local)
                        # i is row of C (which is I block, pi local)
                        k = K * block_size + pj
                        i = I * block_size + pi
                        W[k * n + i][col_idx] += val
                    
                    col_idx += 1
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
