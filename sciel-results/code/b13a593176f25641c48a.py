"""Strassen-based decomposition for 6x6x6 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140

    n, m, p = DIMENSIONS
    block = 3

    # Strassen's 7 terms for 2x2 blocks of 3x3 matrices
    # Each term: (A_components, B_components, C_coefficients)
    # A_components: list of (block_row, block_col, coeff)
    # B_components: list of (block_row, block_col, coeff)
    # C_coefficients: dict from (out_block_row, out_block_col) to coeff
    
    strassen_terms = [
        # M1 = (A11+A22)*(B11+B22)
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)],
         {(0,0): 1, (1,1): 1}),
        # M2 = (A21+A22)*B11
        ([(1,0,1), (1,1,1)], [(0,0,1)],
         {(1,0): 1, (1,1): -1}),
        # M3 = A11*(B12-B22)
        ([(0,0,1)], [(0,1,1), (1,1,-1)],
         {(0,0): 1, (0,1): 1}),
        # M4 = A22*(B21-B11)
        ([(1,1,1)], [(1,0,1), (0,0,-1)],
         {(0,0): 1, (1,0): 1}),
        # M5 = (A11+A12)*B22
        ([(0,0,1), (0,1,1)], [(1,1,1)],
         {(0,0): -1, (0,1): 1}),
        # M6 = (A21-A11)*(B11+B12)
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)],
         {(1,1): 1}),
        # M7 = (A12-A22)*(B21+B22)
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)],
         {(0,0): 1, (1,1): 1}),
    ]
    
    rank = 7 * block * block  # 63
    
    U = [[0] * rank for _ in range(n * m)]  # 36 x 63
    V = [[0] * rank for _ in range(m * p)]  # 36 x 63
    W = [[0] * rank for _ in range(p * n)]  # 36 x 63
    
    col = 0
    for A_comps, B_comps, C_coeffs in strassen_terms:
        for i in range(block):
            for k in range(block):
                for j in range(block):
                    # U: A-side, index = (a_br*block+i)*m + (a_bc*block+k)
                    for a_br, a_bc, a_coeff in A_comps:
                        u_idx = (a_br * block + i) * m + (a_bc * block + k)
                        U[u_idx][col] = a_coeff
                    
                    # V: B-side, index = (b_br*block+k)*p + (b_bc*block+j)
                    for b_br, b_bc, b_coeff in B_comps:
                        v_idx = (b_br * block + k) * p + (b_bc * block + j)
                        V[v_idx][col] = b_coeff
                    
                    # W: C-side, index = (c_bc*block+j)*n + (c_br*block+i)
                    for (c_br, c_bc), c_coeff in C_coeffs.items():
                        w_idx = (c_bc * block + j) * n + (c_br * block + i)
                        W[w_idx][col] = c_coeff
                    
                    col += 1
    
    assert col == rank, f"Expected {rank}, got {col}"
    
    # Quick verification
    if time.time() < deadline:
        rng = np.random.RandomState(42)
        A = rng.randn(6, 6)
        B = rng.randn(6, 6)
        C_expected = A @ B
        
        C_check = np.zeros((6, 6))
        for i in range(6):
            for j in range(6):
                total = 0.0
                for a in range(6):
                    u_idx = i * m + a
                    v_idx = a * p + j
                    w_idx = j * n + i
                    for r in range(rank):
                        total += U[u_idx][r] * V[v_idx][r] * W[w_idx][r]
                C_check[i, j] = total
        
        err = np.max(np.abs(C_check - C_expected))
        if err > 1e-10:
            # Fallback to schoolbook
            rank = n * m * p
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            column = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i * m + j][column] = 1
                        V[j * p + k][column] = 1
                        W[k * n + i][column] = 1
                        column += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
