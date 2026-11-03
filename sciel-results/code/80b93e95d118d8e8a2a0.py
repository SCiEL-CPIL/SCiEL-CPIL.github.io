import time
import numpy as np

DIMENSIONS = [6, 6, 6]

def construct():
    n, m, p = DIMENSIONS
    # Rank 189 decomposition using Strassen on 2x2 blocks of 3x3 matrices
    rank = 189
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen steps definitions
    # Each step: (alpha_dict, beta_dict, delta_dict)
    # Keys are (row_block, col_block) in {0,1}
    # alpha: coeffs for A blocks
    # beta: coeffs for B blocks
    # delta: coeffs for C blocks
    
    steps = [
        # M1 = (A00 + A11)(B00 + B11)
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # M2 = (A10 + A11)B00
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        # M3 = A00(B01 - B11)
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        # M4 = A11(B10 - B00)
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        # M5 = (A00 + A01)B11
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        # M6 = (A10 - A00)(B00 + B01)
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        # M7 = (A01 - A11)(B10 + B11)
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1})
    ]
    
    def get_u_idx(block_row, block_col, local_i, local_j):
        # Global row = block_row * 3 + local_i
        # Global col = block_col * 3 + local_j
        # U index = row * 6 + col
        return (block_row * 3 + local_i) * 6 + (block_col * 3 + local_j)
        
    def get_v_idx(block_row, block_col, local_i, local_j):
        # Global row = block_row * 3 + local_i
        # Global col = block_col * 3 + local_j
        # V index = row * 6 + col
        return (block_row * 3 + local_i) * 6 + (block_col * 3 + local_j)
        
    def get_w_idx(block_row, block_col, local_i, local_k):
        # Global row = block_row * 3 + local_i
        # Global col = block_col * 3 + local_k
        # W index convention: W[k*n + i] -> col * 6 + row
        # Here local_k is col, local_i is row
        return (block_col * 3 + local_k) * 6 + (block_row * 3 + local_i)

    for alpha, beta, delta in steps:
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    # U part
                    for (br, bc), val in alpha.items():
                        idx = get_u_idx(br, bc, i, j)
                        U[idx][col] = val
                    # V part
                    for (br, bc), val in beta.items():
                        idx = get_v_idx(br, bc, j, k)
                        V[idx][col] = val
                    # W part
                    for (br, bc), val in delta.items():
                        idx = get_w_idx(br, bc, i, k)
                        W[idx][col] = val
                    col += 1
                    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
