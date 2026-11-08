import numpy as np
import time

DIMENSIONS = [4, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    n, m, p = DIMENSIONS
    # Rank 85: 49 (4x4x4 Strassen) + 16 (4x4x1) + 16 (4x1x4) + 4 (4x1x1)
    rank = 85
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    def get_strassen_2x2_factors():
        u = np.zeros((4, 7))
        v = np.zeros((4, 7))
        w = np.zeros((4, 7))
        
        # Standard Strassen 7 terms
        # M1 = (A11 + A22)(B11 + B22)
        u[0, 0] = 1; u[3, 0] = 1
        v[0, 0] = 1; v[3, 0] = 1
        w[0, 0] = 1; w[3, 0] = 1
        
        # M2 = (A21 + A22)B11
        u[2, 1] = 1; u[3, 1] = 1
        v[0, 1] = 1
        w[2, 1] = 1; w[3, 1] = -1
        
        # M3 = A11(B21 - B22)
        u[0, 2] = 1
        v[1, 2] = 1; v[3, 2] = -1
        w[1, 2] = 1; w[3, 2] = 1
        
        # M4 = A22(B12 - B11)
        u[3, 3] = 1
        v[2, 3] = 1; v[0, 3] = -1
        w[0, 3] = 1; w[2, 3] = 1
        
        # M5 = (A11 + A12)B22
        u[0, 4] = 1; u[1, 4] = 1
        v[3, 4] = 1
        w[0, 4] = -1; w[1, 4] = 1
        
        # M6 = (A21 - A11)(B11 + B12)
        u[0, 5] = -1; u[2, 5] = 1
        v[0, 5] = 1; v[1, 5] = 1
        w[3, 5] = 1
        
        # M7 = (A12 - A22)(B21 + B22)
        u[1, 6] = 1; u[3, 6] = -1
        v[2, 6] = 1; v[3, 6] = 1
        w[0, 6] = 1
        
        return u, v, w

    s_u, s_v, s_w = get_strassen_2x2_factors()
    
    col = 0
    # 4x4x4 part: A[0:4, 0:4] * B[0:4, 0:4] -> C[0:4, 0:4]
    # This uses 7x7=49 terms
    # We apply Strassen to each of the 4 2x2 block multiplications?
    # No, the code structure applies the 7 Strassen terms to the 4 2x2 blocks of A and B?
    # Actually, the loop `for t_idx in range(7)` and `for inner_k in range(7)`
    # suggests we are doing 7 Strassen steps, each involving 7 multiplications of 2x2 blocks?
    # No, that would be 49 multiplications of 2x2 blocks.
    # Each 2x2 block multiplication has rank 7.
    # So total rank 49.
    
    # The mapping in the code:
    # a_c, b_c, c_c define which 2x2 blocks are involved.
    # For t_idx=0, it's the first Strassen term for the top-left 2x2 block?
    # No, the code iterates t_idx (Strassen term) and inner_k (which 2x2 block? No, inner_k is the Strassen factor index).
    # Wait, `u_vec = s_u[:, inner_k]`.
    # This means for each `t_idx` (0..6), we use the `t_idx`-th Strassen combination of blocks?
    # And `inner_k` iterates over the 7 factors of the 2x2 Strassen algorithm?
    # This implies we are computing 7 independent 2x2 Strassen multiplications?
    # No, that would be rank 7 * 7 = 49.
    # And each 2x2 Strassen multiplication computes one 2x2 block product.
    # So we compute 7 block products?
    # But we only need 4 block products (TL, TR, BL, BR).
    # Strassen computes all 4 with 7 multiplications.
    # So we should only use 7 terms total for the 4x4x4 part?
    # Why does the code use 49?
    
    # Let's re-examine the code's 4x4x4 logic.
    # `terms` has 7 entries.
    # Each entry has `a_c`, `b_c`, `c_c`.
    # `a_c` is a dict mapping (block_row, block_col) to coeff.
    # For t_idx=0: a_c = {(0,0):1, (1,1):1}.
    # This corresponds to A11 + A22.
    # So `terms` are the 7 Strassen terms for the 4x4 matrix.
    # Inside the loop, `inner_k` iterates 0..6.
    # `u_vec = s_u[:, inner_k]`.
    # This applies the 2x2 Strassen factor `inner_k` to the blocks specified in `a_c`.
    # This effectively expands the 4x4 Strassen terms into 2x2 Strassen terms.
    # Since each 4x4 Strassen term is a sum of 2x2 blocks, and each 2x2 block product
    # is computed with 7 multiplications, the total number of scalar multiplications
    # is 7 (4x4 terms) * 7 (2x2 factors) = 49.
    
    # This is correct.
    
    terms = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]
    
    for t_idx, (a_c, b_c, c_c) in enumerate(terms):
        for inner_k in range(7):
            col_idx = t_idx * 7 + inner_k
            u_vec = s_u[:, inner_k]
            v_vec = s_v[:, inner_k]
            w_vec = s_w[:, inner_k]
            
            for (a_br, a_bc), coeff in a_c.items():
                for lr in range(2):
                    for lc in range(2):
                        li = lr * 2 + lc
                        if u_vec[li] != 0:
                            gi = 2 * a_br + lr
                            gj = 2 * a_bc + lc
                            U[gi * 5 + gj, col_idx] += coeff * u_vec[li]
            
            for (b_br, b_bc), coeff in b_c.items():
                for lr in range(2):
                    for lc in range(2):
                        li = lr * 2 + lc
                        if v_vec[li] != 0:
                            gj = 2 * b_br + lr
                            gk = 2 * b_bc + lc
                            V[gj * 5 + gk, col_idx] += coeff * v_vec[li]
            
            for (c_br, c_bc), coeff in c_c.items():
                for lr in range(2):
                    for lc in range(2):
                        li = lr * 2 + lc
                        if w_vec[li] != 0:
                            gk = 2 * c_bc + lc
                            gi = 2 * c_br + lr
                            W[gk * 4 + gi, col_idx] += coeff * w_vec[li]
        
    col = 49
    
    # 4x4x1: A[0:4, 0:4] * B[0:4, 4] -> C[0:4, 4]
    # C is 4x5. Col 4 is indices 16..19 (column-major with stride 4).
    # W index: k*4 + i. k=4 (col 4), i=0..3 (rows).
    for i in range(4):
        for j in range(4):
            U[i * 5 + j, col] = 1
            V[j * 5 + 4, col] = 1
            W[4 * 4 + i, col] = 1
            col += 1
            
    # 4x1x4: A[0:4, 4] * B[4, 0:4] -> C[0:4, 0:4]
    # C cols 0..3.
    for i in range(4):
        for k in range(4):
            U[i * 5 + 4, col] = 1
            V[4 * 5 + k, col] = 1
            W[k * 4 + i, col] = 1
            col += 1
            
    # 4x1x1: A[0:4, 4] * B[4, 4] -> C[0:4, 4]
    for i in range(4):
        U[i * 5 + 4, col] = 1
        V[4 * 5 + 4, col] = 1
        W[4 * 4 + i, col] = 1
        col += 1
        
    if time.time() > deadline:
        # Should not happen, but safety
        pass
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
