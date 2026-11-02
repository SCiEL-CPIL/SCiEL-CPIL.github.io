# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    n, m, p = 2, 3, 9
    
    # Schoolbook fallback
    r_school = n * m * p
    U = [[0] * r_school for _ in range(n * m)]
    V = [[0] * r_school for _ in range(m * p)]
    W = [[0] * r_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    def verify_block(U_b, V_b, W_b, r_b, n_b, m_b, p_b):
        """
        Verify if U_b, V_b, W_b of rank r_b decompose the n_b x m_b x p_b matmul tensor.
        U_b: (n_b*m_b, r_b)
        V_b: (m_b*p_b, r_b)
        W_b: (p_b*n_b, r_b)
        """
        # Check all (a, b, c)
        for a in range(n_b * m_b):
            i = a // m_b
            j = a % m_b
            for b in range(m_b * p_b):
                j2 = b // p_b
                k = b % p_b
                for c in range(p_b * n_b):
                    k2 = c // n_b
                    i2 = c % n_b
                    # Target: 1 if i==i2, j==j2, k==k2 else 0
                    target = 1 if (i == i2 and j == j2 and k == k2) else 0
                    s = 0
                    for t in range(r_b):
                        s += U_b[a][t] * V_b[b][t] * W_b[c][t]
                    if s != target:
                        return False
        return True

    # Strategy: Find rank for 2x3x3, then replicate 3 times for 2x3x9
    # Rank(2x3x9) = 3 * Rank(2x3x3)
    # Schoolbook Rank(2x3x3) = 18 -> Total 54
    # Target Rank(2x3x3) = 16 -> Total 48
    
    best_block_rank = 18
    best_block_U, best_block_V, best_block_W = None, None, None
    
    # Initialize schoolbook for 2x3x3
    U_b = [[0] * 18 for _ in range(6)]
    V_b = [[0] * 18 for _ in range(9)]
    W_b = [[0] * 18 for _ in range(6)]
    col = 0
    for i in range(2):
        for j in range(3):
            for k in range(3):
                U_b[i * 3 + j][col] = 1
                V_b[j * 3 + k][col] = 1
                W_b[k * 2 + i][col] = 1
                col += 1
    best_block_U, best_block_V, best_block_W = U_b, V_b, W_b

    random.seed(42)
    
    # Search for lower rank in 2x3x3
    # Try rank 16, 17
    for r_try in [16, 17]:
        if time.time() > deadline - 10:
            break
        found = False
        for attempt in range(5000):
            if time.time() > deadline - 10:
                break
            # Random integers
            U_r = [[random.randint(-2, 2) for _ in range(r_try)] for _ in range(6)]
            V_r = [[random.randint(-2, 2) for _ in range(r_try)] for _ in range(9)]
            W_r = [[random.randint(-2, 2) for _ in range(r_try)] for _ in range(6)]
            
            if verify_block(U_r, V_r, W_r, r_try, 2, 3, 3):
                best_block_rank = r_try
                best_block_U, best_block_V, best_block_W = U_r, V_r, W_r
                found = True
                break
        if found:
            break
            
    # Construct final 2x3x9 decomposition
    r_final = 3 * best_block_rank
    U_final = [[0] * r_final for _ in range(6)]
    V_final = [[0] * r_final for _ in range(27)]
    W_final = [[0] * r_final for _ in range(18)]
    
    # Replicate block 3 times
    # The 9 columns of B are split into 3 groups of 3
    # B = [B_0, B_1, B_2]
    # C = [C_0, C_1, C_2]
    # The tensor is the direct sum.
    for block_idx in range(3):
        offset = block_idx * best_block_rank
        for a in range(6):
            for t in range(best_block_rank):
                U_final[a][offset + t] = best_block_U[a][t]
        
        for b in range(9):
            # b indexes (j, k) in the 3x3 block
            # In the full 3x9, b_full = j * 9 + (block_idx * 3 + k)
            # So for a given block, the local b is 0..8
            # We need to map local b to global b
            j_local = b // 3
            k_local = b % 3
            b_global = j_local * 9 + (block_idx * 3 + k_local)
            for t in range(best_block_rank):
                V_final[b_global][offset + t] = best_block_V[b][t]
                
        for c in range(6):
            # c indexes (k, i) in the 2x3 block
            # In the full 9x2, c_full = k * 2 + i
            # k_global = block_idx * 3 + k_local
            k_local = c // 2
            i_local = c % 2
            c_global = (block_idx * 3 + k_local) * 2 + i_local
            for t in range(best_block_rank):
                W_final[c_global][offset + t] = best_block_W[c][t]

    return {"U": U_final, "V": V_final, "W": W_final}
# EVOLVE-BLOCK-END
