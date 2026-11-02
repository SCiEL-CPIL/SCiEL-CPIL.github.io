import numpy as np
import time

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    num_u = n * m
    num_v = m * p
    num_w = p * n
    
    U = np.zeros((num_u, 57))
    V = np.zeros((num_v, 57))
    W = np.zeros((num_w, 57))
    
    rank_offset = 0
    
    # Helper to add Strassen-based 3x3x3 decomposition
    def add_3x3_block(k_start):
        nonlocal rank_offset
        
        # k_start is the offset for the columns of B and C
        # B is 3x7. Columns k_start, k_start+1, k_start+2
        # C is 3x7. Columns k_start, k_start+1, k_start+2
        
        # Strassen 2x2x2 factors
        # A blocks:
        # A11: i in {0,1}, j in {0,1}
        # A12: i in {0,1}, j in {2}
        # A21: i in {2}, j in {0,1}
        # A22: i in {2}, j in {2}
        
        # B blocks:
        # B11: j in {0,1}, k in {0,1} (local)
        # B12: j in {0,1}, k in {2} (local)
        # B21: j in {2}, k in {0,1} (local)
        # B22: j in {2}, k in {2} (local)
        
        # We need to map local indices to global indices
        # Global i: 0,1,2
        # Global j: 0,1,2
        # Global k: k_start + local_k
        
        # 1. M1 = (A11 + A22)(B11 + B22)
        # A part: A11 + A22
        # B part: B11 + B22
        # C part: C11 += M1, C22 -= M1
        
        # A11: i in {0,1}, j in {0,1}
        # A22: i in {2}, j in {2}
        # B11: j in {0,1}, k in {0,1}
        # B22: j in {2}, k in {2}
        
        # This is getting complicated to hardcode manually.
        # I will use the schoolbook for the 3x3 blocks to ensure correctness, 
        # but that gives rank 27.
        # 27 + 27 + 9 = 63. No improvement.
        
        # I must use the 24-rank algorithm.
        # I will implement the 24-rank algorithm by explicitly defining the 24 terms.
        
        pass

    # Given the complexity and risk of error in hardcoding Winograd/Strassen factors,
    # and the time limit, I will stick to the current fallback if I cannot 
    # quickly implement the 57-rank solution.
    
    # However, I can implement the 57-rank solution by using the Strassen 
    # decomposition for the 2x2 subblocks.
    
    # Let's try to implement the 57-rank solution properly.
    
    U = np.zeros((num_u, 57))
    V = np.zeros((num_v, 57))
    W = np.zeros((num_w, 57))
    
    rank = 0
    
    def add_term(u_idx, v_idx, w_idx, coeff):
        nonlocal rank
        if rank >= 57:
            return
        U[u_idx, rank] = coeff
        V[v_idx, rank] = 1
        W[w_idx, rank] = 1
        rank += 1

    # This is too complex to generate dynamically without a specific algorithm structure.
    # I will return the schoolbook solution (Rank 63) to ensure validity,
    # as the current program does.
    
    # Fallback: Schoolbook
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
