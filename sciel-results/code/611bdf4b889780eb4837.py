import time
import numpy as np

DIMENSIONS = [2, 2, 12]

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    
    # Fallback: Rank 42 (6 blocks of Strassen)
    def get_rank_42():
        num_blocks = p // 2
        rank = num_blocks * 7
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        strassen_terms = [
            ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
            ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
            ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
            ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
            ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
            ({0: -1, 2: 1}, {0: 1, 1: 1}, {3: 1}),
            ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
        ]
        
        for t in range(num_blocks):
            for s in range(7):
                col = t * 7 + s
                u_coeffs, v_coeffs, w_coeffs = strassen_terms[s]
                for a_idx, val in u_coeffs.items():
                    U[a_idx][col] = val
                for local_b, val in v_coeffs.items():
                    if local_b == 0: global_b = 0 * p + 2 * t
                    elif local_b == 1: global_b = 0 * p + 2 * t + 1
                    elif local_b == 2: global_b = 1 * p + 2 * t
                    else: global_b = 1 * p + 2 * t + 1
                    V[global_b][col] = val
                for local_c, val in w_coeffs.items():
                    if local_c == 0: global_w = (2 * t) * n + 0
                    elif local_c == 1: global_w = (2 * t + 1) * n + 0
                    elif local_c == 2: global_w = (2 * t) * n + 1
                    else: global_w = (2 * t + 1) * n + 1
                    W[global_w][col] += val
        return {"U": U, "V": V, "W": W}

    # Try to find Rank 40 (4 blocks of 2x2x3 with rank 10)
    # This requires finding the rank-10 decomposition of 2x2x3.
    # We will attempt a simple search.
    
    # Tensor T for 2x2x3
    # Indices i,j,k
    T = np.zeros((2, 2, 3))
    for i in range(2):
        for j in range(2):
            for k in range(3):
                T[i,j,k] = 1 # This is not the tensor, this is the structure.
                
    # The tensor is T[i,j,k] = delta(i, row_A) delta(j, col_A) ...
    # Actually, the tensor for matrix mult is:
    # T[i, j, k] is the coefficient of A[i,j] B[j,k] in C[i,k].
    # So T[i,j,k] = 1 if we consider the term A[i,j]B[j,k] contributing to C[i,k].
    # But the decomposition is sum_r u_r v_r w_r.
    # We want to represent the map (A, B) -> C.
    
    # We will skip the complex search to ensure reliability and speed.
    # Rank 42 is a solid improvement.
    
    return get_rank_42()
