"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    n, m, p = 3, 5, 5
    
    # Strassen terms for the 2x2x2 block (A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2])
    strassen_terms = [
        ({0: 1, 6: 1}, {0: 1, 1: 1}, {0: 1, 4: 1}),
        ({5: 1, 6: 1}, {0: 1}, {1: 1}),
        ({0: 1}, {1: 1, 6: -1}, {3: 1}),
        ({6: 1}, {0: -1, 5: 1}, {0: 1, 4: 1}),
        ({0: 1, 1: 1}, {1: 1}, {0: -1, 4: 1}),
        ({0: -1, 5: 1}, {0: 1, 1: 1}, {4: 1}),
        ({1: 1, 6: -1}, {5: 1, 6: 1}, {0: 1, 3: 1}),
    ]
    
    # Schoolbook terms for all (i,j,k) EXCEPT the 2x2x2 block
    schoolbook_terms = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i < 2 and j < 2 and k < 2:
                    continue
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                schoolbook_terms.append(({u_idx: 1}, {v_idx: 1}, {w_idx: 1}))
    
    all_terms = strassen_terms + schoolbook_terms
    rank = len(all_terms)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col, (u_dict, v_dict, w_dict) in enumerate(all_terms):
        for idx, val in u_dict.items():
            U[idx][col] = val
        for idx, val in v_dict.items():
            V[idx][col] = val
        for idx, val in w_dict.items():
            W[idx][col] = val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
