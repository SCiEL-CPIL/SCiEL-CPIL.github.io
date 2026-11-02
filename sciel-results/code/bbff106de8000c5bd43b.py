"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Target rank: 4 Strassen blocks (4*7=28) + 43 schoolbook = 71
    rank = 71
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Define the 4 disjoint 2x2x2 blocks
    # Block 1: i in {0,1}, j in {0,1}, k in {0,1}
    # Block 2: i in {0,1}, j in {2,3}, k in {0,1}
    # Block 3: i in {0,1}, j in {0,1}, k in {2,3}
    # Block 4: i in {0,1}, j in {2,3}, k in {2,3}
    
    blocks = [
        ([0, 1], [0, 1], [0, 1]),
        ([0, 1], [2, 3], [0, 1]),
        ([0, 1], [0, 1], [2, 3]),
        ([0, 1], [2, 3], [2, 3]),
    ]
    
    covered = set()
    
    # Strassen terms for a 2x2 block
    # Local indices 0,1
    # U: (i,j), V: (j,k), W: (k,i)
    strassen_terms = [
        # M1
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (0,1):1, (1,0):1, (1,1):1}),
        # M2
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(0,1):1}),
        # M3
        ({(0,0):1}, {(1,0):1, (1,1):-1}, {(1,1):-1}),
        # M4
        ({(1,1):1}, {(0,1):1, (0,0):-1}, {(0,0):-1}),
        # M5
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(1,0):1, (1,1):1}),
        # M6
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(0,1):1, (1,1):-1}),
        # M7
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1, (0,1):1}),
    ]
    
    for bi, bj, bk in blocks:
        for u_dict, v_dict, w_dict in strassen_terms:
            for (li, lj), coeff in u_dict.items():
                gi = bi[li]
                gj = bj[lj]
                U[gi * m + gj][col] = coeff
            for (lj, lk), coeff in v_dict.items():
                gj = bj[lj]
                gk = bk[lk]
                V[gj * p + gk][col] = coeff
            for (lk, li), coeff in w_dict.items():
                gk = bk[lk]
                gi = bi[li]
                W[gk * n + gi][col] = coeff
            
            # Mark covered indices
            for li in range(2):
                for lj in range(2):
                    for lk in range(2):
                        covered.add((bi[li], bj[lj], bk[lk]))
            
            col += 1
    
    # Schoolbook for remaining
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
