import time
import sys

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-22 exact tensor decomposition for multiplying
    a 2 by 3 matrix by a 3 by 4 matrix using Strassen's algorithm
    for the 2x2x2 blocks and direct multiplication for the 2x1x4 part.
    """
    n, m, p = 2, 3, 4
    rank = 22
    
    # Initialize U, V, W
    # U has shape (n*m, rank) -> (6, 22)
    # V has shape (m*p, rank) -> (12, 22)
    # W has shape (p*n, rank) -> (8, 22)
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: Outer product A[0:2, 2] * B[2, 0:4]
    # A indices: i in {0,1}, j=2. Index in U: i*3 + 2
    # B indices: j=2, k in {0,1,2,3}. Index in V: 2*4 + k
    # C indices: k in {0,1,2,3}, i in {0,1}. Index in W: k*2 + i
    
    for i in range(n):
        for k in range(p):
            u_idx = i * m + 2
            v_idx = 2 * p + k
            w_idx = k * n + i
            
            U[u_idx][col] = 1.0
            V[v_idx][col] = 1.0
            W[w_idx][col] = 1.0
            col += 1
            
    # Part 2: Strassen for A[0:2, 0:2] * B[0:2, 0:2]
    # A indices: i in {0,1}, j in {0,1}. Index in U: i*3 + j
    # B indices: j in {0,1}, k in {0,1}. Index in V: j*4 + k
    # C indices: k in {0,1}, i in {0,1}. Index in W: k*2 + i
    
    # Strassen terms definitions
    # Each term is a list of (U_coeff, V_coeff, W_coeff)
    # U_coeff is dict {(i,j): val}
    # V_coeff is dict {(j,k): val}
    # W_coeff is dict {(k,i): val}
    
    strassen_terms = [
        {
            "U": {(0,0): 1, (1,1): 1},
            "V": {(0,0): 1, (1,1): 1},
            "W": {(0,0): 1, (1,0): 1, (0,1): 1, (1,1): 1}
        },
        {
            "U": {(1,0): 1, (1,1): 1},
            "V": {(0,0): 1},
            "W": {(0,1): 1, (1,1): -1} # C[1,0] and C[1,1]? 
            # Wait, C[1,0] is k=0, i=1. C[1,1] is k=1, i=1.
            # Formula: C[1,0] = M2 + M4. C[1,1] = M1 - M2 + M3 + M6.
            # So W for M2 should be: W[0,1] = 1 (k=0, i=1), W[1,1] = -1 (k=1, i=1).
        },
        {
            "U": {(0,0): 1},
            "V": {(1,0): 1, (1,1): -1},
            "W": {(1,0): 1, (1,1): 1} # C[0,1] and C[1,1]?
            # Formula: C[0,1] = M3 + M5. C[1,1] = ... + M3.
            # So W for M3: W[1,0] = 1 (k=1, i=0), W[1,1] = 1 (k=1, i=1).
        },
        {
            "U": {(1,1): 1},
            "V": {(0,1): 1, (0,0): -1},
            "W": {(0,0): 1, (0,1): 1} # C[0,0] and C[1,0]?
            # Formula: C[0,0] = M1 + M4 ... C[1,0] = M2 + M4.
            # So W for M4: W[0,0] = 1 (k=0, i=0), W[0,1] = 1 (k=0, i=1).
        },
        {
            "U": {(0,0): 1, (0,1): 1},
            "V": {(1,1): 1},
            "W": {(0,0): -1, (1,0): 1} # C[0,0] and C[0,1]?
            # Formula: C[0,0] = ... - M5. C[0,1] = M3 + M5.
            # So W for M5: W[0,0] = -1 (k=0, i=0), W[1,0] = 1 (k=1, i=0).
        },
        {
            "U": {(1,0): 1, (0,0): -1},
            "V": {(0,0): 1, (0,1): 1},
            "W": {(1,1): 1} # C[1,1]?
            # Formula: C[1,1] = ... + M6.
            # So W for M6: W[1,1] = 1 (k=1, i=1).
        },
        {
            "U": {(0,1): 1, (1,1): -1},
            "V": {(1,0): 1, (1,1): 1},
            "W": {(0,0): 1} # C[0,0]?
            # Formula: C[0,0] = ... + M7.
            # So W for M7: W[0,0] = 1 (k=0, i=0).
        }
    ]
    
    # Apply Strassen terms for B[0:2, 0:2]
    for term in strassen_terms:
        for (i, j), val in term["U"].items():
            U[i * m + j][col] += val
        for (j, k), val in term["V"].items():
            V[j * p + k][col] += val
        for (k, i), val in term["W"].items():
            W[k * n + i][col] += val
        col += 1
        
    # Apply Strassen terms for B[0:2, 2:4]
    # Map local k to global k: local 0 -> global 2, local 1 -> global 3
    for term in strassen_terms:
        for (i, j), val in term["U"].items():
            U[i * m + j][col] += val
        for (j, k_local), val in term["V"].items():
            k_global = k_local + 2
            V[j * p + k_global][col] += val
        for (k_local, i), val in term["W"].items():
            k_global = k_local + 2
            W[k_global * n + i][col] += val
        col += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
