"""Matrix multiplication 2x3x6 with Strassen-based decomposition."""
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Decompose 2x3 * 3x6 matrix multiplication.
    
    Strategy:
    - Split the 3 middle indices into 3 groups of size 2: (j=0,1), (j=2,3), (j=4,5)
    - For each group t (t=0,1,2), the sub-problem is 2x2 * 2x2 (rows i=0,1; cols k=2t,2t+1)
    - Use Strassen's rank-7 algorithm for each 2x2x2 sub-problem
    - Total rank: 3 * 7 = 21
    
    The 2x2x2 Strassen decomposition:
    For A (2x2) and B (2x2):
    m1 = (A00+A11)(B00+B11)
    m2 = (A10+A11)B00
    m3 = A00(B10-B11)
    m4 = A11(B01-B10)
    m5 = (A00+A01)B11
    m6 = (A10-A00)(B00+B10)
    m7 = (A01-A11)(B01+B11)
    
    C00 = m1 + m4 - m5 + m7
    C01 = m3 + m5
    C10 = m2 + m4
    C11 = m1 - m2 + m3 - m6
    """
    n, m, p = 2, 3, 6
    
    # For each of the 3 blocks of 2 columns in the 3x6 matrix
    # Block t uses middle indices j = 2t, 2t+1 and output columns k = 2t, 2t+1
    rank = 21
    U = [[0] * rank for _ in range(n * m)]  # 6 x 21
    V = [[0] * rank for _ in range(m * p)]  # 18 x 21
    W = [[0] * rank for _ in range(p * n)]  # 12 x 21
    
    # Strassen coefficients for 2x2x2
    # Each term r has:
    #   U_vec: (n*m=6 elements) - coefficients for A[i,j] where j is local (0,1)
    #   V_vec: (m*p=18 elements) - coefficients for B[j,k] where j is local (0,1), k is local (0,1)
    #   W_vec: (p*n=12 elements) - coefficients for C[k,n+i]
    
    # For block t, local middle index j_local=0 -> global j=2t, j_local=1 -> global j=2t+1
    # Local output col k_local=0 -> global k=2t, k_local=1 -> global k=2t+1
    
    # Strassen terms (m1 to m7):
    # Each term has U_part (coeffs for A00,A01,A10,A11 in local coords)
    #                V_part (coeffs for B00,B01,B10,B11 in local coords)
    #                W_part (coeffs for C00,C01,C10,C11 in local coords)
    
    strassen_terms = [
        # m1 = (A00+A11)(B00+B11)
        {'U': [1, 0, 0, 1], 'V': [1, 0, 0, 1], 'W': [1, 0, 0, 1]},
        # m2 = (A10+A11)B00
        {'U': [0, 0, 1, 1], 'V': [1, 0, 0, 0], 'W': [0, 0, 1, -1]},
        # m3 = A00(B10-B11)
        {'U': [1, 0, 0, 0], 'V': [0, 0, 1, -1], 'W': [0, 1, 0, 1]},
        # m4 = A11(B01-B10)
        {'U': [0, 0, 0, 1], 'V': [0, 1, -1, 0], 'W': [1, 0, 1, 0]},
        # m5 = (A00+A01)B11
        {'U': [1, 1, 0, 0], 'V': [0, 0, 0, 1], 'W': [-1, 1, 0, 0]},
        # m6 = (A10-A00)(B00+B10)
        {'U': [-1, 0, 1, 0], 'V': [1, 0, 1, 0], 'W': [0, 0, 0, -1]},
        # m7 = (A01-A11)(B01+B11)
        {'U': [0, 1, 0, -1], 'V': [0, 1, 0, 1], 'W': [1, 0, 0, 0]},
    ]
    
    for t in range(3):
        for term_idx in range(7):
            col = t * 7 + term_idx
            term = strassen_terms[term_idx]
            
            # U: A[i, j] where i in {0,1}, j in {2t, 2t+1}
            # U indices: i*m + j = i*3 + j
            # term['U'] = [A00, A01, A10, A11] in local coords
            # A00 -> i=0, j_local=0 -> global j=2t
            # A01 -> i=0, j_local=1 -> global j=2t+1
            # A10 -> i=1, j_local=0 -> global j=2t
            # A11 -> i=1, j_local=1 -> global j=2t+1
            
            u_local = term['U']
            U[0 * m + (2 * t)][col] = u_local[0]   # A[0, 2t]
            U[0 * m + (2 * t + 1)][col] = u_local[1]  # A[0, 2t+1]
            U[1 * m + (2 * t)][col] = u_local[2]   # A[1, 2t]
            U[1 * m + (2 * t + 1)][col] = u_local[3]  # A[1, 2t+1]
            
            # V: B[j, k] where j in {2t, 2t+1}, k in {2t, 2t+1}
            # V indices: j*p + k = j*6 + k
            # term['V'] = [B00, B01, B10, B11] in local coords
            # B00 -> j_local=0, k_local=0 -> global j=2t, k=2t
            # B01 -> j_local=0, k_local=1 -> global j=2t, k=2t+1
            # B10 -> j_local=1, k_local=0 -> global j=2t+1, k=2t
            # B11 -> j_local=1, k_local=1 -> global j=2t+1, k=2t+1
            
            v_local = term['V']
            V[(2 * t) * p + (2 * t)][col] = v_local[0]          # B[2t, 2t]
            V[(2 * t) * p + (2 * t + 1)][col] = v_local[1]      # B[2t, 2t+1]
            V[(2 * t + 1) * p + (2 * t)][col] = v_local[2]      # B[2t+1, 2t]
            V[(2 * t + 1) * p + (2 * t + 1)][col] = v_local[3]  # B[2t+1, 2t+1]
            
            # W: C[k, i] with trace convention W[k*n + i, col]
            # term['W'] = [C00, C01, C10, C11] in local coords
            # C00 -> k_local=0, i=0 -> global k=2t, i=0 -> index (2t)*n + 0
            # C01 -> k_local=1, i=0 -> global k=2t+1, i=0 -> index (2t+1)*n + 0
            # C10 -> k_local=0, i=1 -> global k=2t, i=1 -> index (2t)*n + 1
            # C11 -> k_local=1, i=1 -> global k=2t+1, i=1 -> index (2t+1)*n + 1
            
            w_local = term['W']
            W[(2 * t) * n + 0][col] = w_local[0]      # C[2t, 0]
            W[(2 * t + 1) * n + 0][col] = w_local[1]  # C[2t+1, 0]
            W[(2 * t) * n + 1][col] = w_local[2]      # C[2t, 1]
            W[(2 * t + 1) * n + 1][col] = w_local[3]  # C[2t+1, 1]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
