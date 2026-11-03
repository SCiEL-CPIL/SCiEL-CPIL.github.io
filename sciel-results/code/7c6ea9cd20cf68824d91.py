"""Tensor decomposition for 2x4 * 4x6 matrix multiplication."""
import numpy as np

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import time

def construct():
    """
    Construct a rank-42 tensor decomposition for 2x4 * 4x6 matrix multiplication.
    Uses Strassen's algorithm on 2x2x2 blocks.
    
    C[i,k] = sum_j A[i,j] * B[j,k] for i in {0,1}, k in {0,...,5}, j in {0,...,3}
    
    We group columns of A / rows of B into pairs:
    j-group 0: j=0,1
    j-group 1: j=2,3
    
    For each j-group g, and each column block c (k=2c, 2c+1):
    C[i, 2c+q] += sum_{r=0,1} A[i, 2g+r] * B[2g+r, 2c+q]
    
    This is a 2x2 * 2x2 multiplication per (g, c) pair, giving 2*3=6 instances.
    Using Strassen (rank 7) for each: total rank = 6*7 = 42.
    
    Output convention: W[k*n+i, r] where k is column, i is row.
    """
    deadline = time.time() + 145  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 2, 4, 6
    
    rank = 42
    U = np.zeros((n * m, rank))  # (8, 42)
    V = np.zeros((m * p, rank))  # (24, 42)
    W = np.zeros((p * n, rank))  # (12, 42)
    
    # Strassen's 7 products for 2x2 * 2x2:
    # M1 = (P00+P11)*(Q00+Q11)
    # M2 = (P10+P11)*Q00
    # M3 = P00*(Q01-Q11)
    # M4 = P11*(Q10-Q00)
    # M5 = (P00+P01)*Q11
    # M6 = (P10-P00)*(Q00+Q01)
    # M7 = (P01-P11)*(Q10+Q11)
    #
    # R00 = M1 - M4 + M5 - M7
    # R01 = M3 + M5
    # R10 = M2 + M4
    # R11 = M1 - M2 + M3 + M6
    
    strassen_terms = [
        # M1 = (P00+P11)*(Q00+Q11)
        ([((0,0), 1), ((1,1), 1)],
         [((0,0), 1), ((1,1), 1)],
         [((0,0), 1), ((1,1), 1)]),
        # M2 = (P10+P11)*Q00
        ([((1,0), 1), ((1,1), 1)],
         [((0,0), 1)],
         [((1,0), 1), ((1,1), -1)]),
        # M3 = P00*(Q01-Q11)
        ([((0,0), 1)],
         [((0,1), 1), ((1,1), -1)],
         [((0,1), 1), ((1,1), 1)]),
        # M4 = P11*(Q10-Q00)
        ([((1,1), 1)],
         [((1,0), 1), ((0,0), -1)],
         [((0,0), 1), ((1,0), 1)]),
        # M5 = (P00+P01)*Q11
        ([((0,0), 1), ((0,1), 1)],
         [((1,1), 1)],
         [((0,0), -1), ((0,1), 1)]),
        # M6 = (P10-P00)*(Q00+Q01)
        ([((1,0), 1), ((0,0), -1)],
         [((0,0), 1), ((0,1), 1)],
         [((1,1), 1)]),
        # M7 = (P01-P11)*(Q10+Q11)
        ([((0,1), 1), ((1,1), -1)],
         [((1,0), 1), ((1,1), 1)],
         [((0,0), 1)]),
    ]
    
    col_idx = 0
    for source in range(2):  # j-group: 0 -> j=0,1; 1 -> j=2,3
        for c in range(3):  # column block of B/C: 0 -> k=0,1; 1 -> k=2,3; 2 -> k=4,5
            for t in range(7):
                if time.time() > deadline:
                    # Return current partial result (should not happen for this small problem)
                    return {
                        "U": [[float(x) for x in row] for row in U],
                        "V": [[float(x) for x in row] for row in V],
                        "W": [[float(x) for x in row] for row in W]
                    }
                
                u_factors, v_factors, w_factors = strassen_terms[t]
                
                for (pi, pj), coeff in u_factors:
                    u_row = pi * m + (2 * source + pj)
                    U[u_row, col_idx] += coeff
                
                for (qi, qj), coeff in v_factors:
                    v_row = (2 * source + qi) * p + (2 * c + qj)
                    V[v_row, col_idx] += coeff
                
                for (ri, rj), coeff in w_factors:
                    w_row = (2 * c + rj) * n + ri
                    W[w_row, col_idx] += coeff
                
                col_idx += 1
    
    return {
        "U": [[float(x) for x in row] for row in U],
        "V": [[float(x) for x in row] for row in V],
        "W": [[float(x) for x in row] for row in W]
    }
# EVOLVE-BLOCK-END
