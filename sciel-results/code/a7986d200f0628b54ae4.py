"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 6
    
    # We use a Strassen-style block decomposition.
    # A is 2x4, B is 4x6.
    # Split A into 2 blocks of size 2x2: A = [A0, A1]
    # Split B into 3 blocks of size 4x2: B = [B0, B1, B2]
    # Each Bj is 4x2, split into top 2x2 and bottom 2x2.
    # C is 2x6.
    #
    # This results in 6 independent 2x2 * 2x2 multiplications.
    # Each uses Strassen (rank 7).
    # Total rank = 6 * 7 = 42.
    
    rank = 42
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Strassen coefficients for 2x2*2x2:
    # M1 = (P00+P11)*(Q00+Q11)
    # M2 = (P10+P11)*Q00
    # M3 = P00*(Q01-Q11)
    # M4 = P11*(Q10-Q00)
    # M5 = (P00+P01)*Q11
    # M6 = (P10-P00)*(Q00+Q01)
    # M7 = (P01-P11)*(Q10+Q11)
    #
    # Result:
    # R00 = M1 + M4 - M5 + M7
    # R01 = M3 + M5
    # R10 = M2 + M4
    # R11 = M1 - M2 + M3 + M6
    
    # Define the 7 Strassen terms as (U_factors, V_factors, W_factors)
    # Each factor is a list of (local_index, coeff)
    # Local indices for P (2x2): 0:P00, 1:P01, 2:P10, 3:P11
    # Local indices for Q (2x2): 0:Q00, 1:Q01, 2:Q10, 3:Q11
    # Local indices for R (2x2): 0:R00, 1:R01, 2:R10, 3:R11
    
    strassen_terms = [
        # M1
        ([ (0, 1), (3, 1) ], [ (0, 1), (3, 1) ], [ (0, 1), (3, 1) ]),
        # M2
        ([ (2, 1), (3, 1) ], [ (0, 1) ], [ (2, 1), (3, -1) ]),
        # M3
        ([ (0, 1) ], [ (1, 1), (3, -1) ], [ (1, 1), (3, 1) ]),
        # M4
        ([ (3, 1) ], [ (2, 1), (0, -1) ], [ (0, 1), (2, 1) ]),
        # M5
        ([ (0, 1), (1, 1) ], [ (3, 1) ], [ (0, -1), (1, 1) ]),
        # M6
        ([ (2, 1), (0, -1) ], [ (0, 1), (1, 1) ], [ (3, 1) ]),
        # M7
        ([ (1, 1), (3, -1) ], [ (2, 1), (3, 1) ], [ (0, 1) ]),
    ]
    
    col_idx = 0
    for source in range(2):  # 0=top (A0, B_top), 1=bot (A1, B_bot)
        for c in range(3):  # column block 0, 1, 2
            for t in range(7):
                u_factors, v_factors, w_factors = strassen_terms[t]
                
                # Map P indices to U indices
                # P is 2x2. P[pi, pj]
                # If source=0: P[pi,pj] = A[pi, pj] -> U[pi*4 + pj, col_idx]
                # If source=1: P[pi,pj] = A[pi, pj+2] -> U[pi*4 + (pj+2), col_idx]
                for (local_idx, coeff) in u_factors:
                    pi = local_idx // 2
                    pj = local_idx % 2
                    u_row = pi * 4 + (pj + 2*source)
                    U[u_row, col_idx] += coeff
                
                # Map Q indices to V indices
                # Q is 2x2. Q[qi, qj]
                # Q[qi,qj] = B[qi + 2*source, 2*c + qj]
                # V index: (qi + 2*source) * 6 + (2*c + qj)
                for (local_idx, coeff) in v_factors:
                    qi = local_idx // 2
                    qj = local_idx % 2
                    v_row = (qi + 2*source) * 6 + (2*c + qj)
                    V[v_row, col_idx] += coeff
                
                # Map R indices to W indices
                # R is 2x2. R[ri, rj]
                # R[ri,rj] contributes to C[ri, 2*c + rj]
                # W index: (2*c + rj) * 2 + ri  (since W is p*n = 6*2 = 12, indexed by k*n+i)
                for (local_idx, coeff) in w_factors:
                    ri = local_idx // 2
                    rj = local_idx % 2
                    w_row = (2*c + rj) * 2 + ri
                    W[w_row, col_idx] += coeff
                
                col_idx += 1
    
    # Convert to lists of floats
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
