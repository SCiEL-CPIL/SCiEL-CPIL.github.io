import time
import sys

DIMENSIONS = [4, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen definitions
    # Outer level (Block level)
    # U_blocks[l] is list of (I, J, coeff)
    # V_blocks[l] is list of (J, K, coeff)
    U_blocks = [
        [(0, 0, 1)],
        [(1, 0, 1), (1, 1, 1)],
        [(0, 0, 1)],
        [(1, 1, 1)],
        [(0, 0, 1), (1, 0, 1)],
        [(1, 0, 1), (0, 0, -1)],
        [(0, 1, 1), (1, 1, 1)]
    ]
    V_blocks = [
        [(0, 0, 1), (1, 1, 1)],
        [(0, 0, 1)],
        [(0, 1, 1), (1, 1, -1)],
        [(1, 0, 1), (0, 0, -1)],
        [(1, 1, 1)],
        [(0, 0, 1), (0, 1, 1)],
        [(1, 0, 1), (1, 1, -1)]
    ]
    
    # Inner level (Scalar level)
    # U_scalars[m] is list of (a, b, coeff) where a,b are indices in 2x2 block
    # V_scalars[m] is list of (c, d, coeff)
    U_scalars = [
        [(0, 0, 1)],
        [(1, 0, 1), (1, 1, 1)],
        [(0, 0, 1)],
        [(1, 1, 1)],
        [(0, 0, 1), (1, 0, 1)],
        [(1, 0, 1), (0, 0, -1)],
        [(0, 1, 1), (1, 1, 1)]
    ]
    V_scalars = [
        [(0, 0, 1), (1, 1, 1)],
        [(0, 0, 1)],
        [(0, 1, 1), (1, 1, -1)],
        [(1, 0, 1), (0, 0, -1)],
        [(1, 1, 1)],
        [(0, 0, 1), (0, 1, 1)],
        [(1, 0, 1), (1, 1, -1)]
    ]
    
    # Reconstruction coefficients
    # P_l[d,e] is sum of M_{l,m} * coeff
    # P_to_M = [ [ [coeff for M0..M6 for P[0,0]], ... ] ]
    # P_l[0,0] = M0 + M3 - M4 + M6
    # P_l[0,1] = M2 + M4
    # P_l[1,0] = M1 + M3
    # P_l[1,1] = M0 - M1 + M2 + M5
    
    P_to_M = [
        [1, 0, 0, 1, -1, 0, 1], # P[0,0]
        [0, 0, 1, 0, 1, 0, 0],  # P[0,1]
        [0, 1, 0, 1, 0, 0, 0],  # P[1,0]
        [1, -1, 1, 0, 0, 1, 0]  # P[1,1]
    ]
    
    # C[K,I] is sum of P_l * coeff
    # C[0,0] = P0 + P3 - P4 + P6
    # C[0,1] = P2 + P4
    # C[1,0] = P1 + P3
    # C[1,1] = P0 - P1 + P2 + P5
    
    C_to_P = [
        [1, 0, 0, 1, -1, 0, 1], # C[0,0]
        [0, 0, 1, 0, 1, 0, 0],  # C[0,1]
        [0, 1, 0, 1, 0, 0, 0],  # C[1,0]
        [1, -1, 1, 0, 0, 1, 0]  # C[1,1]
    ]
    
    for l in range(7):
        for m in range(7):
            c = 7 * l + m
            
            # 1. Fill U
            # U^{(2)}_{l,m} is linear combo of U^{(1)}_l entries
            # U^{(1)}_l is linear combo of A blocks
            # We iterate through the components of U^{(2)}_{l,m}
            for a, b, coeff_u2 in U_scalars[m]:
                # This term corresponds to entry (a,b) of the block U^{(1)}_l
                # U^{(1)}_l is sum of coeff_u1 * A_{I,J}
                for I, J, coeff_u1 in U_blocks[l]:
                    # A_{I,J} consists of entries A[2I+a', 2J+b']
                    # The entry (a,b) of the block A_{I,J} is A[2I+a, 2J+b]
                    i_global = 2 * I + a
                    j_global = 2 * J + b
                    U[i_global * m + j_global][c] += coeff_u1 * coeff_u2
            
            # 2. Fill V
            # V^{(2)}_{l,m} is linear combo of V^{(1)}_l entries
            for c_idx, d, coeff_v2 in V_scalars[m]:
                for J, K, coeff_v1 in V_blocks[l]:
                    # B_{J,K} entry (c_idx, d) is B[2J+c_idx, 2K+d]
                    j_global = 2 * J + c_idx
                    k_global = 2 * K + d
                    V[j_global * p + k_global][c] += coeff_v1 * coeff_v2
            
            # 3. Fill W
            # M_{l,m} contributes to P_l[d,e] with coeff P_to_M[d,e][m]
            # P_l[d,e] contributes to C[K,I] with coeff C_to_P[K*2+I][l]
            # C[K,I] entry (d,e) is C[2K+d, 2I+e]
            # W row index is k*n + i = 4k + i. Here k=2K+d, i=2I+e.
            for K in range(2):
                for I in range(2):
                    coeff_c = C_to_P[K * 2 + I][l]
                    if coeff_c == 0:
                        continue
                    for d in range(2):
                        for e in range(2):
                            coeff_p = P_to_M[d * 2 + e][m]
                            if coeff_p == 0:
                                continue
                            total_coeff = coeff_c * coeff_p
                            k_global = 2 * K + d
                            i_global = 2 * I + e
                            W[k_global * n + i_global][c] += total_coeff
                            
    return {"U": U, "V": V, "W": W}
