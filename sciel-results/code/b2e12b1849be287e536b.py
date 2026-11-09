import numpy as np
import time

def construct():
    n, m, p = 4, 4, 6
    rank = 77
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen 2x2 coefficients
    # u2[k', i, j], v2[k', i, j]
    # k' in 0..6, i,j in 0..1
    # Index mapping: 0->00, 1->01, 2->10, 3->11
    u2 = np.zeros((7, 2, 2))
    v2 = np.zeros((7, 2, 2))
    
    u2[0, 0, 0] = 1; u2[0, 1, 1] = 1
    v2[0, 0, 0] = 1; v2[0, 1, 1] = 1
    
    u2[1, 1, 0] = 1; u2[1, 1, 1] = 1
    v2[1, 0, 0] = 1
    
    u2[2, 0, 0] = 1
    v2[2, 0, 1] = 1; v2[2, 1, 1] = -1
    
    u2[3, 1, 1] = 1
    v2[3, 1, 0] = 1; v2[3, 0, 0] = -1
    
    u2[4, 0, 0] = 1; u2[4, 0, 1] = 1
    v2[4, 1, 1] = 1
    
    u2[5, 1, 0] = 1; u2[5, 0, 0] = -1
    v2[5, 0, 0] = 1; v2[5, 0, 1] = 1
    
    u2[6, 0, 1] = 1; u2[6, 1, 1] = -1
    v2[6, 1, 0] = 1; v2[6, 1, 1] = 1
    
    # w2[k', i, k]
    w2 = np.zeros((7, 2, 2))
    # Z00 = M0 + M3 - M5 + M6
    w2[0, 0, 0] = 1; w2[3, 0, 0] = 1; w2[5, 0, 0] = -1; w2[6, 0, 0] = 1
    # Z01 = M2 + M4 + M5
    w2[2, 0, 1] = 1; w2[4, 0, 1] = 1; w2[5, 0, 1] = 1
    # Z10 = M1 + M2 + M6
    w2[1, 1, 0] = 1; w2[2, 1, 0] = 1; w2[6, 1, 0] = 1
    # Z11 = M0 - M1 + M3 + M4
    w2[0, 1, 1] = 1; w2[1, 1, 1] = -1; w2[3, 1, 1] = 1; w2[4, 1, 1] = 1
    
    # Block Strassen for [2,2,2] (cols 0,1)
    # s in 0..6
    # cA[s, I, J], cB[s, J, K] (K in 0,1)
    cA = np.zeros((7, 2, 2))
    cB = np.zeros((7, 2, 2)) # J in 0,1, K in 0,1
    
    cA[0, 0, 0] = 1; cA[0, 1, 1] = 1
    cB[0, 0, 0] = 1; cB[0, 1, 1] = 1
    
    cA[1, 1, 0] = 1; cA[1, 1, 1] = 1
    cB[1, 0, 0] = 1
    
    cA[2, 0, 0] = 1
    cB[2, 0, 1] = 1; cB[2, 1, 1] = -1
    
    cA[3, 1, 1] = 1
    cB[3, 1, 0] = 1; cB[3, 0, 0] = -1
    
    cA[4, 0, 0] = 1; cA[4, 0, 1] = 1
    cB[4, 1, 1] = 1
    
    cA[5, 1, 0] = 1; cA[5, 0, 0] = -1
    cB[5, 0, 0] = 1; cB[5, 0, 1] = 1
    
    cA[6, 0, 1] = 1; cA[6, 1, 1] = -1
    cB[6, 1, 0] = 1; cB[6, 1, 1] = 1
    
    # w_block[s, I, K] (I in 0,1, K in 0,1)
    w_block = np.zeros((7, 2, 2))
    w_block[0, 0, 0] = 1; w_block[3, 0, 0] = 1; w_block[5, 0, 0] = -1; w_block[6, 0, 0] = 1
    w_block[2, 0, 1] = 1; w_block[4, 0, 1] = 1; w_block[5, 0, 1] = 1
    w_block[1, 1, 0] = 1; w_block[2, 1, 0] = 1; w_block[6, 1, 0] = 1
    w_block[0, 1, 1] = 1; w_block[1, 1, 1] = -1; w_block[3, 1, 1] = 1; w_block[4, 1, 1] = 1
    
    col = 0
    # Part 1: Strassen for cols 0,1
    for s in range(7):
        for k_prime in range(7):
            # U
            for I in range(2):
                for J in range(2):
                    coeff_A = cA[s, I, J]
                    if coeff_A != 0:
                        for i in range(2):
                            for j in range(2):
                                coeff_u2 = u2[k_prime, i, j]
                                if coeff_u2 != 0:
                                    I_glob = I * 2 + i
                                    J_glob = J * 2 + j
                                    U[I_glob * m + J_glob, col] += coeff_A * coeff_u2
            # V
            for J in range(2):
                for K in range(2):
                    coeff_B = cB[s, J, K]
                    if coeff_B != 0:
                        for i in range(2):
                            for j in range(2):
                                coeff_v2 = v2[k_prime, i, j]
                                if coeff_v2 != 0:
                                    J_glob = J * 2 + i # B row index J_glob? No.
                                    # B is (m, p) = (4, 6).
                                    # B index is (J_glob, K_glob).
                                    # J_glob = J * 2 + i_sub?
                                    # In Y_s, the index (i,j) of the 2x2 matrix Y_s corresponds to
                                    # block row i? No.
                                    # Y_s is a 2x2 matrix. Its entries are sums of B blocks.
                                    # (Y_s)_{i,j} = sum_{J,K} cB[s,J,K] (B_{J,K})_{i,j}
                                    # So (B_{J,K})_{i,j} has global indices:
                                    # Row: J * 2 + i
                                    # Col: K * 2 + j
                                    J_glob = J * 2 + i
                                    K_glob = K * 2 + j
                                    V[J_glob * p + K_glob, col] += coeff_B * coeff_v2
            # W
            for I in range(2):
                for K in range(2):
                    coeff_W_block = w_block[s, I, K]
                    if coeff_W_block != 0:
                        for i in range(2):
                            for k in range(2):
                                coeff_w2 = w2[k_prime, i, k]
                                if coeff_w2 != 0:
                                    I_glob = I * 2 + i
                                    K_glob = K * 2 + k
                                    W[K_glob * n + I_glob, col] += coeff_W_block * coeff_w2
            col += 1
            
    # Part 2: Schoolbook for col 2
    # Terms: (0,0,0,2), (0,1,1,2), (1,0,0,2), (1,1,1,2)
    terms = [
        (0, 0, 0, 2),
        (0, 1, 1, 2),
        (1, 0, 0, 2),
        (1, 1, 1, 2)
    ]
    for I, J, Jb, K in terms:
        # cA[s, I, J] = 1, cB[s, Jb, K] = 1, w_block[s, I, K] = 1
        # Note: J in A and Jb in B must match?
        # C_IK = A_IJ B_JK.
        # Here A block is A_IJ, B block is B_JK.
        # So J must equal Jb.
        # In my terms list:
        # (0,0,0,2): A_00 B_02. J=0, Jb=0. OK.
        # (0,1,1,2): A_01 B_12. J=1, Jb=1. OK.
        # (1,0,0,2): A_10 B_02. J=0, Jb=0. OK.
        # (1,1,1,2): A_11 B_12. J=1, Jb=1. OK.
        
        for k_prime in range(7):
            # U
            for i in range(2):
                for j in range(2):
                    coeff_u2 = u2[k_prime, i, j]
                    if coeff_u2 != 0:
                        I_glob = I * 2 + i
                        J_glob = J * 2 + j
                        U[I_glob * m + J_glob, col] += 1.0 * coeff_u2
            # V
            for i in range(2):
                for j in range(2):
                    coeff_v2 = v2[k_prime, i, j]
                    if coeff_v2 != 0:
                        J_glob = Jb * 2 + i
                        K_glob = K * 2 + j
                        V[J_glob * p + K_glob, col] += 1.0 * coeff_v2
            # W
            for i in range(2):
                for k in range(2):
                    coeff_w2 = w2[k_prime, i, k]
                    if coeff_w2 != 0:
                        I_glob = I * 2 + i
                        K_glob = K * 2 + k
                        W[K_glob * n + I_glob, col] += 1.0 * coeff_w2
            col += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
