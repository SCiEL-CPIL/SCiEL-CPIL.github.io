"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Seed decomposition (Schoolbook)
    rank_seed = n * m * p
    U_seed = [[0] * rank_seed for _ in range(n * m)]
    V_seed = [[0] * rank_seed for _ in range(m * p)]
    W_seed = [[0] * rank_seed for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_seed[i * m + j][col] = 1
                V_seed[j * p + k][col] = 1
                W_seed[k * n + i][col] = 1
                col += 1
    
    best_U, best_V, best_W = U_seed, V_seed, W_seed
    best_rank = rank_seed
    
    # Try Strassen 2x2 blocks of 3x3
    if time.time() < deadline:
        try:
            U_new, V_new, W_new, rank_new = _strassen_block_6x6()
            if rank_new < best_rank:
                best_U, best_V, best_W, best_rank = U_new, V_new, W_new, rank_new
        except Exception:
            pass
            
    return {"U": best_U, "V": best_V, "W": best_W}

def _strassen_block_6x6():
    # 2x2 blocks of 3x3 matrices
    # A blocks: A0, A1, A2, A3
    # B blocks: B0, B1, B2, B3
    # C blocks: C0, C1, C2, C3
    
    # Strassen coefficients
    # X1 = A0 + A3, Y1 = B0 + B3
    # X2 = A2 + A3, Y2 = B0
    # X3 = A0, Y3 = B1 - B3
    # X4 = A3, Y4 = B2 - B0
    # X5 = A0 + A1, Y5 = B3
    # X6 = A2 - A0, Y6 = B0 + B1
    # X7 = A1 - A3, Y7 = B2 + B3
    
    # C0 = Z1 + Z4 - Z5 + Z7
    # C1 = Z3 + Z5
    # C2 = Z2 + Z4
    # C3 = Z1 - Z2 + Z3 + Z6
    
    # alpha[k][p] for Xk = sum alpha A_p
    # beta[k][q] for Yk = sum beta B_q
    # w_coeff[k][m] for Zk contributes to Cm
    
    alpha = [
        [1, 0, 0, 1], # X1
        [0, 0, 1, 1], # X2
        [1, 0, 0, 0], # X3
        [0, 0, 0, 1], # X4
        [1, 1, 0, 0], # X5
        [1, 0, 1, 0], # X6 (A2 - A0 -> -1, 0, 1, 0) -- Wait, X6 = A2 - A0. So alpha[5] = [-1, 0, 1, 0]
        [0, 1, 0, -1] # X7 (A1 - A3 -> 0, 1, 0, -1)
    ]
    # Correction:
    # X1 = A0 + A3 -> [1, 0, 0, 1]
    # X2 = A2 + A3 -> [0, 0, 1, 1]
    # X3 = A0 -> [1, 0, 0, 0]
    # X4 = A3 -> [0, 0, 0, 1]
    # X5 = A0 + A1 -> [1, 1, 0, 0]
    # X6 = A2 - A0 -> [-1, 0, 1, 0]
    # X7 = A1 - A3 -> [0, 1, 0, -1]
    
    alpha = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    
    beta = [
        [1, 0, 0, 1], # Y1
        [1, 0, 0, 0], # Y2
        [0, 1, 0, -1], # Y3
        [0, 0, 1, -1], # Y4
        [0, 0, 0, 1], # Y5
        [1, 1, 0, 0], # Y6
        [0, 0, 1, 1] # Y7
    ]
    
    w_coeff = [
        [1, 0, 0, 1], # Z1 -> C0, C3
        [0, 0, 1, -1], # Z2 -> C2, C3
        [0, 1, 0, 1], # Z3 -> C1, C3
        [1, 0, 1, 0], # Z4 -> C0, C2
        [-1, 1, 0, 0], # Z5 -> C0, C1
        [0, 0, 0, 1], # Z6 -> C3
        [1, 0, 0, 0] # Z7 -> C0
    ]
    
    # Collect terms
    terms = []
    
    for k in range(7):
        for a in range(3):
            for c in range(3):
                for b in range(3):
                    for p in range(4):
                        if alpha[k][p] == 0: continue
                        for q in range(4):
                            if beta[k][q] == 0: continue
                            for m in range(4):
                                if w_coeff[k][m] == 0: continue
                                coeff = alpha[k][p] * beta[k][q] * w_coeff[k][m]
                                if coeff == 0: continue
                                
                                # Global indices
                                # A_p is block p, local (a,c)
                                I_A = p * 9 + a * 3 + c
                                # B_q is block q, local (c,b)
                                I_B = q * 9 + c * 3 + b
                                # C_m is block m, local (a,b)
                                I_C = m * 9 + a * 3 + b
                                
                                terms.append((I_A, I_B, I_C, coeff))
    
    # Build U, V, W
    rank = len(terms)
    U = [[0] * rank for _ in range(36)]
    V = [[0] * rank for _ in range(36)]
    W = [[0] * rank for _ in range(36)]
    
    for r, (I_A, I_B, I_C, coeff) in enumerate(terms):
        U[I_A][r] = coeff
        V[I_B][r] = 1
        W[I_C][r] = 1
        
    return U, V, W, rank
# EVOLVE-BLOCK-END
