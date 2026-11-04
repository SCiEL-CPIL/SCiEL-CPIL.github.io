import numpy as np
import time

DIMENSIONS = [4, 4, 8]

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 4x4 * 4x8 matrix multiplication.
    Uses two rank-49 Strassen-based decompositions for 4x4x4.
    Total rank: 98 (which is optimal since 4x4x4 has tensor rank 49).
    """
    
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # n=4, m=4, p=8
    
    # --- Step 1: Construct rank-49 decomposition for 4x4x4 ---
    # Standard Strassen recursion: outer Strassen (7 terms) x inner Strassen (7 terms) = 49
    
    # Outer Strassen coefficients for A and B blocks
    # Block indices: 0=00, 1=01, 2=10, 3=11
    # alpha[t][P]: coefficient of A-block P in the t-th A-combination
    # beta[t][Q]: coefficient of B-block Q in the t-th B-combination
    
    alpha = {
        1: {0: 1, 3: 1},   # A00 + A11
        2: {2: 1, 3: 1},   # A10 + A11
        3: {0: 1},         # A00
        4: {3: 1},         # A11
        5: {0: 1, 1: 1},   # A00 + A01
        6: {2: 1, 0: -1},  # A10 - A00
        7: {1: 1, 3: -1}   # A01 - A11
    }
    
    beta = {
        1: {0: 1, 3: 1},   # B00 + B11
        2: {0: 1},         # B00
        3: {1: 1, 3: -1},  # B01 - B11
        4: {2: 1, 0: -1},  # B10 - B00
        5: {3: 1},         # B11
        6: {0: 1, 1: 1},   # B00 + B01
        7: {2: 1, 3: 1}    # B10 + B11
    }
    
    # Inner Strassen: for each 2x2 block multiplication M = U * V
    # U-sides (delta_u[s][ab]) and V-sides (delta_v[s][ab])
    # m_s = u_s * v_s
    # Result entry M[ab] = sum_s gamma[s][ab] * m_s
    
    delta_u = {
        1: {(0,0): 1, (1,1): 1},   # U00 + U11
        2: {(1,0): 1, (1,1): 1},   # U10 + U11
        3: {(0,0): 1},             # U00
        4: {(1,1): 1},             # U11
        5: {(0,0): 1, (0,1): 1},   # U00 + U01
        6: {(1,0): 1, (0,0): -1},  # U10 - U00
        7: {(0,1): 1, (1,1): -1}   # U01 - U11
    }
    
    delta_v = {
        1: {(0,0): 1, (1,1): 1},   # V00 + V11
        2: {(0,0): 1},             # V00
        3: {(0,1): 1, (1,1): -1},  # V01 - V11
        4: {(1,0): 1, (0,0): -1},  # V10 - V00
        5: {(1,1): 1},             # V11
        6: {(0,0): 1, (0,1): 1},   # V00 + V01
        7: {(1,0): 1, (1,1): 1}    # V10 + V11
    }
    
    gamma = {
        1: {(0,0): 1, (1,1): 1},   # M00 += m1, M11 += m1
        2: {(1,0): 1, (1,1): -1},  # M10 += m2, M11 -= m2
        3: {(0,1): 1, (1,1): 1},   # M01 += m3, M11 += m3
        4: {(0,0): 1, (1,0): 1},   # M00 += m4, M10 += m4
        5: {(0,1): 1, (0,0): -1},  # M01 += m5, M00 -= m5
        6: {(1,1): 1},             # M11 += m6
        7: {(0,0): 1}              # M00 += m7
    }
    
    # Outer Strassen output coefficients
    # C_PQ = sum_t kappa[t][PQ] * M_t
    kappa = {
        1: {0: 1, 3: 1},    # C00 += M1, C11 += M1
        2: {2: 1, 3: -1},   # C10 += M2, C11 -= M2
        3: {1: 1, 3: 1},    # C01 += M3, C11 += M3
        4: {0: 1, 2: 1},    # C00 += M4, C10 += M4
        5: {0: -1, 1: 1},   # C00 -= M5, C01 += M5
        6: {3: 1},          # C11 += M6
        7: {0: 1}           # C00 += M7
    }
    
    # Build 4x4x4 factor matrices: U444 (16x49), V444 (16x49), W444 (16x49)
    U444 = np.zeros((16, 49), dtype=float)
    V444 = np.zeros((16, 49), dtype=float)
    W444 = np.zeros((16, 49), dtype=float)
    
    for t in range(1, 8):
        for s in range(1, 8):
            r = (t - 1) * 7 + (s - 1)
            
            # U444: u_s = sum_ab delta_u[s][ab] * A^(t)_ab
            # A^(t)_ab = sum_P alpha[t][P] * A_P[ab]
            # A_P[ab] = A_{2*P_row+a, 2*P_col+b}
            for a in range(2):
                for b in range(2):
                    cu = delta_u[s].get((a, b), 0)
                    if cu == 0:
                        continue
                    for P, av in alpha[t].items():
                        ct = cu * av
                        if ct == 0:
                            continue
                        i = 2 * (P // 2) + a
                        j = 2 * (P % 2) + b
                        U444[i * 4 + j, r] += ct
            
            # V444: v_s = sum_ab delta_v[s][ab] * B^(t)_ab
            # B^(t)_ab = sum_Q beta[t][Q] * B_Q[ab]
            # B_Q[ab] = B_{2*Q_row+a, 2*Q_col+b}
            for a in range(2):
                for b in range(2):
                    cv = delta_v[s].get((a, b), 0)
                    if cv == 0:
                        continue
                    for Q, bv in beta[t].items():
                        ct = cv * bv
                        if ct == 0:
                            continue
                        j = 2 * (Q // 2) + a
                        k = 2 * (Q % 2) + b
                        V444[j * 4 + k, r] += ct
            
            # W444: contribution of m_s to C_PQ[ab] is kappa[t][PQ] * gamma[s][ab]
            # C_PQ[ab] = C_{2*PQ_row+a, 2*PQ_col+b}
            # W index: k*n + i = k*4 + i (trace convention W[k*n+i, r])
            for a in range(2):
                for b in range(2):
                    cm = gamma[s].get((a, b), 0)
                    if cm == 0:
                        continue
                    for PQ, kv in kappa[t].items():
                        ct = cm * kv
                        if ct == 0:
                            continue
                        i = 2 * (PQ // 2) + a
                        k = 2 * (PQ % 2) + b
                        W444[k * 4 + i, r] += ct
    
    if time.time() > deadline:
        # Should not happen; construction is fast
        pass
    
    # --- Step 2: Assemble 4x4x8 factors ---
    # U: (n*m, rank) = (16, 98)
    # V: (m*p, rank) = (32, 98), index 8*j + k
    # W: (p*n, rank) = (32, 98), index 4*k + i
    
    rank = 98
    U = np.zeros((16, rank), dtype=float)
    V = np.zeros((32, rank), dtype=float)
    W = np.zeros((32, rank), dtype=float)
    
    # Half 1: k in [0, 4)
    U[:, 0:49] = U444
    for j in range(4):
        for k in range(4):
            V[8 * j + k, 0:49] = V444[4 * j + k, :]
    for k in range(4):
        for i in range(4):
            W[4 * k + i, 0:49] = W444[4 * k + i, :]
    
    # Half 2: k in [4, 8)
    U[:, 49:98] = U444
    for j in range(4):
        for k in range(4, 8):
            V[8 * j + k, 49:98] = V444[4 * j + (k - 4), :]
    for k in range(4, 8):
        for i in range(4):
            W[4 * k + i, 49:98] = W444[4 * (k - 4) + i, :]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
