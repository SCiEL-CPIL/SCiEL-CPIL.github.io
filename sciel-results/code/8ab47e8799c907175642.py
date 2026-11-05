import numpy as np
import time

def construct():
    start_time = time.time()
    DEADLINE = 140.0

    # Precomputed Strassen 2x2x2
    # U_s, V_s, W_s as (7, 4)
    U_s = [
        [1, 0, 1, 1, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [1, 1, 0, 0, 0, 1, 0],
        [0, 1, 0, 1, 0, 0, -1]
    ]
    # Transpose to (4, 7) for easier access?
    U_s = np.array(U_s).T
    
    V_s = [
        [1, 0, 0, 1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 1],
        [1, 0, -1, 1, 1, 0, 0],
        [0, 1, 1, 0, 0, 0, 1]
    ]
    V_s = np.array(V_s).T
    
    W_s = [
        [1, 0, 0, 1, 0, 0, 1],
        [0, 1, 1, 0, 1, 0, 0],
        [1, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, 1, 0, 0, -1]
    ]
    W_s = np.array(W_s).T
    
    # Schoolbook 2x2x1
    # U (4, 4), V (2, 4), W (2, 4)
    # Terms:
    # 0: A0 B0 C0
    # 1: A1 B1 C0
    # 2: A2 B0 C1
    # 3: A3 B1 C1
    U_sb = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]) # (4, 4)
    
    V_sb = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1]
    ]) # (2, 4)
    
    W_sb = np.array([
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]) # (2, 4)
    
    R = 77
    U = np.zeros((16, R))
    V = np.zeros((24, R))
    W = np.zeros((24, R))
    
    col = 0
    
    # Part 1: C_L (4x4) using Strassen x Strassen
    for k in range(7):
        for l in range(7):
            # U
            for bi in range(2):
                for bj in range(2):
                    if U_s[bi*2 + bj, k] != 0:
                        for ri in range(2):
                            for cj in range(2):
                                if U_s[ri*2 + cj, l] != 0:
                                    u_idx = (bi*2+ri)*4 + (bj*2+cj)
                                    U[u_idx, col] += U_s[bi*2 + bj, k] * U_s[ri*2 + cj, l]
                                    
            # V
            for bi in range(2):
                for bj in range(2):
                    if V_s[bi*2 + bj, k] != 0:
                        for ri in range(2):
                            for cj in range(2):
                                if V_s[ri*2 + cj, l] != 0:
                                    v_idx = (bi*2+ri)*6 + (bj*2+cj)
                                    V[v_idx, col] += V_s[bi*2 + bj, k] * V_s[ri*2 + cj, l]
                                    
            # W
            for bi in range(2):
                for bj in range(2):
                    if W_s[bi*2 + bj, k] != 0:
                        for ri in range(2):
                            for cj in range(2):
                                if W_s[ri*2 + cj, l] != 0:
                                    w_idx = (bi*2+ri)*6 + (bj*2+cj)
                                    W[w_idx, col] += W_s[bi*2 + bj, k] * W_s[ri*2 + cj, l]
                                    
            col += 1
            if time.time() - start_time > DEADLINE:
                break
        if time.time() - start_time > DEADLINE:
            break
            
    # Part 2: C_R (4x2) using Strassen x Schoolbook
    for k in range(7):
        for l in range(4):
            # U
            for bi in range(2):
                for bj in range(2):
                    if U_s[bi*2 + bj, k] != 0:
                        for ri in range(2):
                            for cj in range(2):
                                if U_sb[ri*2 + cj, l] != 0:
                                    u_idx = (bi*2+ri)*4 + (bj*2+cj)
                                    U[u_idx, col] += U_s[bi*2 + bj, k] * U_sb[ri*2 + cj, l]
                                    
            # V
            # B_R is 2x1 blocks.
            # V_s is 2x2. We use only first column of blocks?
            # No, V_s mixes B00, B01, B10, B11.
            # B_R has B02, B12.
            # So we need V_outer for 2x1?
            # We use V_s[:, k] but it expects 4 inputs.
            # B_R has 2 inputs (B02, B12).
            # So we set B01=0, B11=0.
            # So V_s[1, k] and V_s[3, k] should be 0?
            # No, V_s is fixed.
            # We must construct V_outer for 2x1 Strassen?
            # Rank 4.
            # If we use Rank 4 for Outer, total 16.
            # But we want 28.
            # So we use Rank 7 for Outer.
            # We map B02 -> B00, B12 -> B10. B01, B11 = 0.
            
            for bi in range(2):
                for bj in range(2):
                    if bj == 0: # Only first column of B_R blocks
                        val = V_s[bi*2 + 0, k]
                        if val != 0:
                            for ri in range(2):
                                if V_sb[ri, l] != 0:
                                    v_idx = (bi*2+ri)*6 + 2
                                    V[v_idx, col] += val * V_sb[ri, l]
                    # bj == 1 is 0
                    
            # W
            # C_R is 2x1 blocks.
            # W_s is 2x2.
            # We map C02, C12 -> C00, C10.
            
            for bi in range(2):
                for bj in range(2):
                    if bj == 0:
                        val = W_s[bi*2 + 0, k]
                        if val != 0:
                            for ri in range(2):
                                if W_sb[ri, l] != 0:
                                    w_idx = (bi*2+ri)*6 + 2
                                    W[w_idx, col] += val * W_sb[ri, l]
                                    
            col += 1
            if time.time() - start_time > DEADLINE:
                break
        if time.time() - start_time > DEADLINE:
            break

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
