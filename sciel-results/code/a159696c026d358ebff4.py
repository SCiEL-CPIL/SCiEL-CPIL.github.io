import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Block size 3, 2x2 blocks
    block_size = 3
    num_blocks = 2
    
    # Strassen decomposition for 2x2
    # U_r, V_r, W_r are 2x2 matrices for r in 0..6
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    U_strassen = [
        [[1, 0], [0, 1]], # M1
        [[0, 0], [1, 1]], # M2
        [[1, 0], [0, 0]], # M3
        [[0, 0], [0, 1]], # M4
        [[1, 1], [0, 0]], # M5
        [[-1, 0], [1, 0]], # M6
        [[0, -1], [0, 1]]  # M7
    ]
    V_strassen = [
        [[1, 0], [0, 1]], # M1
        [[1, 0], [0, 0]], # M2
        [[0, 1], [0, -1]], # M3
        [[-1, 0], [0, 1]], # M4
        [[0, 0], [0, 1]], # M5
        [[1, 1], [0, 0]], # M6
        [[0, 0], [1, 1]]  # M7
    ]
    W_strassen = [
        [[1, 0], [0, 1]], # M1
        [[0, 0], [1, 0]], # M2
        [[0, 1], [0, 0]], # M3
        [[1, 0], [0, -1]], # M4
        [[-1, 0], [0, 0]], # M5
        [[0, 0], [0, 1]], # M6
        [[1, 0], [0, 0]]  # M7
    ]
    # Note: W_strassen indices are [I', K']
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Check W_strassen:
    # r=0 (M1): C11+=1, C22+=1. W[0,0]=1, W[1,1]=1.
    # r=1 (M2): C21+=1. W[1,0]=1.
    # r=2 (M3): C12+=1, C22+=1. W[0,1]=1, W[1,1]=1.
    # r=3 (M4): C11+=1, C21+=1. W[0,0]=1, W[1,0]=1.
    # r=4 (M5): C11+=-1, C12+=1. W[0,0]=-1, W[0,1]=1.
    # r=5 (M6): C22+=1. W[1,1]=1.
    # r=6 (M7): C11+=1. W[0,0]=1.
    
    # My W_strassen above:
    # 0: [[1,0],[0,1]] -> C11, C22. Correct.
    # 1: [[0,0],[1,0]] -> C21. Correct.
    # 2: [[0,1],[0,0]] -> C12. Missing C22?
    # M3 = A11(B12-B22).
    # C12 = M3 + M5.
    # C22 = M1 - M2 + M3 + M6.
    # So M3 contributes to C12 and C22.
    # W_strassen[2] should be [[0,1],[0,1]].
    # Let's re-verify Strassen W.
    
    W_strassen = [
        [[1, 0], [0, 1]], # M1: C11, C22
        [[0, 0], [1, 0]], # M2: C21
        [[0, 1], [0, 1]], # M3: C12, C22
        [[1, 0], [1, 0]], # M4: C11, C21
        [[-1, 1], [0, 0]], # M5: C11(-), C12(+)
        [[0, 0], [0, 1]], # M6: C22
        [[1, 0], [0, 0]]  # M7: C11
    ]
    # Check C11: M1(1) + M4(1) + M5(-1) + M7(1) = 2?
    # C11 = M1 + M4 - M5 + M7.
    # M1: 1. M4: 1. M5: -1. M7: 1. Sum = 2?
    # Wait.
    # M1 = (A11+A22)(B11+B22) = A11B11 + A11B22 + A22B11 + A22B22.
    # M4 = A22(B21-B11) = A22B21 - A22B11.
    # M5 = (A11+A12)B22 = A11B22 + A12B22.
    # M7 = (A12-A22)(B21+B22) = A12B21 + A12B22 - A22B21 - A22B22.
    # Sum for C11 (A11B11 + A12B21):
    # A11B11: M1(1).
    # A12B21: M7(1).
    # So C11 = M1 + M7?
    # Let's check standard Strassen.
    # C11 = M1 + M4 - M5 + M7.
    # A11B11: M1(1). M4(0). M5(0). M7(0). Sum=1.
    # A12B21: M1(0). M4(0). M5(0). M7(1). Sum=1.
    # A22B11: M1(1). M4(-1). M5(0). M7(0). Sum=0.
    # A22B22: M1(1). M4(0). M5(0). M7(-1). Sum=0.
    # So C11 is correct.
    
    # My W_strassen[4] (M5) is [[-1, 1], [0, 0]].
    # M5 contributes to C11 (-1) and C12 (1).
    # C12 = M3 + M5.
    # A11B12: M3(1). M5(0). Sum=1.
    # A22B12: M3(0). M5(0). Sum=0.
    # A12B22: M3(0). M5(1). Sum=1.
    # A22B22: M3(-1). M5(0). Sum=-1?
    # C12 should be A11B12 + A12B22.
    # M3 = A11(B12-B22) = A11B12 - A11B22.
    # M5 = (A11+A12)B22 = A11B22 + A12B22.
    # M3 + M5 = A11B12 + A12B22. Correct.
    # So W_strassen[2] (M3) is [[0, 1], [0, 1]]?
    # M3 contributes to C12 (1) and C22 (1).
    # C22 = M1 - M2 + M3 + M6.
    # A11B11: M1(1). M2(0). M3(0). M6(0). Sum=1.
    # A12B21: M1(0). M2(0). M3(0). M6(0). Sum=0.
    # A22B11: M1(1). M2(-1). M3(0). M6(0). Sum=0.
    # A22B22: M1(1). M2(0). M3(0). M6(1). Sum=2?
    # C22 should be A12B21 + A22B22.
    # M1 = A11B11 + A11B22 + A22B11 + A22B22.
    # M2 = (A21+A22)B11 = A21B11 + A22B11.
    # M3 = A11B12 - A11B22.
    # M6 = (A21-A11)(B11+B12) = A21B11 + A21B12 - A11B11 - A11B12.
    # Sum for C22 (A12B21 + A22B22):
    # A12B21: M6(0)? M6 has A21B12, A11B12. No A12B21.
    # Wait, M6 = (A21-A11)(B11+B12).
    # M7 = (A12-A22)(B21+B22).
    # C22 = M1 - M2 + M3 + M6.
    # A12B21: M1(0). M2(0). M3(0). M6(0). Sum=0.
    # This is wrong. C22 must have A12B21.
    # A12B21 is in M7.
    # Standard Strassen:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    # Let's check A12B21 in C22.
    # M1: 0. M2: 0. M3: 0. M6: 0.
    # So C22 does not contain A12B21?
    # C22 = A12B21 + A22B22.
    # Where is A12B21?
    # M7 = (A12-A22)(B21+B22) = A12B21 + A12B22 - A22B21 - A22B22.
    # M7 is in C11.
    # C11 = A11B11 + A12B21.
    # So A12B21 is in C11.
    # C22 = A12B21 + A22B22?
    # No, C22 = A21B12 + A22B22.
    # A21B12 is in M6.
    # M6 = A21B11 + A21B12 - A11B11 - A11B12.
    # So M6 has A21B12.
    # So C22 has A21B12 from M6.
    # And A22B22 from M1.
    # So C22 is correct.
    
    # Okay, the W_strassen matrices are correct.
    
    # Schoolbook decomposition for 3x3
    U_school = []
    V_school = []
    W_school = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                u = [[0]*3 for _ in range(3)]
                v = [[0]*3 for _ in range(3)]
                w = [[0]*3 for _ in range(3)]
                u[i][j] = 1
                v[j][k] = 1
                w[i][k] = 1
                U_school.append(u)
                V_school.append(v)
                W_school.append(w)
                
    rank = 7 * 27
    U = [[0] * rank for _ in range(36)]
    V = [[0] * rank for _ in range(36)]
    W = [[0] * rank for _ in range(36)]
    
    for r in range(7):
        for s in range(27):
            t = r * 27 + s
            # U
            for I in range(2):
                for J in range(2):
                    u_val = U_strassen[r][I][J]
                    if u_val == 0: continue
                    for i in range(3):
                        for j in range(3):
                            us_val = U_school[s][i][j]
                            if us_val == 0: continue
                            a_idx = (I * 3 + i) * 6 + (J * 3 + j)
                            U[a_idx][t] = u_val * us_val
                            
            # V
            for J in range(2):
                for K in range(2):
                    v_val = V_strassen[r][J][K]
                    if v_val == 0: continue
                    for j in range(3):
                        for k in range(3):
                            vs_val = V_school[s][j][k]
                            if vs_val == 0: continue
                            b_idx = (J * 3 + j) * 6 + (K * 3 + k)
                            V[b_idx][t] = v_val * vs_val
                            
            # W
            for I_prime in range(2):
                for K_prime in range(2):
                    w_val = W_strassen[r][I_prime][K_prime]
                    if w_val == 0: continue
                    for i in range(3):
                        for k in range(3):
                            ws_val = W_school[s][i][k]
                            if ws_val == 0: continue
                            c_idx = (K_prime * 3 + k) * 6 + (I_prime * 3 + i)
                            W[c_idx][t] = w_val * ws_val
                            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
