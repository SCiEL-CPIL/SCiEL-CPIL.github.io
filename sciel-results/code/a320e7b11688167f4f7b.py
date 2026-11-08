import time

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 10.0 # 10s safety margin
    
    n, m, p = DIMENSIONS
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # U_A[s, i, j], V_B[s, j, k], W_C[s, i, k]
    # s in 0..6, i,j,k in 0..1
    
    U_A = [
        [[1, 0], [1, 1]], # M1: A00+A11 -> idx (0,0)=1, (1,1)=1. Wait.
        # Let's re-verify the indices.
        # M1 = (A00+A11)(B00+B11)
        # U_A[0] should have 1 at (0,0) and (1,1).
        # My previous numpy code:
        # U_A[0, 0, 0] = 1; U_A[0, 1, 1] = 1
        # This means U_A[0][0][0]=1, U_A[0][1][1]=1.
        # Correct.
    ]
    
    # I will just hardcode the lists to avoid numpy dependency in the block if possible, 
    # but numpy is allowed. I'll stick to the logic derived.
    
    U_A = [
        [[1, 0], [1, 1]], # M1: (0,0)=1, (1,1)=1
        [[0, 0], [1, 1]], # M2: (1,0)=1, (1,1)=1
        [[1, 0], [0, 0]], # M3: (0,0)=1
        [[0, 0], [0, 1]], # M4: (1,1)=1
        [[1, 1], [0, 0]], # M5: (0,0)=1, (0,1)=1
        [[-1, 0], [1, 0]],# M6: (0,0)=-1, (1,0)=1
        [[0, 1], [-1, 0]] # M7: (0,1)=1, (1,1)=-1
    ]
    # Check M6: (A10 - A00). A10 is (1,0), A00 is (0,0).
    # U_A[5][1][0]=1, U_A[5][0][0]=-1.
    # My list: [[-1, 0], [1, 0]] -> [0][0]=-1, [1][0]=1. Correct.
    # Check M7: (A01 - A11). A01 is (0,1), A11 is (1,1).
    # U_A[6][0][1]=1, U_A[6][1][1]=-1.
    # My list: [[0, 1], [-1, 0]] -> [0][1]=1, [1][1]=-1. Correct.
    
    V_B = [
        [[1, 0], [0, 1]], # M1: (0,0)=1, (1,1)=1
        [[1, 0], [0, 0]], # M2: (0,0)=1
        [[0, 1], [-1, 0]],# M3: (0,1)=1, (1,1)=-1
        [[-1, 0], [1, 0]],# M4: (0,0)=-1, (1,0)=1
        [[0, 0], [0, 1]], # M5: (1,1)=1
        [[1, 1], [0, 0]], # M6: (0,0)=1, (0,1)=1
        [[0, 0], [1, 1]]  # M7: (1,0)=1, (1,1)=1
    ]
    # Check M3: (B01 - B11). B01 is (0,1), B11 is (1,1).
    # V_B[2][0][1]=1, V_B[2][1][1]=-1.
    # My list: [[0, 1], [-1, 0]] -> [0][1]=1, [1][1]=-1. Correct.
    # Check M4: (B10 - B00). B10 is (1,0), B00 is (0,0).
    # V_B[3][1][0]=1, V_B[3][0][0]=-1.
    # My list: [[-1, 0], [1, 0]] -> [0][0]=-1, [1][0]=1. Correct.
    # Check M6: (B00 + B01).
    # V_B[5][0][0]=1, V_B[5][0][1]=1.
    # My list: [[1, 1], [0, 0]]. Correct.
    # Check M7: (B10 + B11).
    # V_B[6][1][0]=1, V_B[6][1][1]=1.
    # My list: [[0, 0], [1, 1]]. Correct.
    
    W_C = [
        [[1, 0], [0, 0]], # M1 contributes to C00
        [[0, 0], [1, -1]],# M2 contributes to C10 (1) and C11 (-1)
        [[0, 1], [0, 1]], # M3 contributes to C01 (1) and C11 (1)
        [[1, 0], [1, 0]], # M4 contributes to C00 (1) and C10 (1)
        [[-1, 1], [0, 0]],# M5 contributes to C00 (-1) and C01 (1)
        [[0, 0], [0, 1]], # M6 contributes to C11 (1)
        [[1, 0], [0, 0]]  # M7 contributes to C00 (1)
    ]
    # Check C00 = M1 + M4 - M5 + M7
    # W_C[0][0][0]=1, W_C[3][0][0]=1, W_C[4][0][0]=-1, W_C[6][0][0]=1.
    # My lists:
    # W_C[0]: [[1, 0], ...] -> C00 gets 1.
    # W_C[3]: [[1, 0], ...] -> C00 gets 1.
    # W_C[4]: [[-1, 1], ...] -> C00 gets -1.
    # W_C[6]: [[1, 0], ...] -> C00 gets 1.
    # Sum: 1+1-1+1 = 2? 
    # Wait. C00 = M1 + M4 - M5 + M7.
    # M1 coeff 1. M4 coeff 1. M5 coeff -1. M7 coeff 1.
    # My W_C definition: W_C[s, i, k] is coeff of M_s in C_{i,k}.
    # So for C00 (i=0, k=0):
    # s=0 (M1): 1
    # s=3 (M4): 1
    # s=4 (M5): -1
    # s=6 (M7): 1
    # My W_C list:
    # W_C[0] = [[1, 0], [0, 0]] -> W_C[0][0][0]=1. OK.
    # W_C[3] = [[1, 0], [1, 0]] -> W_C[3][0][0]=1. OK.
    # W_C[4] = [[-1, 1], [0, 0]] -> W_C[4][0][0]=-1. OK.
    # W_C[6] = [[1, 0], [0, 0]] -> W_C[6][0][0]=1. OK.
    
    # Check C01 = M3 + M5
    # s=2 (M3): 1
    # s=4 (M5): 1
    # My W_C list:
    # W_C[2] = [[0, 1], [0, 1]] -> W_C[2][0][1]=1. OK.
    # W_C[4] = [[-1, 1], [0, 0]] -> W_C[4][0][1]=1. OK.
    
    # Check C10 = M2 + M4
    # s=1 (M2): 1
    # s=3 (M4): 1
    # My W_C list:
    # W_C[1] = [[0, 0], [1, -1]] -> W_C[1][1][0]=1. OK.
    # W_C[3] = [[1, 0], [1, 0]] -> W_C[3][1][0]=1. OK.
    
    # Check C11 = M1 - M2 + M3 + M6
    # s=0 (M1): 1
    # s=1 (M2): -1
    # s=2 (M3): 1
    # s=5 (M6): 1
    # My W_C list:
    # W_C[0] = [[1, 0], [0, 0]] -> W_C[0][1][1]=0. ERROR.
    # M1 contributes to C11? No, M1 is (A00+A11)(B00+B11).
    # C11 = A01B11 + A11B01? No.
    # C = AB.
    # C11 = A10B01 + A11B11.
    # Strassen:
    # C11 = M1 - M2 + M3 + M6
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M6 = (A10-A00)(B00+B01)
    
    # Let's re-verify the Strassen formulas.
    # C11 = M1 - M2 + M3 + M6
    # My W_C[0] for C11 (i=1, k=1) is 0.
    # But M1 should contribute to C11?
    # Let's check the formula source.
    # Standard Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # My indices: i=0 is row 1, i=1 is row 2. k=0 is col 1, k=1 is col 2.
    # C00 (C11) = M1 + M4 - M5 + M7
    # C01 (C12) = M3 + M5
    # C10 (C21) = M2 + M4
    # C11 (C22) = M1 - M2 + M3 + M6
    
    # So for C11 (i=1, k=1):
    # M1 coeff 1.
    # M2 coeff -1.
    # M3 coeff 1.
    # M6 coeff 1.
    
    # My W_C list:
    # W_C[0] (M1) should have 1 at (1,1).
    # My W_C[0] = [[1, 0], [0, 0]]. (1,1) is 0. ERROR.
    # I need to fix W_C.
    
    W_C = [
        [[1, 0], [1, 1]], # M1: C00=1, C11=1
        [[0, 0], [1, -1]],# M2: C10=1, C11=-1
        [[0, 1], [0, 1]], # M3: C01=1, C11=1
        [[1, 0], [1, 0]], # M4: C00=1, C10=1
        [[-1, 1], [0, 0]],# M5: C00=-1, C01=1
        [[0, 0], [0, 1]], # M6: C11=1
        [[1, 0], [0, 0]]  # M7: C00=1
    ]
    
    # Re-verify C00: M1(1) + M4(1) - M5(1) + M7(1) = 1+1-(-1)+1?
    # Formula: C00 = M1 + M4 - M5 + M7.
    # M5 coeff in formula is -1.
    # My W_C[4] has -1 at C00. So W_C[4][0][0] = -1.
    # Sum: 1 (M1) + 1 (M4) + (-1) (M5) + 1 (M7) = 2?
    # Wait. The formula is $C_{11} = M_1 + M_4 - M_5 + M_7$.
    # So the coefficient of M5 is -1.
    # In my W_C, I store the coefficient.
    # So W_C[4][0][0] should be -1.
    # My list W_C[4] = [[-1, 1], [0, 0]]. So W_C[4][0][0] = -1.
    # Sum: 1 + 1 + (-1) + 1 = 2.
    # Is C00 = 2*M1...? No.
    # Let's check the algebra.
    # M1 + M4 - M5 + M7
    # = (A00+A11)(B00+B11) + A11(B10-B00) - (A00+A01)B11 + (A01-A11)(B10+B11)
    # Terms in A00B00: 1 (M1) - 1 (M4) = 0?
    # M4 = A11(B10-B00). No A00B00.
    # M1 has A00B00.
    # M5 has -A00B11.
    # M7 has no A00B00.
    # So A00B00 coeff is 1.
    # Terms in A00B11: 1 (M1) - 1 (M5) = 0.
    # Terms in A11B00: 1 (M1) - 1 (M4) = 0.
    # Terms in A11B11: 1 (M1) + 1 (M7) = 2?
    # M7 = (A01-A11)(B10+B11) = A01B10 + A01B11 - A11B10 - A11B11.
    # So A11B11 coeff in M7 is -1.
    # M1 has A11B11 coeff 1.
    # Sum 1 + (-1) = 0.
    # So it works.
    
    for block_idx, j_offset in enumerate([0, 2]):
        for b_block in range(3):
            k_offset = b_block * 2
            for s in range(7):
                col = block_idx * 21 + b_block * 7 + s
                
                for i in range(2):
                    for j_sub in range(2):
                        val = U_A[s][i][j_sub]
                        if val != 0:
                            u_idx = i * m + (j_offset + j_sub)
                            U[u_idx][col] += val
                            
                for j_sub in range(2):
                    for k_sub in range(2):
                        val = V_B[s][j_sub][k_sub]
                        if val != 0:
                            v_idx = (j_offset + j_sub) * p + (k_offset + k_sub)
                            V[v_idx][col] += val
                            
                for i in range(2):
                    for k_sub in range(2):
                        val = W_C[s][i][k_sub]
                        if val != 0:
                            w_idx = (k_offset + k_sub) * n + i
                            W[w_idx][col] += val
                            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
