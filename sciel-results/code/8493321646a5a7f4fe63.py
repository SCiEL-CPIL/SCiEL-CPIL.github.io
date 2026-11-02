import numpy as np

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    num_blocks = p // 2 # 6 blocks of 2x2
    
    # Strassen decomposition for 2x2x2 (Rank 7)
    # U: (4, 7), V: (4, 7), W: (4, 7)
    # Indices for 2x2 matrix: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    U_s = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=float)
    
    V_s = np.array([
        [1, 1, 0, -1, 1, 0, 0],
        [0, 0, 1, 0, 0, 1, 1],
        [1, 0, 0, 1, 0, 0, 0],
        [0, 1, -1, 0, 0, 1, 0]
    ], dtype=float)
    
    W_s = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 1, 0, 0, 1, 0],
        [0, 0, 1, 1, 1, 0, 0],
        [1, -1, 0, 0, 0, 1, 0]
    ], dtype=float)
    
    # Verify Strassen correctness mentally or via code
    # A = [[a00, a01], [a10, a11]]
    # B = [[b00, b01], [b10, b11]]
    # M0 = (a00+a11)(b00+b11)
    # M1 = (a10+a11)b00
    # M2 = a00(b01-b11)
    # M3 = a11(b10-b00)
    # M4 = (a00+a01)b11
    # M5 = (a10-a00)(b00+b01)
    # M6 = (a01-a11)(b10+b11)
    # C00 = M0+M3-M4+M6
    # C01 = M2+M4
    # C10 = M1+M3
    # C11 = M0-M1+M2+M5
    
    # My U_s:
    # Row 0 (a00): 1, 0, 1, 0, 1, -1, 0 -> a00 in M0, M2, M4, M5(-1). Correct.
    # Row 1 (a01): 0, 0, 0, 0, 1, 0, 1 -> a01 in M4, M6. Correct.
    # Row 2 (a10): 0, 1, 0, 0, 0, 1, 0 -> a10 in M1, M5. Correct.
    # Row 3 (a11): 1, 1, 0, 1, 0, 0, -1 -> a11 in M0, M1, M3, M6(-1). Correct.
    
    # My V_s:
    # Row 0 (b00): 1, 1, 0, -1, 0, 1, 0 -> b00 in M0, M1, M3(-1), M5. 
    # Wait, Strassen M1 is (a10+a11)b00. So b00 coeff is 1.
    # M3 is a11(b10-b00). So b00 coeff is -1.
    # M5 is (a10-a00)(b00+b01). So b00 coeff is 1.
    # M0 is (a00+a11)(b00+b11). So b00 coeff is 1.
    # So V_s[0] should be [1, 1, 0, -1, 0, 1, 0].
    # My V_s[0] is [1, 1, 0, -1, 1, 0, 0]. 
    # M4 is (a00+a01)b11. b00 coeff is 0.
    # So V_s[0][4] should be 0.
    # Let's re-verify V_s.
    
    # M0: b00+b11 -> V[0]+=1, V[3]+=1
    # M1: b00 -> V[0]+=1
    # M2: b01-b11 -> V[1]+=1, V[3]-=1
    # M3: b10-b00 -> V[2]+=1, V[0]-=1
    # M4: b11 -> V[3]+=1
    # M5: b00+b01 -> V[0]+=1, V[1]+=1
    # M6: b10+b11 -> V[2]+=1, V[3]+=1
    
    # V_s rows:
    # b00 (idx 0): M0(1), M1(1), M3(-1), M5(1). Others 0.
    #   -> [1, 1, 0, -1, 0, 1, 0]
    # b01 (idx 1): M2(1), M5(1). Others 0.
    #   -> [0, 0, 1, 0, 0, 1, 0]
    # b10 (idx 2): M3(1), M6(1). Others 0.
    #   -> [0, 0, 0, 1, 0, 0, 1]
    # b11 (idx 3): M0(1), M2(-1), M4(1), M6(1). Others 0.
    #   -> [1, 0, -1, 0, 1, 0, 1]
    
    # Let's check W_s.
    # C00 (idx 0): M0(1), M3(1), M4(-1), M6(1).
    #   -> [1, 0, 0, 1, -1, 0, 1]
    # C01 (idx 1): M2(1), M4(1).
    #   -> [0, 0, 1, 0, 1, 0, 0]
    # C10 (idx 2): M1(1), M3(1).
    #   -> [0, 1, 0, 1, 0, 0, 0]
    # C11 (idx 3): M0(1), M1(-1), M2(1), M5(1).
    #   -> [1, -1, 1, 0, 0, 1, 0]
    
    # I will correct the arrays in the code.
    
    U_s_correct = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ])
    
    V_s_correct = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ])
    
    W_s_correct = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ])
    
    # Construct full U, V, W
    # U is (4, 42). 6 blocks of U_s.
    # V is (24, 42). 6 blocks of V_s?
    # B is 2x12. Rows 0,1 for block 0. Rows 2,3 for block 1?
    # No, B is 2x12.
    # Block u corresponds to columns 2u, 2u+1.
    # So V_s is applied to the sub-vector of B corresponding to columns 2u, 2u+1.
    # B is flattened to (24,).
    # Index for B row r, col c is r*12 + c.
    # For block u, cols are 2u, 2u+1.
    # So for row 0 of B, cols 2u, 2u+1 are at indices 0*12+2u, 0*12+2u+1.
    # For row 1 of B, cols 2u, 2u+1 are at indices 1*12+2u, 1*12+2u+1.
    # V_s is 4x7.
    # We need to place V_s into V (24x42) such that:
    # V[0*12+2u, u*7:t] = V_s[0, t]
    # V[0*12+2u+1, u*7+t] = V_s[1, t]
    # V[1*12+2u, u*7+t] = V_s[2, t]
    # V[1*12+2u+1, u*7+t] = V_s[3, t]
    
    # W is (24, 42).
    # C is 2x12. Flattened to (24,).
    # Index for C row i, col k is i*12 + k.
    # For block u, cols are 2u, 2u+1.
    # W_s is 4x7.
    # W[0*12+2u, u*7+t] = W_s[0, t]
    # W[0*12+2u+1, u*7+t] = W_s[1, t]
    # W[1*12+2u, u*7+t] = W_s[2, t]
    # W[1*12+2u+1, u*7+t] = W_s[3, t]
    
    rank = 42
    U = np.zeros((4, rank))
    V = np.zeros((24, rank))
    W = np.zeros((24, rank))
    
    for u in range(6):
        U[:, u*7:(u+1)*7] = U_s_correct
        # Fill V
        V[0*12 + 2*u, u*7:(u+1)*7] = V_s_correct[0]
        V[0*12 + 2*u+1, u*7:(u+1)*7] = V_s_correct[1]
        V[1*12 + 2*u, u*7:(u+1)*7] = V_s_correct[2]
        V[1*12 + 2*u+1, u*7:(u+1)*7] = V_s_correct[3]
        
        # Fill W
        W[0*12 + 2*u, u*7:(u+1)*7] = W_s_correct[0]
        W[0*12 + 2*u+1, u*7:(u+1)*7] = W_s_correct[1]
        W[1*12 + 2*u, u*7:(u+1)*7] = W_s_correct[2]
        W[1*12 + 2*u+1, u*7:(u+1)*7] = W_s_correct[3]
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
