import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Strassen decomposition for M_2,2,2 (rank 7)
    # U_S, V_S, W_S are 4x7
    U_S = np.array([
        [1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0]
    ])
    # This is identity, not Strassen. 
    # Strassen's 7 multiplications for 2x2:
    # m1 = (A11 + A22) * (B11 + B22)
    # m2 = (A21 + A22) * B11
    # m3 = A11 * (B12 - B22)
    # m4 = A22 * (B21 - B11)
    # m5 = (A11 + A12) * B22
    # m6 = (A21 - A11) * (B11 + B12)
    # m7 = (A12 - A22) * (B21 + B22)
    
    # U_S (4x7)
    U_S = np.array([
        [1, 0, 1, 0, 1, -1, 0], # A11
        [1, 0, 0, 0, 0, 0, -1], # A22? No, let's map carefully.
        [0, 1, 0, 0, 0, 1, 0], # A21
        [0, 1, -1, 1, 1, 0, 1]  # A12
    ])
    # Let's verify U_S rows correspond to A11, A22, A21, A12? 
    # Standard flattening: A11, A12, A21, A22.
    # Row 0: A11. Row 1: A12. Row 2: A21. Row 3: A22.
    # m1: A11+A22 -> U[0]=1, U[3]=1.
    # m2: A21+A22 -> U[2]=1, U[3]=1.
    # m3: A11 -> U[0]=1.
    # m4: A22 -> U[3]=1.
    # m5: A11+A12 -> U[0]=1, U[1]=1.
    # m6: A21-A11 -> U[2]=1, U[0]=-1.
    # m7: A12-A22 -> U[1]=1, U[3]=-1.
    
    U_S = np.array([
        [1, 0, 1, 0, 1, -1, 0], # A11 (row 0)
        [0, 0, 0, 0, 1, 0, 1],  # A12 (row 1)
        [0, 1, 0, 0, 0, 1, 0],  # A21 (row 2)
        [1, 1, 0, 1, 0, 0, -1]  # A22 (row 3)
    ])
    
    V_S = np.array([
        [1, 1, 0, -1, 0, 1, 1], # B11 (row 0)
        [0, 0, 1, 0, 0, 1, 0],  # B12 (row 1)
        [0, 0, 0, 1, 0, 0, 1],  # B21 (row 2)
        [1, 0, -1, 0, 1, 0, 1]  # B22 (row 3)
    ])
    # Check m1: B11+B22 -> V[0]=1, V[3]=1. OK.
    # m2: B11 -> V[0]=1. OK.
    # m3: B12-B22 -> V[1]=1, V[3]=-1. OK.
    # m4: B21-B11 -> V[2]=1, V[0]=-1. OK.
    # m5: B22 -> V[3]=1. OK.
    # m6: B11+B12 -> V[0]=1, V[1]=1. OK.
    # m7: B21+B22 -> V[2]=1, V[3]=1. OK.
    
    W_S = np.array([
        [1, 0, 0, 0, 1, 0, 0], # C11 (row 0)
        [0, 0, 1, 0, 0, 1, 0], # C12 (row 1)
        [0, 1, 0, 1, 0, 0, 0], # C21 (row 2)
        [1, 0, 0, 0, -1, 0, 0] # C22 (row 3)
    ])
    # Check m1: C11 += m1, C12 -= m1, C21 -= m1, C22 += m1.
    # W[0]=1, W[1]=-1, W[2]=-1, W[3]=1.
    # My W_S row 0 (C11) is [1,0,0,0,1,0,0]. This has 1 at m1 and m5.
    # Let's re-verify W_S for Strassen.
    # C11 = m1 - m4 + m5 + m7
    # C12 = m3 + m5
    # C21 = m2 + m4
    # C22 = m1 - m3 + m6 + m7
    
    W_S = np.array([
        [1, 0, 0, -1, 1, 0, 1], # C11
        [0, 0, 1, 0, 1, 0, 0],  # C12
        [0, 1, 0, 1, 0, 0, 0],  # C21
        [1, 0, -1, 0, 0, 1, 1]  # C22
    ])
    
    # M_1,2,2 decomposition (rank 4)
    # A is 1x2, B is 2x2, C is 1x2.
    # C0 = A0*B00 + A1*B10
    # C1 = A0*B01 + A1*B11
    # U_I (2x4): Rows A0, A1.
    U_I = np.array([
        [1, 0, 1, 0], # A0
        [0, 1, 0, 1]  # A1
    ])
    # V_I (4x4): Rows B00, B01, B10, B11.
    V_I = np.array([
        [1, 0, 0, 0], # B00
        [0, 1, 0, 0], # B01
        [0, 0, 1, 0], # B10
        [0, 0, 0, 1]  # B11
    ])
    # W_I (2x4): Rows C0, C1.
    W_I = np.array([
        [1, 0, 1, 0], # C0
        [0, 1, 0, 1]  # C1
    ])
    
    rank = 7 * 4
    
    U = np.zeros((8, rank), dtype=int)
    V = np.zeros((16, rank), dtype=int)
    W = np.zeros((8, rank), dtype=int)
    
    for cS in range(7):
        for cI in range(4):
            c = cS * 4 + cI
            for iS in range(2):
                for jS in range(2):
                    for iI in range(1): # iI is always 0
                        for jI in range(2):
                            # U row index
                            r_U = iS * 4 + jS * 2 + jI
                            U[r_U, c] = U_S[iS * 2 + jS, cS] * U_I[iI * 2 + jI, cI]
                            
                            # V row index
                            # Standard index: j * 4 + k
                            # j = jS * 2 + jI
                            # k = kS * 2 + kI
                            # We need to iterate kS, kI for V
                            pass
                            
    # Fill V properly
    for cS in range(7):
        for cI in range(4):
            c = cS * 4 + cI
            for jS in range(2):
                for kS in range(2):
                    for jI in range(2):
                        for kI in range(2):
                            r_V = (jS * 2 + jI) * 4 + (kS * 2 + kI)
                            V[r_V, c] = V_S[jS * 2 + kS, cS] * V_I[jI * 2 + kI, cI]
                            
    # Fill W properly
    for cS in range(7):
        for cI in range(4):
            c = cS * 4 + cI
            for kS in range(2):
                for iS in range(2):
                    for kI in range(2):
                        for iI in range(1): # iI is 0
                            r_W = iS * 4 + (kS * 2 + kI)
                            W[r_W, c] = W_S[kS * 2 + iS, cS] * W_I[kI, cI]
                            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
