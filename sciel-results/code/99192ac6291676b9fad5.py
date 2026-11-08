import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # We use block decomposition:
    # A (2x4) = [A1 (2x2) | A2 (2x2)]
    # B (4x6) = [B1 (2x6); B2 (2x6)]
    # C = A1 B1 + A2 B2
    # Each A_i B_i is 2x2 * 2x6.
    # 2x2 * 2x6 = 2x2 * (2x3 + 2x3) = 2x2*2x3 + 2x2*2x3
    # So we need 4 blocks of 2x2 * 2x3.
    
    # We will use rank 11 for 2x2 * 2x3 (7 Strassen + 4 direct)
    # Total rank = 4 * 11 = 44.
    
    # Helper to get U, V, W for 2x2 * 2x3
    def get_2x2x3():
        # n=2, m=2, p=3
        # U shape (4, 11), V shape (6, 11), W shape (6, 11)
        U = np.zeros((4, 11))
        V = np.zeros((6, 11))
        W = np.zeros((6, 11))
        
        # Strassen for first 2 cols (2x2x2)
        # M1 = (a+b)(e+h)
        # U: a=1, b=1 -> U[0]=1, U[1]=1
        # V: e=1, h=1 -> V[0]=1, V[3]=1 (B row 0 col 0, B row 1 col 0)
        # W: c00, c10? 
        # Strassen outputs:
        # c00 = M1 + M4 - M2 - M5
        # c01 = M3 + M5
        # c10 = M2 + M4
        # c11 = M1 - M3 + M5
        
        # Let's map Strassen terms to indices.
        # A indices: 00, 01, 10, 11 (a,b,c,d)
        # B indices: 00, 01, 10, 11 (e,f,h,i) for first 2 cols.
        # C indices: 00, 01, 10, 11 for first 2 cols.
        
        # Term 1: (a+b)(e+h)
        U[0, 0] = 1; U[1, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1  # c00
        W[3, 0] = 1  # c11? No, W index is k*n+i.
        # W shape (p*n, R) = (6, 11).
        # W index for c_{i,k} is k*n + i.
        # c00 (i=0, k=0) -> 0*2+0 = 0.
        # c01 (i=0, k=1) -> 1*2+0 = 2.
        # c10 (i=1, k=0) -> 0*2+1 = 1.
        # c11 (i=1, k=1) -> 1*2+1 = 3.
        
        # Strassen W contributions:
        # M1 contributes to c00 and c11?
        # c00 = M1 + ...
        # c11 = M1 + ...
        # So W[0, 0] = 1, W[3, 0] = 1.
        
        # Term 2: (c+d)(e+h)
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1; V[3, 1] = 1
        W[1, 1] = 1 # c10
        W[3, 1] = -1 # c11? c11 = M1 - M3 + M5. No M2 in c11.
        # c10 = M2 + M4. So W[1, 1] = 1.
        # c11 does not have M2.
        
        # Term 3: a(f-h)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[2, 2] = 1 # c01 = M3 + M5
        
        # Term 4: d(f+i)
        U[3, 3] = 1
        V[1, 3] = 1; V[3, 3] = 1
        W[1, 3] = 1 # c10 = M2 + M4
        
        # Term 5: (a+d)(e+i)
        U[0, 4] = 1; U[3, 4] = 1
        V[0, 4] = 1; V[3, 4] = 1
        W[0, 4] = -1 # c00
        W[2, 4] = 1  # c01
        W[3, 4] = 1  # c11
        
        # Now 4 direct terms for col 2 (g, j)
        # c02 = a g + b j
        # c12 = c g + d j
        
        # Term 6: a g
        U[0, 5] = 1
        V[2, 5] = 1 # B row 0 col 2 (g)
        W[4, 5] = 1 # c02 (i=0, k=2) -> 2*2+0 = 4
        
        # Term 7: b j
        U[1, 6] = 1
        V[5, 6] = 1 # B row 1 col 2 (j)
        W[4, 6] = 1
        
        # Term 8: c g
        U[2, 7] = 1
        V[2, 7] = 1
        W[5, 7] = 1 # c12 (i=1, k=2) -> 2*2+1 = 5
        
        # Term 9: d j
        U[3, 8] = 1
        V[5, 8] = 1
        W[5, 8] = 1
        
        return U, V, W

    # Assemble 2x4 * 4x6
    # 4 blocks of 2x2 * 2x3
    # Block 0: A1 (rows 0,1 of A? No, A is 2x4. Cols 0,1), B1 (rows 0,1 of B, cols 0,1,2)
    # Wait, B is 4x6.
    # A1 is A[:, 0:2] (2x2).
    # B1 is B[0:2, :] (2x6).
    # A2 is A[:, 2:4] (2x2).
    # B2 is B[2:4, :] (2x6).
    
    # We need to map the local indices of 2x2x3 to global indices.
    
    U_total = np.zeros((8, 44))
    V_total = np.zeros((24, 44))
    W_total = np.zeros((12, 44))
    
    u0, v0, w0 = get_2x2x3()
    
    # Block 0: A1, B1, C1 (cols 0-2 of C)
    # A1 indices: 0,1,2,3 (global A indices 0-3)
    # B1 indices: 0..5 (global B indices 0-5)
    # C1 indices: 0..5 (global C indices 0-5)
    
    U_total[0:4, 0:11] = u0
    V_total[0:6, 0:11] = v0
    W_total[0:6, 0:11] = w0
    
    # Block 1: A1, B1b (cols 3-5 of B)
    # A1 indices: 0,1,2,3
    # B1b indices: 6..11 (global B indices 6-11)
    # C1b indices: 6..11 (global C indices 6-11)
    
    U_total[0:4, 11:22] = u0
    V_total[6:12, 11:22] = v0
    W_total[6:12, 11:22] = w0
    
    # Block 2: A2, B2a (rows 2-3 of B, cols 0-2)
    # A2 indices: 4,5,6,7 (global A indices 4-7)
    # B2a indices: 12..17 (global B indices 12-17)
    # C2a indices: 0..5 (global C indices 0-5)
    
    U_total[4:8, 22:33] = u0
    V_total[12:18, 22:33] = v0
    W_total[0:6, 22:33] = w0
    
    # Block 3: A2, B2b (rows 2-3 of B, cols 3-5)
    # A2 indices: 4,5,6,7
    # B2b indices: 18..23 (global B indices 18-23)
    # C2b indices: 6..11 (global C indices 6-11)
    
    U_total[4:8, 33:44] = u0
    V_total[18:24, 33:44] = v0
    W_total[6:12, 33:44] = w0
    
    return {"U": U_total.tolist(), "V": V_total.tolist(), "W": W_total.tolist()}
