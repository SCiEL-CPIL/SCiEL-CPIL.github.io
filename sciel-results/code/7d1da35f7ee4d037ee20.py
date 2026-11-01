import numpy as np

def construct():
    n, m, p = 2, 2, 12
    rank = 42
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Strassen coefficients for 7 products
    # M1: (A00+A11)(B00+B11)
    # M2: (A10+A11)B00
    # M3: A00(B01-B11)
    # M4: A11(B10-B00)
    # M5: (A00+A01)B11
    # M6: (A10-A00)(B00+B01)
    # M7: (A01-A11)(B10+B11)
    
    # U vectors (size 4)
    U_M1 = [1, 0, 0, 1]
    U_M2 = [0, 0, 1, 1]
    U_M3 = [1, 0, 0, 0]
    U_M4 = [0, 0, 0, 1]
    U_M5 = [1, 1, 0, 0]
    U_M6 = [-1, 0, 1, 0]
    U_M7 = [0, 1, 0, -1]
    
    # For each block b (0 to 5)
    for b in range(6):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # V indices
        # B00 is V[k0], B01 is V[k1]
        # B10 is V[12+k0], B11 is V[12+k1]
        
        # V vectors (size 24)
        # We only set 4 non-zero entries for each
        V_M1 = {k0: 1, 12+k1: 1}
        V_M2 = {k0: 1}
        V_M3 = {k1: 1, 12+k1: -1}
        V_M4 = {12+k0: 1, k0: -1}
        V_M5 = {12+k1: 1}
        V_M6 = {k0: 1, k1: 1}
        V_M7 = {12+k0: 1, 12+k1: 1}
        
        # W indices
        # C00 is W[2*k0], C01 is W[2*k1]
        # C10 is W[2*k0+1], C11 is W[2*k1+1]
        
        # W vectors (size 24)
        # Coeffs for M1..M7 in C00, C01, C10, C11
        # C00: M1+M4-M5+M7
        # C01: M3+M5
        # C10: M2+M4
        # C11: M1-M2+M3+M6
        
        W_M1 = {2*k0: 1, 2*k1+1: 1}
        W_M2 = {2*k0+1: 1, 2*k1+1: -1}
        W_M3 = {2*k1: 1, 2*k1+1: 1}
        W_M4 = {2*k0: 1, 2*k0+1: 1}
        W_M5 = {2*k0: -1, 2*k1: 1}
        W_M6 = {2*k1+1: 1}
        W_M7 = {2*k0: 1}
        
        U_vecs = [U_M1, U_M2, U_M3, U_M4, U_M5, U_M6, U_M7]
        V_vecs = [V_M1, V_M2, V_M3, V_M4, V_M5, V_M6, V_M7]
        W_vecs = [W_M1, W_M2, W_M3, W_M4, W_M5, W_M6, W_M7]
        
        for r in range(7):
            c = 7 * b + r
            # U
            for i, val in enumerate(U_vecs[r]):
                if val != 0:
                    U[i, c] = val
            # V
            for idx, val in V_vecs[r].items():
                V[idx, c] = val
            # W
            for idx, val in W_vecs[r].items():
                W[idx, c] = val
                
    return {"U": U, "V": V, "W": W}
