import numpy as np

def construct():
    n, m, p = 2, 2, 12
    rank = 42
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    # Strassen coefficients for 2x2 matrix multiplication
    # U indices: i*2+j (i=0,1; j=0,1)
    # V indices: j*2+k (j=0,1; k=0,1)
    # W indices: k*2+i (k=0,1; i=0,1)
    
    # The 7 Strassen terms
    # s=0: (A11+A22)(B11+B22) -> w00, w11
    # s=1: (A21+A22)B11 -> w10, w11
    # s=2: A11(B21-B22) -> w00, w01
    # s=3: A22(B12-B11) -> w10, w11
    # s=4: (A11+A12)B22 -> w00, w01
    # s=5: (A21-A11)(B11+B12) -> w10, w11
    # s=6: (A12-A22)(B21+B22) -> w00, w01
    
    # U vectors (for A)
    # Indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    U_0 = [1, 0, 0, 1] # A11+A22
    U_1 = [0, 0, 1, 1] # A21+A22
    U_2 = [1, 0, 0, 0] # A11
    U_3 = [0, 0, 0, 1] # A22
    U_4 = [1, 1, 0, 0] # A11+A12
    U_5 = [0, 0, 1, -1] # A21-A11
    U_6 = [0, 1, 0, -1] # A12-A22
    
    U_list = [U_0, U_1, U_2, U_3, U_4, U_5, U_6]
    
    # V vectors (for B)
    # Indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    V_0 = [1, 0, 0, 1] # B11+B22
    V_1 = [1, 0, 0, 0] # B11
    V_2 = [0, 1, 0, -1] # B21-B22
    V_3 = [-1, 1, 0, 0] # B12-B11  (Note: standard Strassen uses B12-B11)
    # Let's verify standard Strassen
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C11 = M1 + M3 - M5 + M7
    # C12 = M5 + M4
    # C21 = M2 + M6
    # C22 = M1 - M2 + M4 + M7
    
    # So:
    # s0: U=[1,0,0,1], V=[1,0,0,1], W=[1,0,0,-1] (C11=+1, C22=-1) -> Wait, W indices are k*2+i.
    # C11 is i=0, k=0 -> index 0. C22 is i=1, k=1 -> index 3.
    # C12 is i=0, k=1 -> index 1. C21 is i=1, k=0 -> index 2.
    
    # Let's map W for each term:
    # s0: M1. C11 += 1, C22 -= 1. W = [1, 0, 0, -1]
    # s1: M2. C21 += 1. W = [0, 0, 1, 0]
    # s2: M3. C11 += 1. W = [1, 0, 0, 0]
    # s3: M4. C12 += 1, C22 += 1. W = [0, 1, 0, 1]
    # s4: M5. C11 -= 1, C12 += 1. W = [-1, 1, 0, 0]
    # s5: M6. C21 += 1. W = [0, 0, 1, 0]
    # s6: M7. C11 += 1, C22 += 1. W = [1, 0, 0, 1]
    
    # Let's re-verify V and U for these M's
    # M1: (A11+A22)(B11+B22). U=[1,0,0,1], V=[1,0,0,1]. Correct.
    # M2: (A21+A22)B11. U=[0,0,1,1], V=[1,0,0,0]. Correct.
    # M3: A11(B21-B22). U=[1,0,0,0], V=[0,1,0,-1]. Correct.
    # M4: A22(B12-B11). U=[0,0,0,1], V=[-1,1,0,0]. Correct.
    # M5: (A11+A12)B22. U=[1,1,0,0], V=[0,0,0,1]. Correct.
    # M6: (A21-A11)(B11+B12). U=[0,0,1,-1], V=[1,1,0,0]. Correct.
    # M7: (A12-A22)(B21+B22). U=[0,1,0,-1], V=[0,0,1,1]. Correct.
    
    V_list = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    
    W_list = [
        [1, 0, 0, -1],
        [0, 0, 1, 0],
        [1, 0, 0, 0],
        [0, 1, 0, 1],
        [-1, 1, 0, 0],
        [0, 0, 1, 0],
        [1, 0, 0, 1]
    ]
    
    # Fill the matrices
    for b in range(6):
        for s in range(7):
            r = b * 7 + s
            # U is shared, but we put it in each slot to keep the factorization simple
            # Actually, U is the same for all b, so U[:, r] = U_list[s]
            U[:, r] = U_list[s]
            
            # V is specific to block b
            # Block b corresponds to columns 2b and 2b+1 of B
            # V indices: j*12 + k.
            # In the block, local k is 0,1. Global k is 2b, 2b+1.
            for j in range(2):
                for k_local in range(2):
                    k_global = 2 * b + k_local
                    val = V_list[s][j * 2 + k_local]
                    if val != 0:
                        V[j * 12 + k_global, r] = val
                        
            # W is specific to block b
            # W indices: k*n + i = k*2 + i.
            # In the block, local k is 0,1. Global k is 2b, 2b+1.
            for k_local in range(2):
                for i in range(2):
                    k_global = 2 * b + k_local
                    val = W_list[s][k_local * 2 + i]
                    if val != 0:
                        W[k_global * 2 + i, r] = val
                        
    return {"U": U, "V": V, "W": W}
