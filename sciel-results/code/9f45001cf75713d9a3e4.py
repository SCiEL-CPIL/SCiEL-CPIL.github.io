import numpy as np

def construct():
    n, m, p = 2, 2, 8
    rank = 28
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Strassen decomposition for 2x2 * 2x2
    # A = [[a, b], [c, d]]
    # B = [[x, y], [z, w]]
    # C = [[C00, C01], [C10, C11]]
    
    # M1 = (a+d)(x+w)
    # M2 = (c+d)x
    # M3 = a(y-w)
    # M4 = d(z-x)
    # M5 = (a+b)w
    # M6 = (c-a)(x+y)
    # M7 = (b-d)(z+w)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Indices for A: 0=a, 1=b, 2=c, 3=d
    # Indices for B: 0=x, 1=y, 2=z, 3=w
    # Indices for C: 0=C00, 1=C01, 2=C10, 3=C11
    
    A_forms = [
        [1, 0, 0, 1],  # M1: a+d
        [0, 0, 1, 1],  # M2: c+d
        [1, 0, 0, 0],  # M3: a
        [0, 0, 0, 1],  # M4: d
        [1, 1, 0, 0],  # M5: a+b
        [-1, 0, 1, 0], # M6: c-a (wait, c-a is -a+c. Indices: a=0, c=2. So -1 at 0, 1 at 2)
        [0, 1, 0, -1]  # M7: b-d (b=1, d=3. So 1 at 1, -1 at 3)
    ]
    
    B_forms = [
        [1, 0, 0, 1],  # M1: x+w
        [1, 0, 0, 0],  # M2: x
        [0, 1, 0, -1], # M3: y-w
        [0, 0, 1, -1], # M4: z-x
        [0, 0, 0, 1],  # M5: w
        [1, 1, 0, 0],  # M6: x+y
        [0, 0, 1, 1]   # M7: z+w
    ]
    
    C_coeffs = [
        [1, 0, 0, 0],  # M1 in C00
        [0, 0, 1, 0],  # M2 in C10
        [0, 1, 0, 0],  # M3 in C01
        [0, 0, 1, 0],  # M4 in C10 (Wait, M4 is in C00 and C10)
        # Let's map correctly.
        # C00: M1, M4, -M5, M7
        # C01: M3, M5
        # C10: M2, M4
        # C11: M1, -M2, M3, M6
    ]
    
    # Let's define W contributions per M
    # M1: C00(+1), C11(+1)
    # M2: C10(+1), C11(-1)
    # M3: C01(+1), C11(+1)
    # M4: C00(+1), C10(+1)
    # M5: C00(-1), C01(+1)
    # M6: C11(+1)
    # M7: C00(+1)
    
    W_contrib = [
        {0: 1, 3: 1},   # M1
        {2: 1, 3: -1},  # M2
        {1: 1, 3: 1},   # M3
        {0: 1, 2: 1},   # M4
        {0: -1, 1: 1},  # M5
        {3: 1},         # M6
        {0: 1}          # M7
    ]
    
    for block in range(4):
        for s in range(7):
            r = block * 7 + s
            
            # U
            for idx in range(4):
                U[idx, r] = A_forms[s][idx]
                
            # V
            # B indices for this block: 2*block to 2*block+1
            # Local B indices: 0=x, 1=y, 2=z, 3=w
            # Global B index: j*p + k
            # j is 0 or 1. k is 2*block + local_k
            # Local B form indices: 0,1,2,3 correspond to x,y,z,w
            # x is (0, 2b), y is (0, 2b+1)
            # z is (1, 2b), w is (1, 2b+1)
            
            for local_idx in range(4):
                val = B_forms[s][local_idx]
                if val != 0:
                    if local_idx == 0: # x
                        j, k_local = 0, 0
                    elif local_idx == 1: # y
                        j, k_local = 0, 1
                    elif local_idx == 2: # z
                        j, k_local = 1, 0
                    else: # w
                        j, k_local = 1, 1
                    
                    k_global = 2 * block + k_local
                    V[j * p + k_global, r] = val
            
            # W
            # C indices: 0=C00, 1=C01, 2=C10, 3=C11
            # Global C index: k*n + i
            # k is 2*block + local_k
            # i is 0 or 1
            # C00: i=0, k_local=0
            # C01: i=0, k_local=1
            # C10: i=1, k_local=0
            # C11: i=1, k_local=1
            
            for c_local_idx, coeff in W_contrib[s].items():
                i = c_local_idx // 2
                k_local = c_local_idx % 2
                k_global = 2 * block + k_local
                W[k_global * n + i, r] = coeff

    return {"U": U, "V": V, "W": W}
