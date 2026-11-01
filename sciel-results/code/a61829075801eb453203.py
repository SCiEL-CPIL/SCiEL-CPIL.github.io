import time

DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=10
    # Rank 35 using Strassen for 5 blocks of 2x2
    rank = 35
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen factors for A (i, j)
    # s=0..6
    # U_idx = i*m + j
    # i in 0..1, j in 0..1
    # U[0, s], U[1, s], U[2, s], U[3, s]
    
    # s=0: A00 + A11
    # s=1: A10 + A11
    # s=2: A00
    # s=3: A11
    # s=4: A00 + A01
    # s=5: A10 - A00
    # s=6: A01 - A11
    
    U_factors = [
        [1, 0, 0, 1], # s=0: A00+A11
        [0, 0, 1, 1], # s=1: A10+A11
        [1, 0, 0, 0], # s=2: A00
        [0, 0, 0, 1], # s=3: A11
        [1, 1, 0, 0], # s=4: A00+A01
        [-1, 0, 1, 0],# s=5: A10-A00
        [0, 1, 0, -1] # s=6: A01-A11
    ]
    
    for s in range(7):
        for i in range(2):
            for j in range(2):
                U[i*m + j][s] = U_factors[s][i*m + j]
                
    # Now replicate for 5 blocks
    # c = 7*t + s
    for t in range(5):
        for s in range(7):
            c = 7*t + s
            # U is same for all t
            for i in range(2):
                for j in range(2):
                    U[i*m + j][c] = U_factors[s][i*m + j]
                    
    # V factors for B (j, q) where k = 2t + q
    # V_idx = j*p + k = j*10 + 2t + q
    # s=0: B00 + B11 -> V[0, 2t]=1, V[1, 2t+1]=1
    # s=1: B00 -> V[0, 2t]=1
    # s=2: B10 - B11 -> V[1, 2t]=1, V[1, 2t+1]=-1
    # s=3: B00 - B01 -> V[0, 2t]=1, V[0, 2t+1]=-1
    # s=4: B11 -> V[1, 2t+1]=1
    # s=5: B01 + B11 -> V[0, 2t+1]=1, V[1, 2t+1]=1
    # s=6: B00 + B10 -> V[0, 2t]=1, V[1, 2t]=1
    
    V_factors = [
        [(0, 1), (1, 1)], # s=0: j=0,q=0; j=1,q=1
        [(0, 1)],         # s=1: j=0,q=0
        [(1, 1), (1, -1)],# s=2: j=1,q=0; j=1,q=1
        [(0, 1), (0, -1)],# s=3: j=0,q=0; j=0,q=1
        [(1, 1)],         # s=4: j=1,q=1
        [(0, 1), (1, 1)], # s=5: j=0,q=1; j=1,q=1
        [(0, 1), (1, 1)]  # s=6: j=0,q=0; j=1,q=0
    ]
    # Note: V_factors[s] is list of (j, q, coeff) but I wrote (j, coeff) implicitly?
    # Let's be explicit: (j, q, coeff)
    
    V_factors_explicit = [
        [(0, 0, 1), (1, 1, 1)], # s=0
        [(0, 0, 1)],            # s=1
        [(1, 0, 1), (1, 1, -1)],# s=2
        [(0, 0, 1), (0, 1, -1)],# s=3
        [(1, 1, 1)],            # s=4
        [(0, 1, 1), (1, 1, 1)], # s=5
        [(0, 0, 1), (1, 0, 1)]  # s=6
    ]
    
    for t in range(5):
        for s in range(7):
            c = 7*t + s
            for j, q, coeff in V_factors_explicit[s]:
                k = 2*t + q
                V[j*p + k][c] = coeff
                
    # W factors for C (i, q)
    # W_idx = k*n + i = (2t+q)*2 + i
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5
    
    # s=0 (M0): C00=1, C11=1
    # s=1 (M1): C10=1, C11=-1
    # s=2 (M2): C01=1, C11=1
    # s=3 (M3): C00=1, C10=1
    # s=4 (M4): C00=-1, C01=1
    # s=5 (M5): C11=1
    # s=6 (M6): C00=1
    
    W_factors_explicit = [
        [(0, 0, 1), (0, 1, 0), (1, 0, 0), (1, 1, 1)], # s=0: i=0,q=0; i=0,q=1; i=1,q=0; i=1,q=1
        [(0, 0, 0), (0, 1, 0), (1, 0, 1), (1, 1, -1)],# s=1
        [(0, 0, 0), (0, 1, 1), (1, 0, 0), (1, 1, 1)], # s=2
        [(0, 0, 1), (0, 1, 0), (1, 0, 1), (1, 1, 0)], # s=3
        [(0, 0, -1), (0, 1, 1), (1, 0, 0), (1, 1, 0)],# s=4
        [(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 1)], # s=5
        [(0, 0, 1), (0, 1, 0), (1, 0, 0), (1, 1, 0)]  # s=6
    ]
    
    for t in range(5):
        for s in range(7):
            c = 7*t + s
            for i, q, coeff in W_factors_explicit[s]:
                k = 2*t + q
                W[k*n + i][c] = coeff
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
