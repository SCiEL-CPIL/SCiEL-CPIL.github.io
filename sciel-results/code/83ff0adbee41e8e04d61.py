import time

DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=9
    
    terms = []
    
    # Strassen blocks for k-pairs: (0,1), (2,3), (4,5), (6,7)
    for block_idx in range(4):
        k = 2 * block_idx
        kp1 = k + 1
        
        # M1 = (a+d)(e+h)
        terms.append(({0: 1, 3: 1}, {k: 1, 9+kp1: 1}, {k*2+0: 1, kp1*2+1: 1, k*2+1: -1, kp1*2+0: -1}))
        # M2 = (c+d)e
        terms.append(({2: 1, 3: 1}, {k: 1}, {k*2+1: 1, kp1*2+1: -1}))
        # M3 = a(f-h)
        terms.append(({0: 1}, {kp1: 1, 9+kp1: -1}, {kp1*2+0: 1, kp1*2+1: 1}))
        # M4 = d(g-e)
        terms.append(({3: 1}, {9+k: 1, k: -1}, {k*2+0: 1, k*2+1: 1}))
        # M5 = (a+b)h
        terms.append(({0: 1, 1: 1}, {9+kp1: 1}, {k*2+0: -1, kp1*2+0: 1}))
        # M6 = (c-a)(e+f)
        terms.append(({2: 1, 0: -1}, {k: 1, kp1: 1}, {kp1*2+1: 1}))
        # M7 = (b-d)(g+h)
        terms.append(({1: 1, 3: -1}, {9+k: 1, 9+kp1: 1}, {k*2+0: 1}))
    
    # Schoolbook for k=8
    k = 8
    terms.append(({0: 1}, {k: 1}, {k*2+0: 1}))
    terms.append(({1: 1}, {9+k: 1}, {k*2+0: 1}))
    terms.append(({2: 1}, {k: 1}, {k*2+1: 1}))
    terms.append(({3: 1}, {9+k: 1}, {k*2+1: 1}))
    
    rank = len(terms)
    U = [[0] * rank for _ in range(4)]
    V = [[0] * rank for _ in range(18)]
    W = [[0] * rank for _ in range(18)]
    
    for t, (Ud, Vd, Wd) in enumerate(terms):
        for idx, val in Ud.items():
            U[idx][t] = val
        for idx, val in Vd.items():
            V[idx][t] = val
        for idx, val in Wd.items():
            W[idx][t] = val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
