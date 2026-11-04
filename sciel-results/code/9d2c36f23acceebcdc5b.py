import numpy as np

def construct():
    n, m, p = 4, 5, 5
    rank = 92
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Helper to get indices
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i
    
    # Define Strassen blocks
    # Each block is (I, J, K) where I, J, K are lists of 2 global indices
    blocks = [
        ([0, 1], [0, 1], [0, 1]),
        ([0, 1], [0, 1], [2, 3]),
        ([0, 1], [2, 3], [0, 1]),
        ([0, 1], [2, 3], [2, 3]),
        ([2, 3], [0, 1], [0, 1]),
        ([2, 3], [0, 1], [2, 3]),
        ([2, 3], [2, 3], [0, 1]),
        ([2, 3], [2, 3], [2, 3]),
    ]
    
    covered = set()
    current_rank = 0
    
    # Strassen coefficients for A (U), B (V), C (W)
    # M1: (A00+A11)(B00+B11) -> C00:+1, C11:+1
    # M2: (A10+A11)B00 -> C10:+1, C11:-1  (Note: C21=M2+M4, C22=M1-M2+... so C21:+1, C22:-1)
    # Wait, let's map local (r,c) to global.
    # Local 0,0 -> Global I[0], J[0] etc.
    
    strassen_forms = [
        {
            'name': 'M1',
            'u': { (0,0): 1, (1,1): 1 },
            'v': { (0,0): 1, (1,1): 1 },
            'w': { (0,0): 1, (1,1): 1 } # C00 and C11
        },
        {
            'name': 'M2',
            'u': { (1,0): 1, (1,1): 1 },
            'v': { (0,0): 1 },
            'w': { (0,1): 1, (1,1): -1 } # C21 (local 1,0) and C22 (local 1,1)
        },
        {
            'name': 'M3',
            'u': { (0,0): 1 },
            'v': { (0,1): 1, (1,1): -1 },
            'w': { (1,0): 1, (1,1): 1 } # C12 (local 0,1) and C22 (local 1,1)
        },
        {
            'name': 'M4',
            'u': { (1,1): 1 },
            'v': { (1,0): 1, (0,0): -1 },
            'w': { (0,0): 1, (0,1): 1 } # C11 (local 0,0) and C21 (local 1,0)
        },
        {
            'name': 'M5',
            'u': { (0,0): 1, (0,1): 1 },
            'v': { (1,1): 1 },
            'w': { (0,0): -1, (1,0): 1 } # C11 (local 0,0) and C12 (local 0,1)
        },
        {
            'name': 'M6',
            'u': { (1,0): 1, (0,0): -1 },
            'v': { (0,0): 1, (0,1): 1 },
            'w': { (1,1): 1 } # C22 (local 1,1)
        },
        {
            'name': 'M7',
            'u': { (0,1): 1, (1,1): -1 },
            'v': { (1,0): 1, (1,1): 1 },
            'w': { (0,0): 1 } # C11 (local 0,0)
        }
    ]
    
    # Verify Strassen W coefficients against formulas:
    # C11 = M1 + M4 - M5 + M7
    #   M1: +1, M4: +1, M5: -1, M7: +1. Correct.
    # C12 = M3 + M5
    #   M3: +1, M5: +1. Correct.
    # C21 = M2 + M4
    #   M2: +1, M4: +1. Correct.
    # C22 = M1 - M2 + M3 + M6
    #   M1: +1, M2: -1, M3: +1, M6: +1. Correct.
    
    for I, J, K in blocks:
        for form in strassen_forms:
            if current_rank >= rank:
                break
                
            # Fill U
            for (li, lj), coeff in form['u'].items():
                gi = I[li]
                gj = J[lj]
                U[u_idx(gi, gj), current_rank] = coeff
            
            # Fill V
            for (lj, lk), coeff in form['v'].items():
                gj = J[lj]
                gk = K[lk]
                V[v_idx(gj, gk), current_rank] = coeff
            
            # Fill W
            for (lk, li), coeff in form['w'].items():
                gk = K[lk]
                gi = I[li]
                W[w_idx(gk, gi), current_rank] = coeff
                
            # Mark all (i,j,k) in this block as covered
            for li in range(2):
                for lj in range(2):
                    for lk in range(2):
                        covered.add((I[li], J[lj], K[lk]))
                        
            current_rank += 1
    
    # Fill remaining schoolbook terms
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    if current_rank >= rank:
                        break
                    U[u_idx(i, j), current_rank] = 1
                    V[v_idx(j, k), current_rank] = 1
                    W[w_idx(k, i), current_rank] = 1
                    current_rank += 1
            if current_rank >= rank:
                break
        if current_rank >= rank:
            break
    
    return {'U': U, 'V': V, 'W': W}
