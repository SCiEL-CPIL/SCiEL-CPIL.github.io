import numpy as np

DIMENSIONS = [2, 2, 15]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=15
    # U shape: (4, rank)
    # V shape: (30, rank)
    # W shape: (30, rank)
    
    # Strassen coefficients for 2x2 multiplication
    # A = [[a, b], [c, d]]
    # B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # U coefficients (for A)
    # u[r][i][j]
    u_strassen = [
        [[1, 0], [0, 1]],   # M1: a+d
        [[0, 0], [1, 1]],   # M2: c+d
        [[1, 0], [0, 0]],   # M3: a
        [[0, 0], [0, 1]],   # M4: d
        [[1, 1], [0, 0]],   # M5: a+b
        [[-1, 0], [1, 0]],  # M6: c-a
        [[0, 1], [0, -1]],  # M7: b-d
    ]
    
    # V coefficients (for B)
    # v[r][j][k] where j is row of B (0,1), k is col of B (0,1)
    v_strassen = [
        [[1, 1], [1, 1]],   # M1: e+h (actually e+h is sum of all? No. (a+d)(e+h). B part is e+h.
                            # e is B[0,0], h is B[1,1]. So [1,0; 0,1] for B?
                            # Wait. M1 = (a+d)(e+h).
                            # A part: a+d. B part: e+h.
                            # e is B[0,0], h is B[1,1].
                            # So v[0] = [[1, 0], [0, 1]]?
                            # Let's re-verify Strassen.
                            # M1 = (a+d)(e+h)
                            # M2 = (c+d)e
                            # M3 = a(f-h)
                            # M4 = d(g-e)
                            # M5 = (a+b)h
                            # M6 = (c-a)(e+f)
                            # M7 = (b-d)(g+h)
                            
                            # C = [[M1+M4-M5+M7, M3+M5], [M2+M4, M1-M2+M3+M6]]
                            
                            # Let's check M1: (a+d)(e+h).
                            # A coeff: [[1,0],[0,1]].
                            # B coeff: [[1,0],[0,1]]. (e is 0,0; h is 1,1).
                            # So v[0] = [[1, 0], [0, 1]].
                            
                            # M2: (c+d)e.
                            # A coeff: [[0,0],[1,1]].
                            # B coeff: [[1,0],[0,0]]. (e is 0,0).
                            # v[1] = [[1, 0], [0, 0]].
                            
                            # M3: a(f-h).
                            # A coeff: [[1,0],[0,0]].
                            # B coeff: [[0,1],[0,-1]]. (f is 0,1; h is 1,1).
                            # v[2] = [[0, 1], [0, -1]].
                            
                            # M4: d(g-e).
                            # A coeff: [[0,0],[0,1]].
                            # B coeff: [[-1,0],[1,0]]. (g is 1,0; e is 0,0).
                            # v[3] = [[-1, 0], [1, 0]].
                            
                            # M5: (a+b)h.
                            # A coeff: [[1,1],[0,0]].
                            # B coeff: [[0,0],[0,1]]. (h is 1,1).
                            # v[4] = [[0, 0], [0, 1]].
                            
                            # M6: (c-a)(e+f).
                            # A coeff: [[-1,0],[1,0]].
                            # B coeff: [[1,1],[0,0]]. (e is 0,0; f is 0,1).
                            # v[5] = [[1, 1], [0, 0]].
                            
                            # M7: (b-d)(g+h).
                            # A coeff: [[0,1],[0,-1]].
                            # B coeff: [[0,1],[1,1]]. (g is 1,0; h is 1,1).
                            # v[6] = [[0, 1], [1, 1]].
                            
    ]
    
    # Corrected V coefficients
    v_strassen = [
        [[1, 0], [0, 1]],   # M1
        [[1, 0], [0, 0]],   # M2
        [[0, 1], [0, -1]],  # M3
        [[-1, 0], [1, 0]],  # M4
        [[0, 0], [0, 1]],   # M5
        [[1, 1], [0, 0]],   # M6
        [[0, 1], [1, 1]],   # M7
    ]
    
    # W coefficients (for C)
    # w[r][i][k] where i is row of C (0,1), k is col of C (0,1)
    # C = [[M1+M4-M5+M7, M3+M5], [M2+M4, M1-M2+M3+M6]]
    w_strassen = [
        [[1, 0], [0, 1]],   # M1: C[0,0]+=1, C[1,1]+=1
        [[0, 0], [1, 0]],   # M2: C[1,0]+=1
        [[0, 1], [1, 0]],   # M3: C[0,1]+=1, C[1,1]+=1
        [[1, 0], [1, 0]],   # M4: C[0,0]+=1, C[1,0]+=1
        [[-1, 1], [0, 0]],  # M5: C[0,0]-=1, C[0,1]+=1
        [[0, 0], [1, 1]],   # M6: C[1,0]+=1, C[1,1]+=1
        [[1, 0], [0, 0]],   # M7: C[0,0]+=1
    ]
    
    # Verify W
    # C[0,0] = M1 + M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6
    
    # w[0] (M1): [0,0]=1, [1,1]=1. Correct.
    # w[1] (M2): [1,0]=1. Correct.
    # w[2] (M3): [0,1]=1, [1,1]=1. Correct.
    # w[3] (M4): [0,0]=1, [1,0]=1. Correct.
    # w[4] (M5): [0,0]=-1, [0,1]=1. Correct.
    # w[5] (M6): [1,0]=1, [1,1]=1. Correct.
    # w[6] (M7): [0,0]=1. Correct.
    
    # Standard 2x2x1 multiplication
    # A is 2x2, B is 2x1. C is 2x1.
    # C[0,0] = A[0,0]B[0,0] + A[0,1]B[1,0]
    # C[1,0] = A[1,0]B[0,0] + A[1,1]B[1,0]
    # Rank 4.
    # Term 1: A[0,0]B[0,0] -> C[0,0]
    # Term 2: A[0,1]B[1,0] -> C[0,0]
    # Term 3: A[1,0]B[0,0] -> C[1,0]
    # Term 4: A[1,1]B[1,0] -> C[1,0]
    
    u_std = [
        [[1, 0], [0, 0]],
        [[0, 1], [0, 0]],
        [[0, 0], [1, 0]],
        [[0, 0], [0, 1]],
    ]
    
    v_std = [
        [[1, 0], [0, 0]],
        [[0, 0], [1, 0]],
        [[1, 0], [0, 0]],
        [[0, 0], [1, 0]],
    ]
    
    w_std = [
        [[1, 0], [0, 0]],
        [[1, 0], [0, 0]],
        [[0, 0], [1, 0]],
        [[0, 0], [1, 0]],
    ]
    # Note: w_std indices are [i][k]. k is always 0 for 2x1.
    # C[0,0] gets 1 from term 1 and 2.
    # C[1,0] gets 1 from term 3 and 4.
    
    total_rank = 0
    U = []
    V = []
    W = []
    
    # Initialize matrices
    # We will build them column by column (rank components)
    
    # 7 blocks of size 2
    for block_idx in range(7):
        k_start = block_idx * 2
        for r in range(7):
            # Create column for U, V, W
            u_col = [0] * (n * m)
            v_col = [0] * (m * p)
            w_col = [0] * (p * n)
            
            # Fill U
            for i in range(2):
                for j in range(2):
                    val = u_strassen[r][i][j]
                    if val != 0:
                        u_col[i * m + j] = val
            
            # Fill V
            for j in range(2):
                for k in range(2):
                    val = v_strassen[r][j][k]
                    if val != 0:
                        # V index: j * p + (k_start + k)
                        v_idx = j * p + (k_start + k)
                        v_col[v_idx] = val
            
            # Fill W
            for i in range(2):
                for k in range(2):
                    val = w_strassen[r][i][k]
                    if val != 0:
                        # W index: (k_start + k) * n + i
                        w_idx = (k_start + k) * n + i
                        w_col[w_idx] = val
            
            U.append(u_col)
            V.append(v_col)
            W.append(w_col)
            total_rank += 1
            
    # 1 block of size 1
    k_start = 14
    for r in range(4):
        u_col = [0] * (n * m)
        v_col = [0] * (m * p)
        w_col = [0] * (p * n)
        
        for i in range(2):
            for j in range(2):
                val = u_std[r][i][j]
                if val != 0:
                    u_col[i * m + j] = val
        
        for j in range(2):
            # k is always 0 in the local 2x1 block
            val = v_std[r][j][0]
            if val != 0:
                v_idx = j * p + k_start
                v_col[v_idx] = val
        
        for i in range(2):
            # k is always 0 in the local 2x1 block
            val = w_std[r][i][0]
            if val != 0:
                w_idx = k_start * n + i
                w_col[w_idx] = val
        
        U.append(u_col)
        V.append(v_col)
        W.append(w_col)
        total_rank += 1
        
    return {"U": U, "V": V, "W": W}
