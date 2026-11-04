"""Program for rank-minimal tensor decomposition of 2x3x7 matrix multiplication."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    import time
    import numpy as np

    start_time = time.time()
    deadline = 110

    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n

    # Schoolbook baseline
    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Hybrid construction: Strassen for 2x2x2 blocks, schoolbook for rest
    def hybrid_strassen():
        # Rank calculation:
        # 3 blocks of 2x2x2 (k=0,1; 2,3; 4,5) using Strassen (rank 7 each)
        # 1 block of 2x2x1 (k=6, j=0,1) using schoolbook (rank 4)
        # 1 block of 2x1x7 (j=2, k=0..6) using schoolbook (rank 14)
        # Total: 7*3 + 4 + 14 = 21 + 18 = 39
        
        rank = 39
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        
        col = 0
        
        # Helper to add a Strassen block for i in {i0, i1}, j in {j0, j1}, k in {k0, k1}
        def add_strassen_block(i0, i1, j0, j1, k0, k1):
            nonlocal col
            # Local indices:
            # i_loc: 0, 1 correspond to i0, i1
            # j_loc: 0, 1 correspond to j0, j1
            # k_loc: 0, 1 correspond to k0, k1
            
            # Strassen terms:
            # M1 = (a+d)(e+h)
            # M2 = (c+d)e
            # M3 = a(f-h)
            # M4 = d(g-c)
            # M5 = (a+b)h
            # M6 = (c-d)(e+f)
            # M7 = (b-d)(g+h)
            
            # U entries (A matrix):
            # a = A[i0, j0], b = A[i0, j1], c = A[i1, j0], d = A[i1, j1]
            
            # Term 1: (a+d)(e+h)
            U[i0*m + j0, col] += 1
            U[i1*m + j1, col] += 1
            V[j0*p + k0, col] += 1
            V[j1*p + k1, col] += 1
            # W reconstruction for C[i0, k0], C[i0, k1], C[i1, k0], C[i1, k1]
            # C00 = M1 + M4 - M5 + M7
            # C01 = M3 + M5
            # C10 = M2 + M4
            # C11 = M1 - M2 + M3 + M6
            W[k0*n + i0, col] += 1   # M1 to C00
            W[k1*n + i0, col] += 0   # M1 not in C01? Wait.
            # Strassen reconstruction:
            # C00 = M1 + M4 - M5 + M7
            # C01 = M3 + M5
            # C10 = M2 + M4
            # C11 = M1 - M2 + M3 + M6
            
            # So M1 contributes to C00 and C11.
            W[k0*n + i0, col] += 1
            W[k1*n + i1, col] += 1
            col += 1
            
            # Term 2: (c+d)e
            U[i1*m + j0, col] += 1
            U[i1*m + j1, col] += 1
            V[j0*p + k0, col] += 1
            V[j1*p + k1, col] += 0
            # M2 contributes to C10 and C11 (with -1)
            W[k0*n + i1, col] += 1
            W[k1*n + i1, col] += -1
            col += 1
            
            # Term 3: a(f-h)
            U[i0*m + j0, col] += 1
            U[i0*m + j1, col] += 0
            V[j0*p + k0, col] += 0
            V[j1*p + k1, col] += 1
            # M3 contributes to C01 and C11
            W[k1*n + i0, col] += 1
            W[k1*n + i1, col] += 1
            col += 1
            
            # Term 4: d(g-c)
            U[i0*m + j0, col] += 0
            U[i1*m + j1, col] += 1
            V[j0*p + k0, col] += 1
            V[j1*p + k1, col] += -1
            # M4 contributes to C00 and C10
            W[k0*n + i0, col] += 1
            W[k0*n + i1, col] += 1
            col += 1
            
            # Term 5: (a+b)h
            U[i0*m + j0, col] += 1
            U[i0*m + j1, col] += 1
            V[j0*p + k0, col] += 0
            V[j1*p + k1, col] += 1
            # M5 contributes to C00 (-1) and C01
            W[k0*n + i0, col] += -1
            W[k1*n + i0, col] += 1
            col += 1
            
            # Term 6: (c-d)(e+f)
            U[i0*m + j0, col] += -1
            U[i1*m + j1, col] += 1
            V[j0*p + k0, col] += 1
            V[j1*p + k1, col] += 1
            # M6 contributes to C11
            W[k1*n + i1, col] += 1
            col += 1
            
            # Term 7: (b-d)(g+h)
            U[i0*m + j0, col] += 0
            U[i0*m + j1, col] += 1
            U[i1*m + j1, col] += -1
            V[j0*p + k0, col] += 1
            V[j1*p + k1, col] += 1
            # M7 contributes to C00
            W[k0*n + i0, col] += 1
            col += 1

        # Add Strassen blocks for k pairs (0,1), (2,3), (4,5) with j=0,1
        add_strassen_block(0, 1, 0, 1, 0, 1)
        add_strassen_block(0, 1, 0, 1, 2, 3)
        add_strassen_block(0, 1, 0, 1, 4, 5)
        
        # Schoolbook for remaining parts
        # Part 1: j=0,1 and k=6
        for i in range(2):
            for j in range(2):
                for k in [6]:
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
                    
        # Part 2: j=2 and k=0..6
        for i in range(2):
            for j in [2]:
                for k in range(7):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
                    
        if col != rank:
             return schoolbook()
             
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    result = hybrid_strassen()
    
    # Verify
    U = np.array(result["U"])
    V = np.array(result["V"])
    W = np.array(result["W"])
    
    # Check tensor equation
    # T[i,j,k] = 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target = 1.0
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - target) > 1e-9:
                    return schoolbook()
                    
    # Check zeros
    for row in range(nm):
        for col_idx in range(mp):
            for kidx in range(pn):
                # Check if target is 1
                i = row // m
                j = row % m
                k_from_col = col_idx % p
                j_from_col = col_idx // p
                i_from_kidx = kidx % n
                k_from_kidx = kidx // n
                
                if (j_from_col == j) and (k_from_kidx == k_from_col) and (i_from_kidx == i):
                    continue
                    
                val = np.sum(U[row, :] * V[col_idx, :] * W[kidx, :])
                if abs(val) > 1e-9:
                    return schoolbook()

    return result
# EVOLVE-BLOCK-END
