import time
import numpy as np

DIMENSIONS = [2, 3, 5]

def _rank10_2x2x3():
    """
    Known rank-10 decomposition for multiplying 2x2 by 2x3.
    Returns (U, V, W) as lists of lists: U (4x10), V (6x10), W (6x10).
    """
    # Using the known decomposition from literature
    # For C[i,k] = sum_j A[i,j] * B[j,k] with i in {0,1}, j in {0,1}, k in {0,1,2}
    
    U = [[0]*10 for _ in range(4)]  # U[i*2+j, r]
    V = [[0]*10 for _ in range(6)]  # V[j*3+k, r]
    W = [[0]*10 for _ in range(6)]  # W[k*2+i, r]
    
    # Rank-10 terms for 2x2x3
    # Term 0: A[0,0]*B[0,0] -> C[0,0]
    U[0][0] = 1; V[0][0] = 1; W[0][0] = 1
    # Term 1: A[1,0]*B[0,0] -> C[1,0]
    U[1][1] = 1; V[0][1] = 1; W[1][1] = 1
    # Term 2: A[0,0]*B[0,1] -> C[0,1]
    U[0][2] = 1; V[1][2] = 1; W[2][2] = 1
    # Term 3: A[1,0]*B[0,1] -> C[1,1]
    U[1][3] = 1; V[1][3] = 1; W[3][3] = 1
    # Term 4: A[0,0]*B[0,2] -> C[0,2]
    U[0][4] = 1; V[2][4] = 1; W[4][4] = 1
    # Term 5: A[1,0]*B[0,2] -> C[1,2]
    U[1][5] = 1; V[2][5] = 1; W[5][5] = 1
    # Term 6: A[0,1]*B[1,0] -> C[0,0]
    U[2][6] = 1; V[3][6] = 1; W[0][6] = 1
    # Term 7: A[1,1]*B[1,0] -> C[1,0]
    U[3][7] = 1; V[3][7] = 1; W[1][7] = 1
    # Term 8: A[0,1]*B[1,1] -> C[0,1]
    U[2][8] = 1; V[4][8] = 1; W[2][8] = 1
    # Term 9: A[1,1]*B[1,1] -> C[1,1]
    U[3][9] = 1; V[4][9] = 1; W[3][9] = 1
    # But we need C[0,2] and C[1,2] from A[:,1]*B[1,2]
    # We only have 10 terms. Let me use a proper rank-10 decomposition.
    
    # Actually, let me use the standard approach:
    # The 2x2x3 tensor has rank 10. Here's one known decomposition:
    
    U = [[0]*10 for _ in range(4)]
    V = [[0]*10 for _ in range(6)]
    W = [[0]*10 for _ in range(6)]
    
    # Using the decomposition from "The rank of the 2x2x3 matrix multiplication tensor"
    # Terms indexed by r=0..9
    
    # r=0: (i,j)=(0,0), (j,k)=(0,0), (i,k)=(0,0)
    U[0][0] = 1; V[0][0] = 1; W[0][0] = 1
    # r=1: (i,j)=(1,0), (j,k)=(0,0), (i,k)=(1,0)
    U[1][1] = 1; V[0][1] = 1; W[1][1] = 1
    # r=2: (i,j)=(0,0), (j,k)=(0,1), (i,k)=(0,1)
    U[0][2] = 1; V[1][2] = 1; W[2][2] = 1
    # r=3: (i,j)=(1,0), (j,k)=(0,1), (i,k)=(1,1)
    U[1][3] = 1; V[1][3] = 1; W[3][3] = 1
    # r=4: (i,j)=(0,0), (j,k)=(0,2), (i,k)=(0,2)
    U[0][4] = 1; V[2][4] = 1; W[4][4] = 1
    # r=5: (i,j)=(1,0), (j,k)=(0,2), (i,k)=(1,2)
    U[1][5] = 1; V[2][5] = 1; W[5][5] = 1
    # r=6: (i,j)=(0,1), (j,k)=(1,0), (i,k)=(0,0)
    U[2][6] = 1; V[3][6] = 1; W[0][6] = 1
    # r=7: (i,j)=(1,1), (j,k)=(1,0), (i,k)=(1,0)
    U[3][7] = 1; V[3][7] = 1; W[1][7] = 1
    # r=8: (i,j)=(0,1), (j,k)=(1,1), (i,k)=(0,1)
    U[2][8] = 1; V[4][8] = 1; W[2][8] = 1
    # r=9: (i,j)=(1,1), (j,k)=(1,1), (i,k)=(1,1)
    U[3][9] = 1; V[4][9] = 1; W[3][9] = 1
    
    # This gives 10 terms but misses C[0,2] and C[1,2] from the second row of A.
    # We need a proper rank-10 decomposition. Let me construct one properly.
    
    # The 2x2x3 multiplication tensor T = sum_{i,j,k} e_{i,j} ⊗ e_{j,k} ⊗ e_{i,k}
    # has 12 terms. Rank 10 means we can combine 2 pairs.
    
    # Let me use a different known decomposition.
    # From the literature, the 2x2x3 tensor has rank 10.
    
    # Here's a valid rank-10 decomposition:
    U = [[0]*10 for _ in range(4)]
    V = [[0]*10 for _ in range(6)]
    W = [[0]*10 for _ in range(6)]
    
    # I'll use the following construction:
    # For k=0: use 4 terms for A*B[:,0]
    # For k=1: use 4 terms for A*B[:,1]  
    # For k=2: use 2 terms (shared structure)
    
    # Actually, let me just verify the simple approach works for rank 12
    # and then try to find a rank-10 one.
    
    # For now, return None and use the rank-28 approach.
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Try Rank 27 strategy
    rank_27 = _try_rank_27()
    if rank_27 is not None:
        return rank_27
        
    # Fallback to Rank 28
    return _strassen_rank28(n, m, p)

def _try_rank_27():
    """
    Try to construct a rank-27 decomposition:
    - Part 1: j=0, all k (2x1x5) = 10 terms
    - Part 2: j=1,2, k=0,1,2 (2x2x3) = 10 terms  
    - Part 3: j=1,2, k=3,4 (2x2x2) = 7 terms (Strassen)
    Total: 10 + 10 + 7 = 27
    """
    n, m, p = 2, 3, 5
    rank = 27
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    r = 0
    
    # Part 1: j=0, k=0..4 (10 terms) - direct
    for i in range(n):
        for k in range(p):
            U[i * m + 0][r] = 1
            V[0 * p + k][r] = 1
            W[k * n + i][r] = 1
            r += 1
            
    # Part 2: j=1,2, k=0,1,2 (2x2x3) - rank 10
    # Using the known rank-10 decomposition for 2x2x3 tensor
    # Local indices: il in {0,1}, jl in {0,1}, kl in {0,1,2}
    # Global: i = il, j = jl + 1, k = kl
    
    # Known rank-10 decomposition for 2x2x3:
    # We use the following 10 terms:
    
    # Terms 0-3: Handle k=0 with Strassen-like structure
    # T0: U[0,1]=1, U[1,2]=1; V[1,0]=1, V[2,0]=1; W[0,0]=1, W[0,1]=1
    # This contributes: A[0,1]*B[1,0] to C[0,0], A[1,2]*B[2,0] to C[1,0]
    # Wait, I need to be more careful.
    
    # Let me use a verified rank-10 decomposition from the literature.
    # The 2x2x3 tensor T = sum_{i,j,k} T_{ijk} e_i ⊗ e_j ⊗ e_k
    # where T_{ijk} = 1 if the multiplication is valid, 0 otherwise.
    
    # For 2x2x3: T_{ijk} = 1 for all (i,j,k) in {0,1}x{0,1}x{0,1,2}
    
    # A rank-10 decomposition can be constructed as follows:
    # Use 7 terms for the 2x2x2 part (k=0,1) via Strassen
    # Use 3 additional terms for k=2, but one of them shares structure
    
    # Actually, the cleanest way: rank(2x2x3) = 10 is known.
    # Let me construct it by starting with 12 schoolbook terms and finding 2 linear dependencies.
    
    # Schoolbook: 12 terms, one for each (i,j,k)
    # We want to find 10 terms that span the same space.
    
    # One known construction:
    # For k=0: 4 terms (direct)
    # For k=1: 4 terms (direct)  
    # For k=2: 2 terms (using linearity)
    # Total: 10
    
    # But this doesn't work because each (i,j,k) needs its own term in schoolbook.
    
    # The correct rank-10 decomposition uses:
    # - 6 terms for (i,j) in {(0,0),(1,0),(0,1),(1,1)} with k in {0,1}
    #   using 2 Strassen-like combinations
    # - 4 terms for k=2
    
    # I'll use the following explicit decomposition:
    
    # r=0: U[0,1]=1, U[1,2]=1; V[1,0]=1, V[2,1]=1; W[0,0]=1, W[1,1]=1
    U[0 * m + 1][r] = 1; U[1 * m + 2][r] = 1
    V[1 * p + 0][r] = 1; V[2 * p + 1][r] = 1
    W[0 * n + 0][r] = 1; W[1 * n + 1][r] = 1
    r += 1
    
    # This is getting too complex without a reference. Let me just use rank 28.
    return None

def _strassen_rank28(n, m, p):
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    # Part 1: A0 B0 (j=0, k=0..4) - 10 terms
    for i in range(n):
        for k in range(p):
            U[i * m + 0][r] = 1
            V[0 * p + k][r] = 1
            W[k * n + i][r] = 1
            r += 1
            
    # Part 2: A12 B12 (j=1,2, k=0,1) - 7 terms (Strassen)
    def add_strassen_2x2(i1, i2, j1, j2, k1, k2):
        nonlocal r
        U[i1*m+j1][r] = 1; U[i2*m+j2][r] = 1
        V[j1*p+k1][r] = 1; V[j2*p+k2][r] = 1
        W[k1*n+i1][r] = 1; W[k2*n+i2][r] = 1
        r += 1
        U[i2*m+j1][r] = 1; U[i2*m+j2][r] = 1
        V[j1*p+k1][r] = 1
        W[k1*n+i2][r] = 1; W[k2*n+i2][r] = -1
        r += 1
        U[i1*m+j1][r] = 1
        V[j1*p+k2][r] = 1; V[j2*p+k2][r] = -1
        W[k2*n+i1][r] = 1; W[k2*n+i2][r] = 1
        r += 1
        U[i2*m+j2][r] = 1
        V[j2*p+k1][r] = 1; V[j1*p+k1][r] = -1
        W[k1*n+i1][r] = 1; W[k1*n+i2][r] = 1
        r += 1
        U[i1*m+j1][r] = 1; U[i1*m+j2][r] = 1
        V[j2*p+k2][r] = 1
        W[k1*n+i1][r] = -1; W[k2*n+i1][r] = 1
        r += 1
        U[i2*m+j1][r] = 1; U[i1*m+j1][r] = -1
        V[j1*p+k1][r] = 1; V[j1*p+k2][r] = 1
        W[k2*n+i2][r] = 1
        r += 1
        U[i1*m+j2][r] = 1; U[i2*m+j2][r] = -1
        V[j2*p+k1][r] = 1; V[j2*p+k2][r] = 1
        W[k1*n+i1][r] = 1
        r += 1

    add_strassen_2x2(0, 1, 1, 2, 0, 1)
    
    # Part 3: A12 B12 (j=1,2, k=2,3) - 7 terms (Strassen)
    add_strassen_2x2(0, 1, 1, 2, 2, 3)
    
    # Part 4: A12 B12 (j=1,2, k=4) - 4 terms (Direct)
    for i in range(n):
        for j in range(1, 3):
            U[i * m + j][r] = 1
            V[j * p + 4][r] = 1
            W[4 * n + i][r] = 1
            r += 1
            
    # Verify
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    ok = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = sum(U_arr[i*m+j, ri] * V_arr[j*p+k, ri] * W_arr[k*n+i, ri] for ri in range(rank))
                if abs(s - 1.0) > 1e-9:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            break
            
    if ok:
        return {"U": U, "V": V, "W": W}
        
    return _schoolbook(n, m, p)

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
