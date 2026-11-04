"""Tensor decomposition of 5x5x5 matrix multiplication."""
import time
import random

def construct():
    t0 = time.time()
    tmax = t0 + 105.0
    
    n = m = p = 5
    U_cols = []
    V_cols = []
    W_cols = []
    
    # Partition 5 = 2+2+1
    # Blocks: (0,0,0) 2x2x2, (0,0,2) 2x2x1, (0,1,0) 2x1x2, (0,1,2) 2x1x1,
    #         (1,0,0) 1x2x2, (1,0,2) 1x2x1, (1,1,0) 1x1x2, (1,1,2) 1x1x1,
    #         (2,0,0) 2x2x2, (2,0,2) 2x2x1, (2,1,0) 2x1x2, (2,1,2) 2x1x1,
    #         (0,2,0) 2x2x2, (0,2,2) 2x2x1, (0,3,0) 2x1x2, (0,3,2) 2x1x1,
    # Wait, this is getting confusing. Let me use a cleaner approach.
    
    # Actually, let's use the 4+1 split with Strassen on 4x4x4.
    # 4x4x4 using recursive Strassen: rank 49.
    # Then handle the 4x4x1, 4x1x4, 1x4x4, 4x1x1, 1x4x1, 1x1x4, 1x1x1 blocks.
    
    # For 4x4x4, implement Strassen recursively.
    # First, implement Strassen for 2x2x2.
    
    def add_strassen_2x2x2(i0, j0, k0):
        """Add Strassen terms for a 2x2x2 block starting at (i0, j0, k0)."""
        # The 7 Strassen products for 2x2x2:
        # M1 = (A[i0,j0]+A[i0+1,j0+1]) * (B[j0,k0]+B[j0+1,k0+1])
        # M2 = (A[i0+1,j0]+A[i0+1,j0+1]) * B[j0,k0]
        # M3 = A[i0,j0] * (B[j0,k0+1]-B[j0+1,k0+1])
        # M4 = A[i0+1,j0+1] * (B[j0+1,k0]-B[j0,k0])
        # M5 = (A[i0,j0]+A[i0,j0+1]) * B[j0+1,k0+1]
        # M6 = (A[i0+1,j0]-A[i0,j0]) * (B[j0,k0]+B[j0,k0+1])
        # M7 = (A[i0,j0+1]-A[i0+1,j0+1]) * (B[j0+1,k0]+B[j0+1,k0+1])
        #
        # C[i0,k0]   = M1 + M4 - M5 + M7
        # C[i0,k0+1] = M3 + M5
        # C[i0+1,k0] = M2 + M4
        # C[i0+1,k0+1] = M1 - M2 + M3 + M6
        
        # Each Mi is a product of a linear combination of A entries and a linear combination of B entries.
        # In the tensor decomposition, each Mi contributes one term.
        # U and V have the coefficients of the linear combinations, W has the coefficients for C.
        
        # M1: A part: (i0,j0):1, (i0+1,j0+1):1; B part: (j0,k0):1, (j0+1,k0+1):1; C part: (i0,k0):1, (i0+1,k0+1):1
        U_cols.append({i0*m+j0: 1, (i0+1)*m+(j0+1): 1})
        V_cols.append({j0*p+k0: 1, (j0+1)*p+(k0+1): 1})
        W_cols.append({k0*n+i0: 1, (k0+1)*n+(i0+1): 1})
        
        # M2: A part: (i0+1,j0):1, (i0+1,j0+1):1; B part: (j0,k0):1; C part: (i0+1,k0):1
        U_cols.append({(i0+1)*m+j0: 1, (i0+1)*m+(j0+1): 1})
        V_cols.append({j0*p+k0: 1})
        W_cols.append({k0*n+(i0+1): 1})
        
        # M3: A part: (i0,j0):1; B part: (j0,k0+1):1, (j0+1,k0+1):-1; C part: (i0,k0+1):1
        U_cols.append({i0*m+j0: 1})
        V_cols.append({j0*p+(k0+1): 1, (j0+1)*p+(k0+1): -1})
        W_cols.append({(k0+1)*n+i0: 1})
        
        # M4: A part: (i0+1,j0+1):1; B part: (j0+1,k0):1, (j0,k0):-1; C part: (i0+1,k0):1, (i0,k0):1
        U_cols.append({(i0+1)*m+(j0+1): 1})
        V_cols.append({(j0+1)*p+k0: 1, j0*p+k0: -1})
        W_cols.append({k0*n+(i0+1): 1, k0*n+i0: 1})
        
        # M5: A part: (i0,j0):1, (i0,j0+1):1; B part: (j0+1,k0+1):1; C part: (i0,k0):-1, (i0,k0+1):1
        U_cols.append({i0*m+j0: 1, i0*m+(j0+1): 1})
        V_cols.append({(j0+1)*p+(k0+1): 1})
        W_cols.append({k0*n+i0: -1, (k0+1)*n+i0: 1})
        
        # M6: A part: (i0+1,j0):1, (i0,j0):-1; B part: (j0,k0):1, (j0,k0+1):1; C part: (i0+1,k0+1):1
        U_cols.append({(i0+1)*m+j0: 1, i0*m+j0: -1})
        V_cols.append({j0*p+k0: 1, j0*p+(k0+1): 1})
        W_cols.append({(k0+1)*n+(i0+1): 1})
        
        # M7: A part: (i0,j0+1):1, (i0+1,j0+1):-1; B part: (j0+1,k0):1, (j0+1,k0+1):1; C part: (i0,k0):1
        U_cols.append({i0*m+(j0+1): 1, (i0+1)*m+(j0+1): -1})
        V_cols.append({(j0+1)*p+k0: 1, (j0+1)*p+(k0+1): 1})
        W_cols.append({k0*n+i0: 1})
    
    def add_schoolbook_block(i0, i1, j0, j1, k0, k1):
        """Add schoolbook terms for a block A[i0:i1, j0:j1] * B[j0:j1, k0:k1] -> C[i0:i1, k0:k1]."""
        for i in range(i0, i1):
            for j in range(j0, j1):
                for k in range(k0, k1):
                    U_cols.append({i*m+j: 1})
                    V_cols.append({j*p+k: 1})
                    W_cols.append({k*n+i: 1})
    
    # 4+1 split:
    # 4x4x4 core: use recursive Strassen.
    # For 4x4x4, split into 2x2 blocks: 4 sub-blocks of 2x2x2.
    # But we need to apply Strassen at the 4x4 level, which means treating 2x2 blocks as elements.
    # This requires 7 "block products", each of which is a 2x2x2 matmul, each rank 7.
    # Total rank for 4x4x4: 7*7 = 49.
    
    # To implement this, I need to define the 7 Strassen products at the block level,
    # and for each, add the 7 Strassen terms for the 2x2x2 sub-blocks.
    
    # This is complex. Let me instead use the 2+2+1 partition with Strassen on 2x2x2 blocks.
    
    # Partition: i: 0-1, 2-3, 4; j: 0-1, 2-3, 4; k: 0-1, 2-3, 4
    # Blocks:
    # (0,0,0): 2x2x2 -> Strassen, rank 7
    # (0,0,1): 2x2x1 -> schoolbook, rank 4
    # (0,0,2): 2x2x1 -> schoolbook, rank 4  [wait, k=4 is size 1]
    
    # Let me define the partition clearly:
    # i-part: [0,1], [2,3], [4]
    # j-part: [0,1], [2,3], [4]
    # k-part: [0,1], [2,3], [4]
    
    # So 3x3 = 9 blocks.
    
    # Block (pi, pj, pk) where pi,pj,pk in {0,1,2}:
    # pi=0: i in 0-1, pi=1: i in 2-3, pi=2: i in 4
    # Similarly for j,k.
    
    for pi in range(3):
        for pj in range(3):
            for pk in range(3):
                i0 = [0, 2, 4][pi]
                i1 = [2, 4, 5][pi]
                j0 = [0, 2, 4][pj]
                j1 = [2, 4, 5][pj]
                k0 = [0, 2, 4][pk]
                k1 = [2, 4, 5][pk]
                
                si = i1 - i0
                sj = j1 - j0
                sk = k1 - k0
                
                if si == 2 and sj == 2 and sk == 2:
                    add_strassen_2x2x2(i0, j0, k0)
                else:
                    add_schoolbook_block(i0, i1, j0, j1, k0, k1)
    
    rank = len(U_cols)
    
    # Build dense matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r in range(rank):
        for idx, val in U_cols[r].items():
            U[idx][r] = val
        for idx, val in V_cols[r].items():
            V[idx][r] = val
        for idx, val in W_cols[r].items():
            W[idx][r] = val
    
    # Verify
    rng = random.Random(12345)
    for trial in range(200):
        if time.time() > tmax:
            break
        A = [[rng.randint(0, 5) for _ in range(m)] for _ in range(n)]
        B = [[rng.randint(0, 5) for _ in range(p)] for _ in range(m)]
        C_true = [[sum(A[i][j] * B[j][k] for j in range(m)) for k in range(p)] for i in range(n)]
        C_test = [[0] * p for _ in range(n)]
        for i in range(n):
            for k in range(p):
                s = 0
                for r in range(rank):
                    for j in range(m):
                        s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                C_test[i][k] = s
        if C_test != C_true:
            return _schoolbook()
    
    return {"U": U, "V": V, "W": W}


def _schoolbook():
    n, m, p = 5, 5, 5
    rk = n * m * p
    U = [[0] * rk for _ in range(n * m)]
    V = [[0] * rk for _ in range(m * p)]
    W = [[0] * rk for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
