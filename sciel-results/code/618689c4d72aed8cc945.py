import numpy as np
import time

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=7
    t0 = time.time()
    deadline = t0 + 130  # safety margin before 150s kill
    
    # Strassen-like rank-7 decomposition for 2x3 x 3x3
    # M1 = A[0,:] B[:,0], M2 = A[1,:] B[:,1], M3 = A[0,:] B[:,2]
    # M4 = (A[0,:]-A[1,:]) B[:,0], M5 = (A[0,:]+A[1,:]) B[:,1]
    # M6 = (A[0,:]+A[1,:]) B[:,2], M7 = (A[0,:]-A[1,:]) B[:,2]
    # C[0,0]=M1+M4, C[0,1]=M2, C[0,2]=M3+M6
    # C[1,0]=M1, C[1,1]=M2+M5, C[1,2]=M7
    
    # U vectors (4 entries: A[0,0], A[0,1], A[0,2], A[1,0], A[1,1], A[1,2])
    # Actually U has shape (n*m, rank) = (6, rank)
    # Index: i*m + j for row i, col j of A
    # A[0,0]=idx0, A[0,1]=idx1, A[0,2]=idx2, A[1,0]=idx3, A[1,1]=idx4, A[1,2]=idx5
    
    U_strassen = [
        [1, 0, 0],  # M1: A[0,:]
        [0, 0, 0],  # M2: A[1,:]
        [0, 1, 0],  # M3: A[0,:]
        [0, 0, 0],  # M4: (A[0,:]-A[1,:])
        [0, 0, 0],  # M5: (A[0,:]+A[1,:])
        [0, 0, 0],  # M6: (A[0,:]+A[1,:])
        [0, 0, 0],  # M7: (A[0,:]-A[1,:])
    ]
    # Wait, I need to be more careful. Let me redo.
    # U has shape (n*m, rank) = (6, 21)
    # For each of the 3 blocks, 7 rank-1 terms
    # U[idx, r] where idx = i*m + j (i in 0..1, j in 0..2)
    
    # For block b (b=0,1,2), the 7 terms:
    # Term 0: M1 = A[0,:] * B[:,0]  -> U: A[0,0]=1, A[0,1]=1, A[0,2]=1, A[1,*]=0
    # Term 1: M2 = A[1,:] * B[:,1]  -> U: A[1,0]=1, A[1,1]=1, A[1,2]=1, A[0,*]=0
    # Term 2: M3 = A[0,:] * B[:,2]  -> U: A[0,0]=1, A[0,1]=1, A[0,2]=1, A[1,*]=0
    # Term 3: M4 = (A[0,:]-A[1,:]) * B[:,0] -> U: A[0,0]=1,A[0,1]=1,A[0,2]=1, A[1,0]=-1,A[1,1]=-1,A[1,2]=-1
    # Term 4: M5 = (A[0,:]+A[1,:]) * B[:,1] -> U: A[0,0]=1,A[0,1]=1,A[0,2]=1, A[1,0]=1,A[1,1]=1,A[1,2]=1
    # Term 5: M6 = (A[0,:]+A[1,:]) * B[:,2] -> U: A[0,0]=1,A[0,1]=1,A[0,2]=1, A[1,0]=1,A[1,1]=1,A[1,2]=1
    # Term 6: M7 = (A[0,:]-A[1,:]) * B[:,2] -> U: A[0,0]=1,A[0,1]=1,A[0,2]=1, A[1,0]=-1,A[1,1]=-1,A[1,2]=-1
    
    # V has shape (m*p, rank) = (21, 21)
    # Index: j*p + k for row j, col k of B
    # For block b, columns are b*3, b*3+1, b*3+2
    # Term 0: B[:,0] -> V[j*3+b*3, r] = 1 for j=0,1,2
    # Term 1: B[:,1] -> V[j*3+b*3+1, r] = 1 for j=0,1,2
    # Term 2: B[:,2] -> V[j*3+b*3+2, r] = 1 for j=0,1,2
    # Term 3: B[:,0] -> same as term 0
    # Term 4: B[:,1] -> same as term 1
    # Term 5: B[:,2] -> same as term 2
    # Term 6: B[:,2] -> same as term 2
    
    # W has shape (p*n, rank) = (14, 21)
    # Index: k*n + i for col k, row i of C
    # For block b, output columns are b*3, b*3+1, b*3+2
    # Term 0 (M1): C[0, b*3] += M1, C[1, b*3] += M1
    #   W[(b*3)*n + 0, r] = 1, W[(b*3)*n + 1, r] = 1
    # Term 1 (M2): C[0, b*3+1] += M2, C[1, b*3+1] += M2
    #   W[(b*3+1)*n + 0, r] = 1, W[(b*3+1)*n + 1, r] = 1
    # Term 2 (M3): C[0, b*3+2] += M3
    #   W[(b*3+2)*n + 0, r] = 1
    # Term 3 (M4): C[0, b*3] += M4
    #   W[(b*3)*n + 0, r] = 1
    # Term 4 (M5): C[1, b*3+1] += M5
    #   W[(b*3+1)*n + 1, r] = 1
    # Term 5 (M6): C[0, b*3+2] += M6
    #   W[(b*3+2)*n + 0, r] = 1
    # Term 6 (M7): C[1, b*3+2] += M7
    #   W[(b*3+2)*n + 1, r] = 1
    
    rank = 21
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    for b in range(3):
        for t in range(7):
            r = b * 7 + t
            # U
            if t == 0:  # M1
                U[0, r] = 1; U[1, r] = 1; U[2, r] = 1
            elif t == 1:  # M2
                U[3, r] = 1; U[4, r] = 1; U[5, r] = 1
            elif t == 2:  # M3
                U[0, r] = 1; U[1, r] = 1; U[2, r] = 1
            elif t == 3:  # M4
                U[0, r] = 1; U[1, r] = 1; U[2, r] = 1
                U[3, r] = -1; U[4, r] = -1; U[5, r] = -1
            elif t == 4:  # M5
                U[0, r] = 1; U[1, r] = 1; U[2, r] = 1
                U[3, r] = 1; U[4, r] = 1; U[5, r] = 1
            elif t == 5:  # M6
                U[0, r] = 1; U[1, r] = 1; U[2, r] = 1
                U[3, r] = 1; U[4, r] = 1; U[5, r] = 1
            elif t == 6:  # M7
                U[0, r] = 1; U[1, r] = 1; U[2, r] = 1
                U[3, r] = -1; U[4, r] = -1; U[5, r] = -1
            
            # V
            if t == 0:  # B[:,0]
                for j in range(3):
                    V[j * p + b * 3, r] = 1
            elif t == 1:  # B[:,1]
                for j in range(3):
                    V[j * p + b * 3 + 1, r] = 1
            elif t == 2:  # B[:,2]
                for j in range(3):
                    V[j * p + b * 3 + 2, r] = 1
            elif t == 3:  # B[:,0]
                for j in range(3):
                    V[j * p + b * 3, r] = 1
            elif t == 4:  # B[:,1]
                for j in range(3):
                    V[j * p + b * 3 + 1, r] = 1
            elif t == 5:  # B[:,2]
                for j in range(3):
                    V[j * p + b * 3 + 2, r] = 1
            elif t == 6:  # B[:,2]
                for j in range(3):
                    V[j * p + b * 3 + 2, r] = 1
            
            # W
            if t == 0:  # M1: C[0, b*3] += M1, C[1, b*3] += M1
                W[(b * 3) * n + 0, r] = 1
                W[(b * 3) * n + 1, r] = 1
            elif t == 1:  # M2: C[0, b*3+1] += M2, C[1, b*3+1] += M2
                W[(b * 3 + 1) * n + 0, r] = 1
                W[(b * 3 + 1) * n + 1, r] = 1
            elif t == 2:  # M3: C[0, b*3+2] += M3
                W[(b * 3 + 2) * n + 0, r] = 1
            elif t == 3:  # M4: C[0, b*3] += M4
                W[(b * 3) * n + 0, r] = 1
            elif t == 4:  # M5: C[1, b*3+1] += M5
                W[(b * 3 + 1) * n + 1, r] = 1
            elif t == 5:  # M6: C[0, b*3+2] += M6
                W[(b * 3 + 2) * n + 0, r] = 1
            elif t == 6:  # M7: C[1, b*3+2] += M7
                W[(b * 3 + 2) * n + 1, r] = 1
    
    # Verify the decomposition
    # C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Let's check with random matrices
    np.random.seed(42)
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_expected = A_test @ B_test
    
    C_computed = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            s = 0.0
            for r in range(rank):
                for j in range(m):
                    s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
            C_computed[i, k] = s
    
    if not np.allclose(C_computed, C_expected, atol=1e-10):
        # Fall back to schoolbook
        rank_school = n * m * p
        U_school = np.zeros((n * m, rank_school), dtype=float)
        V_school = np.zeros((m * p, rank_school), dtype=float)
        W_school = np.zeros((p * n, rank_school), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i * m + j, col] = 1
                    V_school[j * p + k, col] = 1
                    W_school[k * n + i, col] = 1
                    col += 1
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
