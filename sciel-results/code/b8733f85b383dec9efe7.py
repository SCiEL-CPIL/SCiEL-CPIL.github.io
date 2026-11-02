import numpy as np
import time

def construct():
    deadline = time.time() + 140
    
    # Define rank-7 decomposition for 2x2x2
    # U, V, W are 4x7
    U222 = np.array([
        [1, 1, 0, 0, 1, 1, 0],
        [0, 0, 1, 1, 1, -1, 1],
        [0, 0, 1, -1, 1, 1, -1],
        [1, -1, 0, 0, 1, -1, 1]
    ], dtype=float)
    
    V222 = np.array([
        [1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 1, -1, 1],
        [1, -1, 1, -1, 1, 1, 1],
        [1, -1, 1, 1, 1, -1, 1]
    ], dtype=float)
    
    W222 = np.array([
        [1, -1, 1, -1, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1],
        [0, 0, 1, -1, 1, 1, 0],
        [1, -1, 1, 1, 1, -1, 1]
    ], dtype=float)
    
    # Verify 2x2x2 rank 7
    # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    # i,j,k in 0..1
    valid = True
    for i in range(2):
        for j in range(2):
            for k in range(2):
                s = 0
                for r in range(7):
                    s += U222[i*2+j, r] * V222[j*2+k, r] * W222[k*2+i, r]
                if abs(s - 1.0) > 1e-9:
                    valid = False
                    break
        if not valid: break
    if not valid:
        # Fallback to schoolbook 4x4x6
        n, m, p = 4, 4, 6
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i*m*p + j*p + k
                    U[i*m+j, r] = 1
                    V[j*p+k, r] = 1
                    W[k*n+i, r] = 1
        return {"U": U, "V": V, "W": W}

    # Define schoolbook rank-12 decomposition for 2x2x3
    n2, m2, p2 = 2, 2, 3
    rank223 = n2 * m2 * p2 # 12
    U223 = np.zeros((n2*m2, rank223))
    V223 = np.zeros((m2*p2, rank223))
    W223 = np.zeros((p2*n2, rank223))
    for i in range(n2):
        for j in range(m2):
            for k in range(p2):
                r = i*m2*p2 + j*p2 + k
                U223[i*m2+j, r] = 1
                V223[j*p2+k, r] = 1
                W223[k*n2+i, r] = 1

    # Construct 4x4x6 via Kronecker product
    # n=4=2*2, m=4=2*2, p=6=2*3
    # i = 2*i1 + i2
    # j = 2*j1 + j2
    # k = 3*k1 + k2  (since p2=3, p1=2)
    
    rank = 7 * 12
    U = np.zeros((4*4, rank))
    V = np.zeros((4*6, rank))
    W = np.zeros((6*4, rank))
    
    for i1 in range(2):
        for i2 in range(2):
            i = 2*i1 + i2
            for j1 in range(2):
                for j2 in range(2):
                    j = 2*j1 + j2
                    for k1 in range(2):
                        for k2 in range(3):
                            k = 3*k1 + k2
                            for r1 in range(7):
                                for r2 in range(12):
                                    r = r1*12 + r2
                                    val = U222[i1*2+j1, r1] * U223[i2*2+j2, r2]
                                    U[i*4+j, r] += val
                                    val_v = V222[j1*2+k1, r1] * V223[j2*3+k2, r2]
                                    V[j*6+k, r] += val_v
                                    val_w = W222[k1*2+i1, r1] * W223[k2*2+i2, r2]
                                    W[k*4+i, r] += val_w

    # Verify full decomposition
    # T[i,j,k] = 1
    for i in range(4):
        for j in range(4):
            for k in range(6):
                s = 0
                for r in range(rank):
                    s += U[i*4+j, r] * V[j*6+k, r] * W[k*4+i, r]
                if abs(s - 1.0) > 1e-9:
                    # Fallback
                    n, m, p = 4, 4, 6
                    rank_sb = n * m * p
                    U_sb = np.zeros((n*m, rank_sb))
                    V_sb = np.zeros((m*p, rank_sb))
                    W_sb = np.zeros((p*n, rank_sb))
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                r = i*m*p + j*p + k
                                U_sb[i*m+j, r] = 1
                                V_sb[j*p+k, r] = 1
                                W_sb[k*n+i, r] = 1
                    return {"U": U_sb, "V": V_sb, "W": W_sb}

    return {"U": U, "V": V, "W": W}
