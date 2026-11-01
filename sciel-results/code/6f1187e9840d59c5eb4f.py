import numpy as np
import time

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 145  # Safety margin before 150s kill
    
    # Build target tensor for verification
    target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target[a, b, c] = 1.0
    
    # Strassen rank-28 decomposition
    def strassen_decomposition():
        rank = 28
        U = np.zeros((n*m, rank))  # 8 x 28
        V = np.zeros((m*p, rank))  # 16 x 28
        W = np.zeros((p*n, rank))  # 8 x 28
        
        # Strassen 2x2x2 products:
        # m1 = (a11+a22)(b11+b22)
        # m2 = (a21+a22)(b11)
        # m3 = (a11)(b12-b22)
        # m4 = (a22)(b21-b11)
        # m5 = (a11+a12)(b22)
        # m6 = (a21-a11)(b11+b12)
        # m7 = (a12-a22)(b21+b22)
        # c11 = m1+m4-m5+m7
        # c12 = m3+m5
        # c21 = m2+m4
        # c22 = m1-m2+m3+m6
        
        # A-side coefficients for each Strassen product (for 2x2 block A[i, j0:j0+2]):
        # a11 = A[0, j0], a12 = A[0, j0+1], a21 = A[1, j0], a22 = A[1, j0+1]
        # U row index: i*4 + j
        a_side = [
            # s=0 (m1): a11+a22
            {0*4+0: 1, 1*4+1: 1},
            # s=1 (m2): a21+a22
            {1*4+0: 1, 1*4+1: 1},
            # s=2 (m3): a11
            {0*4+0: 1},
            # s=3 (m4): a22
            {1*4+1: 1},
            # s=4 (m5): a11+a12
            {0*4+0: 1, 0*4+1: 1},
            # s=5 (m6): a21-a11
            {1*4+0: 1, 0*4+0: -1},
            # s=6 (m7): a12-a22
            {0*4+1: 1, 1*4+1: -1},
        ]
        
        # B-side coefficients for each Strassen product (for 2x2 block B[j0:j0+2, k0:k0+2]):
        # b11 = B[j0, k0], b12 = B[j0, k0+1], b21 = B[j0+1, k0], b22 = B[j0+1, k0+1]
        # V row index: j*4 + k
        b_side = [
            # s=0 (m1): b11+b22
            {0*4+0: 1, 1*4+1: 1},
            # s=1 (m2): b11
            {0*4+0: 1},
            # s=2 (m3): b12-b22
            {0*4+1: 1, 1*4+1: -1},
            # s=3 (m4): b21-b11
            {1*4+0: 1, 0*4+0: -1},
            # s=4 (m5): b22
            {1*4+1: 1},
            # s=5 (m6): b11+b12
            {0*4+0: 1, 0*4+1: 1},
            # s=6 (m7): b21+b22
            {1*4+0: 1, 1*4+1: 1},
        ]
        
        # W-side coefficients for each Strassen product:
        # c11 = m1+m4-m5+m7 -> W[0*2+0] gets contributions from s=0,3,4,6
        # c12 = m3+m5 -> W[0*2+1] gets contributions from s=2,4
        # c21 = m2+m4 -> W[1*2+0] gets contributions from s=1,3
        # c22 = m1-m2+m3+m6 -> W[1*2+1] gets contributions from s=0,1,2,5
        # W row index: k*2 + i (trace convention: W[k*n+i])
        w_side = [
            # s=0 (m1): c11+=1, c22+=1
            {0*2+0: 1, 1*2+1: 1},
            # s=1 (m2): c21+=1, c22+=-1
            {1*2+0: 1, 1*2+1: -1},
            # s=2 (m3): c12+=1, c22+=1
            {0*2+1: 1, 1*2+1: 1},
            # s=3 (m4): c11+=1, c21+=1
            {0*2+0: 1, 1*2+0: 1},
            # s=4 (m5): c11+=-1, c12+=1
            {0*2+0: -1, 0*2+1: 1},
            # s=5 (m6): c22+=1
            {1*2+1: 1},
            # s=6 (m7): c11+=1
            {0*2+0: 1},
        ]
        
        col = 0
        for b_blk in range(2):  # B column block: 0 or 1
            for a_blk in range(2):  # A column block / B row block: 0 or 1
                j0 = 2 * a_blk  # Starting column of A block / starting row of B block
                k0 = 2 * b_blk  # Starting column of B block
                for s in range(7):
                    # Fill U
                    for u_idx, coeff in a_side[s].items():
                        U[u_idx, col] += coeff
                    # Fill V
                    for v_idx, coeff in b_side[s].items():
                        # V index: j*4 + k where j is row in B, k is col in B
                        # j0 + local_j, k0 + local_k
                        V[v_idx, col] += coeff
                    # Fill W
                    for w_idx, coeff in w_side[s].items():
                        # W index: k*n + i where k is output col, i is output row
                        # k0 + local_k, i (local row in 2x2)
                        W[w_idx, col] += coeff
                    col += 1
        
        return U, V, W, rank
    
    # Verify Strassen solution
    U, V, W, rank = strassen_decomposition()
    
    # Verify
    err = 0
    for a in range(n*m):
        for b in range(m*p):
            for c in range(p*n):
                val = np.sum(U[a, :] * V[b, :] * W[c, :])
                err += (val - target[a, b, c])**2
    
    if err < 1e-9:
        # Convert to lists and return
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback: schoolbook rank 32
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
