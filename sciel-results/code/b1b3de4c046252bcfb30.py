import time
import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 8
    deadline = time.time() + 140.0  # safety margin before 150s kill

    # Rank 7 decomposition for 2x2x2 (Strassen)
    # T[i,j,k] = delta_{i,k}
    # Terms are (u, v, w) where u,v,w are length 4 vectors (indices 0..3 for 2x2)
    # u[i,j], v[j,k], w[k,i]
    def strassen_2x2x2():
        terms = []
        # Standard Strassen formulas for 2x2
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B10)
        # M7 = (A01-A11)(B01+B11)
        
        # Indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
        
        # Term 1
        u = [1, 0, 0, 1] # A00+A11
        v = [1, 1, 1, 1] # B00+B11 (actually B00+B01+B10+B11? No. B00+B11 is [1,0,0,1]? No.
        # Let's be precise.
        # B is 2x2. B00=idx0, B01=idx1, B10=idx2, B11=idx3.
        # B00+B11 -> [1, 0, 0, 1]
        # But Strassen M1 is (A00+A11)(B00+B11).
        # Result C00 = M1 - M4 - M6 + M7? No.
        # C00 = M1 - M4 - M6 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M3 - M5 + M6
        
        # Let's list the 7 multiplications M1..M7 as rank-1 terms.
        # Each term contributes to C via w.
        
        # M1: A=(1,0,0,1), B=(1,0,0,1), C=(1,0,0,1) -> C00 += 1, C11 += 1? No.
        # The tensor term is u \otimes v \otimes w.
        # T[i,j,k] += u[i,j] v[j,k] w[k,i].
        
        # M1 = (A00+A11)(B00+B11)
        u1 = [1, 0, 0, 1]
        v1 = [1, 0, 0, 1]
        # C00 gets M1, C11 gets M1.
        # w1 must be such that w1[k,i] extracts M1 for C00 (k=0,i=0) and C11 (k=1,i=1).
        # So w1[0,0]=1, w1[1,1]=1.
        w1 = [1, 0, 0, 1]
        terms.append((u1, v1, w1))
        
        # M2 = (A10+A11)B00
        u2 = [0, 0, 1, 1]
        v2 = [1, 0, 0, 0]
        # C10 gets M2.
        w2 = [0, 0, 1, 0] # w[1,0] = 1
        terms.append((u2, v2, w2))
        
        # M3 = A00(B10-B11)
        u3 = [1, 0, 0, 0]
        v3 = [0, 0, 1, -1]
        # C01 gets M3.
        w3 = [0, 1, 0, 0] # w[0,1] = 1
        terms.append((u3, v3, w3))
        
        # M4 = A11(B01-B00)
        u4 = [0, 0, 0, 1]
        v4 = [-1, 1, 0, 0]
        # C00 gets -M4, C10 gets M4.
        w4 = [-1, 0, 1, 0] # w[0,0]=-1, w[1,0]=1
        terms.append((u4, v4, w4))
        
        # M5 = (A00+A01)B11
        u5 = [1, 1, 0, 0]
        v5 = [0, 0, 0, 1]
        # C01 gets M5, C11 gets -M5.
        w5 = [0, 1, 0, -1] # w[0,1]=1, w[1,1]=-1
        terms.append((u5, v5, w5))
        
        # M6 = (A10-A00)(B00+B10)
        u6 = [-1, 0, 1, 0]
        v6 = [1, 0, 1, 0]
        # C00 gets -M6, C11 gets M6.
        w6 = [-1, 0, 0, 1] # w[0,0]=-1, w[1,1]=1
        terms.append((u6, v6, w6))
        
        # M7 = (A01-A11)(B01+B11)
        u7 = [0, 1, 0, -1]
        v7 = [0, 1, 0, 1]
        # C00 gets M7, C11 gets -M7.
        w7 = [1, 0, 0, -1] # w[0,0]=1, w[1,1]=-1
        terms.append((u7, v7, w7))
        
        return terms

    strassen_terms = strassen_2x2x2()
    
    # Construct 4x4x4 terms (49 terms)
    # Kronecker product of the 7 terms with themselves.
    terms_444 = []
    for t1 in strassen_terms:
        u1, v1, w1 = t1
        for t2 in strassen_terms:
            u2, v2, w2 = t2
            # u is length 16, index (i1, i2) -> i1*2 + i2
            # v is length 16, index (j1, j2) -> j1*2 + j2
            # w is length 16, index (k1, k2) -> k1*2 + k2
            
            u = np.kron(u1, u2).tolist()
            v = np.kron(v1, v2).tolist()
            w = np.kron(w1, w2).tolist()
            terms_444.append((u, v, w))
            
    # Verify 4x4x4 decomposition
    def verify_444(terms):
        T = np.zeros((4, 4, 4), dtype=float)
        for (a, b, c) in terms:
            a4 = np.array(a).reshape(4, 4)
            b4 = np.array(b).reshape(4, 4)
            c4 = np.array(c).reshape(4, 4)
            # T[i,j,k] += a[i,j] b[j,k] c[k,i]
            for i in range(4):
                for j in range(4):
                    for k in range(4):
                        T[i, j, k] += a4[i, j] * b4[j, k] * c4[k, i]
        target = np.zeros((4, 4, 4), dtype=float)
        for i in range(4):
            for k in range(4):
                for j in range(4):
                    target[i, j, k] = 1.0 if i == k else 0.0
        return bool(np.allclose(T, target, atol=1e-9))

    if not verify_444(terms_444):
        # Fallback to schoolbook if verification fails (should not happen)
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

    # Construct 4x4x8 decomposition
    # Rank = 49 + 49 = 98
    rank = 98
    U = [[0] * rank for _ in range(n * m)]   # 16 x 98
    V = [[0] * rank for _ in range(m * p)]   # 32 x 98
    W = [[0] * rank for _ in range(p * n)]   # 32 x 98

    col = 0
    # Block 1: output k in 0..3
    for (u, v, w) in terms_444:
        # U: 16 entries
        for i in range(16):
            U[i][col] = u[i]
        
        # V: 32 entries (j in 0..3, k in 0..7)
        # v is length 16, index j*4 + k_local
        for j in range(4):
            for k_local in range(4):
                v_idx = j * 4 + k_local
                v_global_idx = j * 8 + k_local
                V[v_global_idx][col] = v[v_idx]
        
        # W: 32 entries (k in 0..7, i in 0..3)
        # w is length 16, index k_local*4 + i
        for k_local in range(4):
            for i in range(4):
                w_idx = k_local * 4 + i
                w_global_idx = k_local * 4 + i
                W[w_global_idx][col] = w[w_idx]
        
        col += 1

    # Block 2: output k in 4..7
    for (u, v, w) in terms_444:
        # U: same
        for i in range(16):
            U[i][col] = u[i]
        
        # V: 32 entries
        # v is length 16, index j*4 + k_local
        for j in range(4):
            for k_local in range(4):
                v_idx = j * 4 + k_local
                v_global_idx = j * 8 + (4 + k_local)
                V[v_global_idx][col] = v[v_idx]
        
        # W: 32 entries
        # w is length 16, index k_local*4 + i
        for k_local in range(4):
            for i in range(4):
                w_idx = k_local * 4 + i
                w_global_idx = (4 + k_local) * 4 + i
                W[w_global_idx][col] = w[w_idx]
        
        col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
