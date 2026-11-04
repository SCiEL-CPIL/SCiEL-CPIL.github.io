import time
import numpy as np

DIMENSIONS = [5, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = 5, 5, 5
    
    # Block sizes for partitioning 5 = 4 + 1
    block_sizes = [4, 1]
    
    # Block offsets
    offsets = [0]
    for s in block_sizes:
        offsets.append(offsets[-1] + s)
        
    # Strassen decomposition for 2x2x2
    def get_strassen_2x2_factors():
        # Returns list of 7 triples (u, v, w) where u, v, w are 2x2 matrices (flattened to 4 or kept as 2x2)
        # We keep them as 2x2 numpy arrays for clarity, then flatten
        factors = []
        # M1 = (A11+A22)(B11+B22)
        factors.append((np.array([[1,0],[1,1]]), np.array([[1,0],[0,1]]), np.array([[1,0],[0,1]]))) # Wait, W is for C reconstruction
        # Let's use the standard Strassen formulas directly as coefficients.
        # C11 = M1 + M4 - M5 - M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M3 + M6 - M7
        
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A21)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # u coefficients for A, v for B, w for C
        # Indexing: 0=A11, 1=A12, 2=A21, 3=A22
        
        u1 = np.array([[1, 0], [0, 1]]) # A11+A22
        v1 = np.array([[1, 0], [0, 1]]) # B11+B22
        w1 = np.array([[1, 0], [0, -1]]) # C11+=1, C22-=1
        
        u2 = np.array([[0, 0], [1, 1]]) # A21+A22
        v2 = np.array([[1, 0], [0, 0]]) # B11
        w2 = np.array([[0, 0], [1, 0]]) # C21+=1
        
        u3 = np.array([[1, 0], [0, 0]]) # A11
        v3 = np.array([[0, 1], [0, -1]]) # B12-B22
        w3 = np.array([[0, 1], [-1, 0]]) # C12+=1, C22-=1
        
        u4 = np.array([[0, 0], [0, 1]]) # A22
        v4 = np.array([[0, -1], [1, 0]]) # B21-B11
        w4 = np.array([[1, 0], [1, 0]]) # C11+=1, C21+=1
        
        u5 = np.array([[1, 0], [1, 0]]) # A11+A21
        v5 = np.array([[0, 0], [0, 1]]) # B22
        w5 = np.array([[0, 1], [1, 0]]) # C12+=1, C21+=1
        
        u6 = np.array([[0, 0], [-1, 1]]) # A21-A11
        v6 = np.array([[1, 1], [0, 0]]) # B11+B12
        w6 = np.array([[0, 0], [0, -1]]) # C22-=1
        
        u7 = np.array([[0, -1], [0, 1]]) # A12-A22
        v7 = np.array([[0, 0], [1, 1]]) # B21+B22
        w7 = np.array([[1, 0], [0, 1]]) # C11-=1, C22+=1
        
        return [(u1,v1,w1), (u2,v2,w2), (u3,v3,w3), (u4,v4,w4), (u5,v5,w5), (u6,v6,w6), (u7,v7,w7)]

    strassen_2x2 = get_strassen_2x2_factors()
    
    # Generate 4x4 factors
    def get_strassen_4x4_factors():
        factors_4x4 = []
        for k in range(7):
            uk, vk, wk = strassen_2x2[k]
            for s in range(7):
                us, vs, ws = strassen_2x2[s]
                
                # Coefficients for A, B, C are element-wise products
                uk_flat = uk.flatten()
                vk_flat = vk.flatten()
                wk_flat = wk.flatten()
                
                us_flat = us.flatten()
                vs_flat = vs.flatten()
                ws_flat = ws.flatten()
                
                u_ks = uk_flat * us_flat
                v_ks = vk_flat * vs_flat
                w_ks = wk_flat * ws_flat
                
                factors_4x4.append((u_ks, v_ks, w_ks))
        return factors_4x4

    strassen_4x4 = get_strassen_4x4_factors()
    
    total_rank = 0
    for i in range(2):
        for j in range(2):
            for k in range(2):
                ni = block_sizes[i]
                mj = block_sizes[j]
                pk = block_sizes[k]
                if ni == 4 and mj == 4 and pk == 4:
                    total_rank += 49
                else:
                    total_rank += ni * mj * pk

    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    col = 0
    for i in range(2):
        for j in range(2):
            for k in range(2):
                if time.time() > deadline:
                    pass
                
                ni = block_sizes[i]
                mj = block_sizes[j]
                pk = block_sizes[k]
                
                if ni == 4 and mj == 4 and pk == 4:
                    for (uf, vf, wf) in strassen_4x4:
                        # uf, vf, wf are length 16
                        for idx in range(16):
                            u_idx = (offsets[i] + idx // 4) * m + (offsets[j] + idx % 4)
                            v_idx = (offsets[j] + idx // 4) * p + (offsets[k] + idx % 4)
                            w_idx = (offsets[k] + idx // 4) * n + (offsets[i] + idx % 4)
                            
                            if uf[idx] != 0:
                                U[u_idx][col] += uf[idx]
                            if vf[idx] != 0:
                                V[v_idx][col] += vf[idx]
                            if wf[idx] != 0:
                                W[w_idx][col] += wf[idx]
                        col += 1
                else:
                    for li in range(ni):
                        for lj in range(mj):
                            for lk in range(pk):
                                U_idx = (offsets[i] + li) * m + (offsets[j] + lj)
                                V_idx = (offsets[j] + lj) * p + (offsets[k] + lk)
                                W_idx = (offsets[k] + lk) * n + (offsets[i] + li)
                                U[U_idx][col] = 1.0
                                V[V_idx][col] = 1.0
                                W[W_idx][col] = 1.0
                                col += 1
                                
    assert col == total_rank, f"Column mismatch: {col} != {total_rank}"
    
    return {"U": U, "V": V, "W": W}
