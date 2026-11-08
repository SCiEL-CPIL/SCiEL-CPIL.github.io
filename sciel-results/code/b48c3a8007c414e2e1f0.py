def construct():
    n, m, p = DIMENSIONS  # 2, 4, 8
    # 8 blocks: jb in {0,1}, kb in {0,1,2,3}
    # Each block: Strassen rank 7
    # Total rank: 8 * 7 = 56
    
    num_blocks = 8  # 2 j-blocks * 4 k-blocks
    rank = num_blocks * 7
    
    U = np.zeros((n*m, rank))  # (8, 56)
    V = np.zeros((m*p, rank))  # (32, 56)
    W = np.zeros((p*n, rank))  # (16, 56)
    
    # Strassen decomposition for 2x2 matmul
    # U factors: 7 vectors in R^4 (indexed by (i, j_local))
    # V factors: 7 vectors in R^4 (indexed by (j_local, k_local))
    # W factors: 7 vectors in R^4 (indexed by (k_local, i))
    
    # U_strassen[s][i][j_local] for s in 0..6
    U_s = [
        [1,0,1,0],  # M1: A00+A11
        [0,0,1,1],  # M2: A10+A11
        [1,0,0,0],  # M3: A00
        [0,0,0,1],  # M4: A11
        [1,1,0,0],  # M5: A00+A01
        [-1,0,1,0], # M6: A10-A00
        [0,1,-1,0], # M7: A01-A11
    ]
    # Flattened: index = i*2 + j_local
    # U_s[s][i*2+jl]
    
    V_s = [
        [1,0,0,1],  # M1: B00+B11
        [1,0,0,0],  # M2: B00
        [0,1,-1,0], # M3: B10-B11
        [0,1,1,0],  # M4: B01-B10  -> wait, B01-B10: V[jl,kl], B01 is (0,1), B10 is (1,0)
        [0,0,0,1],  # M5: B11
        [1,0,1,0],  # M6: B00+B10
        [0,1,0,1],  # M7: B01+B11
    ]
    # Flattened: index = jl*2 + kl
    # V_s[s][jl*2+kl]
    
    # Let me re-verify V for M4: B01 - B10
    # B01 means j_local=0, k_local=1 -> index 0*2+1=1
    # B10 means j_local=1, k_local=0 -> index 1*2+0=2
    # So V[1]=1, V[2]=-1: [0,1,-1,0]
    # I had [0,1,1,0] which is wrong. Let me fix.
    
    V_s = [
        [1,0,0,1],  # M1: B00+B11
        [1,0,0,0],  # M2: B00
        [0,1,-1,0], # M3: B10-B11
        [0,1,-1,0], # M4: B01-B10 -> B01 at idx 1, B10 at idx 2: [0,1,-1,0]
        [0,0,0,1],  # M5: B11
        [1,0,1,0],  # M6: B00+B10
        [0,1,0,1],  # M7: B01+B11
    ]
    # Wait, M3: B10-B11: B10 at idx 2, B11 at idx 3: [0,0,1,-1]
    # Let me redo carefully.
    # B indices: (jl, kl), flat = jl*2+kl
    # B00 = (0,0) = idx 0
    # B01 = (0,1) = idx 1
    # B10 = (1,0) = idx 2
    # B11 = (1,1) = idx 3
    
    V_s = [
        [1,0,0,1],  # M1: B00+B11
        [1,0,0,0],  # M2: B00
        [0,0,1,-1], # M3: B10-B11
        [0,1,-1,0], # M4: B01-B10
        [0,0,0,1],  # M5: B11
        [1,0,1,0],  # M6: B00+B10
        [0,1,0,1],  # M7: B01+B11
    ]
    
    # W factors: W[kl, i], flat = kl*2+i
    # C00 = M1+M4-M5-M7: kl=0,i=0 -> idx 0
    # C01 = M3+M5: kl=0,i=1 -> idx 1
    # C10 = M2+M4: kl=1,i=0 -> idx 2
    # C11 = M1-M2+M3+M6: kl=1,i=1 -> idx 3
    
    W_s = [
        [1,0,0,0],  # M1: C00
        [0,0,1,0],  # M2: C10
        [0,1,0,0],  # M3: C01
        [1,0,1,0],  # M4: C00+C10
        [-1,1,0,0], # M5: -C00+C01
        [0,0,0,1],  # M6: C11
        [-1,0,0,0], # M7: -C00
    ]
    # Wait, let me redo:
    # W_s[s][kl*2+i]
    # M1: C00=+1 -> [1,0,0,0]
    # M2: C10=+1 -> [0,0,1,0]
    # M3: C01=+1 -> [0,1,0,0]
    # M4: C00=+1, C10=+1 -> [1,0,1,0]
    # M5: C00=-1, C01=+1 -> [-1,1,0,0]
    # M6: C11=+1 -> [0,0,0,1]
    # M7: C00=-1 -> [-1,0,0,0]
    
    W_s = [
        [1,0,0,0],  # M1
        [0,0,1,0],  # M2
        [0,1,0,0],  # M3
        [1,0,1,0],  # M4
        [-1,1,0,0], # M5
        [0,0,0,1],  # M6
        [-1,0,0,0], # M7
    ]
    
    # Now fill in the global matrices
    for jb in range(2):  # j block
        for kb in range(4):  # k block
            for s in range(7):  # Strassen step
                col = (jb*4 + kb)*7 + s
                
                # U: index = i*m + j = i*4 + (2*jb + jl)
                for i in range(2):
                    for jl in range(2):
                        U[i*4 + 2*jb + jl, col] = U_s[s][i*2 + jl]
                
                # V: index = j*p + k = (2*jb + jl)*8 + (2*kb + kl)
                for jl in range(2):
                    for kl in range(2):
                        V[(2*jb + jl)*8 + 2*kb + kl, col] = V_s[s][jl*2 + kl]
                
                # W: index = k*n + i = (2*kb + kl)*2 + i
                for kl in range(2):
                    for i in range(2):
                        W[(2*kb + kl)*2 + i, col] = W_s[s][kl*2 + i]
    
    # Verify
    max_err = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for c in range(rank):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                max_err = max(max_err, abs(val - 1))
    
    if max_err > 1e-9:
        # Fallback to schoolbook
        rank_schoolbook = n * m * p
        U_s2 = np.zeros((n*m, rank_schoolbook))
        V_s2 = np.zeros((m*p, rank_schoolbook))
        W_s2 = np.zeros((p*n, rank_schoolbook))
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s2[i*m+j, column] = 1
                    V_s2[j*p+k, column] = 1
                    W_s2[k*n+i, column] = 1
                    column += 1
        return {"U": U_s2.tolist(), "V": V_s2.tolist(), "W": W_s2.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
