def construct():
    import time
    start = time.time()
    deadline = start + 120.0
    
    n, m, p = 2, 2, 8
    rank = 28
    
    # Initialize factor matrices
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # The tensor is a direct sum of 4 independent 2x2x2 multiplication tensors.
    # We use the Strassen algorithm (rank 7) for each of the 4 column pairs.
    # Column pairs: (0,1), (2,3), (4,5), (6,7)
    
    col = 0
    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # Strassen's 7 terms for 2x2 matrix multiplication
        # Let A = [[a, b], [c, d]], B = [[e, f], [g, h]]
        # m1 = (a+d)(e+h)
        # m2 = (c+d)e
        # m3 = a(f-h)
        # m4 = d(g-e)
        # m5 = (a+b)h
        # m6 = (c-a)(e+f)
        # m7 = (b-d)(g+h)
        # C = [[m1+m4-m5+m7, m3+m5], [m2+m4, m1-m2+m3+m6]]
        
        # Term 1: m1 = (A[0,0]+A[1,1]) * (B[0,k0]+B[1,k1])
        # U: A[0,0]=1, A[1,1]=1 -> U[0]=1, U[3]=1
        # V: B[0,k0]=1, B[1,k1]=1 -> V[k0]=1, V[p+k1]=1
        # W: C[0,k0]+=1, C[1,k1]+=1 -> W[k0*n+0]=1, W[k1*n+1]=1
        U[0][col] = 1.0
        U[3][col] = 1.0
        V[0*p + k0][col] = 1.0
        V[1*p + k1][col] = 1.0
        W[k0*n + 0][col] = 1.0
        W[k1*n + 1][col] = 1.0
        col += 1
        
        # Term 2: m2 = (A[1,0]+A[1,1]) * B[0,k0]
        # U: A[1,0]=1, A[1,1]=1 -> U[2]=1, U[3]=1
        # V: B[0,k0]=1 -> V[k0]=1
        # W: C[1,k0]+=1, C[1,k1]-=1 -> W[k0*n+1]=1, W[k1*n+1]=-1
        U[2][col] = 1.0
        U[3][col] = 1.0
        V[0*p + k0][col] = 1.0
        W[k0*n + 1][col] = 1.0
        W[k1*n + 1][col] = -1.0
        col += 1
        
        # Term 3: m3 = A[0,0] * (B[0,k1]-B[1,k1])
        # U: A[0,0]=1 -> U[0]=1
        # V: B[0,k1]=1, B[1,k1]=-1 -> V[k1]=1, V[p+k1]=-1
        # W: C[0,k1]+=1, C[1,k1]+=1 -> W[k1*n+0]=1, W[k1*n+1]=1
        U[0][col] = 1.0
        V[0*p + k1][col] = 1.0
        V[1*p + k1][col] = -1.0
        W[k1*n + 0][col] = 1.0
        W[k1*n + 1][col] = 1.0
        col += 1
        
        # Term 4: m4 = A[1,1] * (B[1,k0]-B[0,k0])
        # U: A[1,1]=1 -> U[3]=1
        # V: B[1,k0]=1, B[0,k0]=-1 -> V[p+k0]=1, V[k0]=-1
        # W: C[0,k0]+=1, C[1,k0]+=1 -> W[k0*n+0]=1, W[k0*n+1]=1
        U[3][col] = 1.0
        V[1*p + k0][col] = 1.0
        V[0*p + k0][col] = -1.0
        W[k0*n + 0][col] = 1.0
        W[k0*n + 1][col] = 1.0
        col += 1
        
        # Term 5: m5 = (A[0,0]+A[0,1]) * B[1,k1]
        # U: A[0,0]=1, A[0,1]=1 -> U[0]=1, U[1]=1
        # V: B[1,k1]=1 -> V[p+k1]=1
        # W: C[0,k0]-=1, C[0,k1]+=1 -> W[k0*n+0]=-1, W[k1*n+0]=1
        U[0][col] = 1.0
        U[1][col] = 1.0
        V[1*p + k1][col] = 1.0
        W[k0*n + 0][col] = -1.0
        W[k1*n + 0][col] = 1.0
        col += 1
        
        # Term 6: m6 = (A[1,0]-A[0,0]) * (B[0,k0]+B[0,k1])
        # U: A[1,0]=1, A[0,0]=-1 -> U[2]=1, U[0]=-1
        # V: B[0,k0]=1, B[0,k1]=1 -> V[k0]=1, V[k1]=1
        # W: C[1,k1]+=1 -> W[k1*n+1]=1
        U[2][col] = 1.0
        U[0][col] = -1.0
        V[0*p + k0][col] = 1.0
        V[0*p + k1][col] = 1.0
        W[k1*n + 1][col] = 1.0
        col += 1
        
        # Term 7: m7 = (A[0,1]-A[1,1]) * (B[1,k0]+B[1,k1])
        # U: A[0,1]=1, A[1,1]=-1 -> U[1]=1, U[3]=-1
        # V: B[1,k0]=1, B[1,k1]=1 -> V[p+k0]=1, V[p+k1]=1
        # W: C[0,k0]+=1 -> W[k0*n+0]=1
        U[1][col] = 1.0
        U[3][col] = -1.0
        V[1*p + k0][col] = 1.0
        V[1*p + k1][col] = 1.0
        W[k0*n + 0][col] = 1.0
        col += 1

    # Verification
    try:
        import numpy as np
        U_np = np.array(U)
        V_np = np.array(V)
        W_np = np.array(W)
        
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1.0
                    actual = 0.0
                    for c in range(rank):
                        actual += U_np[i*m+j, c] * V_np[j*p+k, c] * W_np[k*n+i, c]
                    if abs(actual - expected) > 1e-10:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if not valid:
            # Fallback to schoolbook if verification fails (should not happen)
            rank_fb = n * m * p
            U_fb = [[0.0] * rank_fb for _ in range(n * m)]
            V_fb = [[0.0] * rank_fb for _ in range(m * p)]
            W_fb = [[0.0] * rank_fb for _ in range(p * n)]
            c2 = 0
            for i2 in range(n):
                for j2 in range(m):
                    for k2 in range(p):
                        U_fb[i2*m+j2][c2] = 1.0
                        V_fb[j2*p+k2][c2] = 1.0
                        W_fb[k2*n+i2][c2] = 1.0
                        c2 += 1
            return {"U": U_fb, "V": V_fb, "W": W_fb}
    except Exception:
        pass

    if time.time() > deadline - 5.0:
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
