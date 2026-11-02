import numpy as np
import time

DIMENSIONS = [2, 2, 7]

def construct():
    start_time = time.time()
    deadline = 110.0 # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Schoolbook
    def schoolbook():
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

    # Strassen Block
    def strassen_block(k1, k2, cols_U, cols_V, cols_W):
        # Add 7 columns
        # P1 = (a+d)(e+h)
        u1 = [0]*(n*m); u1[0]=1; u1[3]=1
        v1 = [0]*(m*p); v1[k1]=1; v1[7+k2]=1
        w1 = [0]*(p*n); w1[2*k1]=1; w1[2*k2+1]=1
        cols_U.append(u1); cols_V.append(v1); cols_W.append(w1)
        
        # P2 = (c+d)e
        u2 = [0]*(n*m); u2[2]=1; u2[3]=1
        v2 = [0]*(m*p); v2[k1]=1
        w2 = [0]*(p*n); w2[2*k1+1]=1; w2[2*k2+1]=-1
        cols_U.append(u2); cols_V.append(v2); cols_W.append(w2)
        
        # P3 = a(f-h)
        u3 = [0]*(n*m); u3[0]=1
        v3 = [0]*(m*p); v3[k2]=1; v3[7+k1]=-1
        w3 = [0]*(p*n); w3[2*k2]=1; w3[2*k2+1]=1
        cols_U.append(u3); cols_V.append(v3); cols_W.append(w3)
        
        # P4 = d(g+h)
        u4 = [0]*(n*m); u4[3]=1
        v4 = [0]*(m*p); v4[7+k1]=1; v4[7+k2]=1
        w4 = [0]*(p*n); w4[2*k1]=1; w4[2*k1+1]=1
        cols_U.append(u4); cols_V.append(v4); cols_W.append(w4)
        
        # P5 = (a+b)h
        u5 = [0]*(n*m); u5[0]=1; u5[1]=1
        v5 = [0]*(m*p); v5[7+k2]=1
        w5 = [0]*(p*n); w5[2*k1]=-1; w5[2*k2]=1
        cols_U.append(u5); cols_V.append(v5); cols_W.append(w5)
        
        # P6 = (c-a)(e+f)
        u6 = [0]*(n*m); u6[2]=1; u6[0]=-1
        v6 = [0]*(m*p); v6[k1]=1; v6[k2]=1
        w6 = [0]*(p*n) # Not used in output?
        cols_U.append(u6); cols_V.append(v6); cols_W.append(w6)
        
        # P7 = (b-d)(g+h)
        u7 = [0]*(n*m); u7[1]=1; u7[3]=-1
        v7 = [0]*(m*p); v7[7+k1]=1; v7[7+k2]=1
        w7 = [0]*(p*n); w7[2*k1]=-1; w7[2*k2+1]=1
        cols_U.append(u7); cols_V.append(v7); cols_W.append(w7)

    # Schoolbook Block for 1 column
    def schoolbook_block(k, cols_U, cols_V, cols_W):
        # S1 = a e
        u1 = [0]*(n*m); u1[0]=1
        v1 = [0]*(m*p); v1[k]=1
        w1 = [0]*(p*n); w1[2*k]=1
        cols_U.append(u1); cols_V.append(v1); cols_W.append(w1)
        
        # S2 = b g
        u2 = [0]*(n*m); u2[1]=1
        v2 = [0]*(m*p); v2[7+k]=1
        w2 = [0]*(p*n); w2[2*k]=1
        cols_U.append(u2); cols_V.append(v2); cols_W.append(w2)
        
        # S3 = c e
        u3 = [0]*(n*m); u3[2]=1
        v3 = [0]*(m*p); v3[k]=1
        w3 = [0]*(p*n); w3[2*k+1]=1
        cols_U.append(u3); cols_V.append(v3); cols_W.append(w3)
        
        # S4 = d g
        u4 = [0]*(n*m); u4[3]=1
        v4 = [0]*(m*p); v4[7+k]=1
        w4 = [0]*(p*n); w4[2*k+1]=1
        cols_U.append(u4); cols_V.append(v4); cols_W.append(w4)

    # Build Rank 25
    cols_U = []
    cols_V = []
    cols_W = []
    
    strassen_block(0, 1, cols_U, cols_V, cols_W)
    strassen_block(2, 3, cols_U, cols_V, cols_W)
    strassen_block(4, 5, cols_U, cols_V, cols_W)
    schoolbook_block(6, cols_U, cols_V, cols_W)
    
    rank = len(cols_U)
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    for r in range(rank):
        for i in range(n*m):
            U[i][r] = cols_U[r][i]
        for j in range(m*p):
            V[j][r] = cols_V[r][j]
        for k in range(p*n):
            W[k][r] = cols_W[r][k]
            
    # Verify
    if time.time() < deadline:
        if not verify(U, V, W):
            return schoolbook()
            
    return {"U": U, "V": V, "W": W}

def verify(U, V, W):
    n, m, p = DIMENSIONS
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    
    # Check tensor
    for i in range(n):
        for k in range(p):
            target = 0
            for j in range(m):
                # C[i,k] = sum_j A[i,j] B[j,k]
                # Tensor coeff is 1 if i,j,k match structure?
                # No, the tensor is fixed.
                # We check if U V W reconstructs the identity mapping?
                # No, we check if the decomposition is exact.
                # The decomposition satisfies:
                # sum_r U[i,j,r] V[j,k,r] W[k,i,r] = delta?
                # No, it should equal the tensor T[i,j,k].
                # T[i,j,k] is 1 if the indices correspond to a valid multiplication term?
                # Actually, T[i,j,k] is the coefficient of A[i,j] B[j,k] in C[i,k].
                # So T[i,j,k] = 1 if the output index is (i,k) and input is (i,j) and (j,k).
                # This is always 1 for valid i,j,k.
                # So we check if sum_r U[i,j,r] V[j,k,r] W[k,i,r] == 1 for all i,j,k?
                # No, W index is k*n+i.
                # So sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] should be 1?
                # Yes, because C[i,k] = sum_j A[i,j] B[j,k].
                # The term A[i,j] B[j,k] appears in C[i,k] with coeff 1.
                # So the tensor T[i,j,k] is 1 for all i,j,k?
                # No, T is a 3-tensor.
                # T[i,j,k] is the coefficient of A[i,j] B[j,k] in C[i,k].
                # It is 1 if the indices match.
                # So for any i, j, k, the product A[i,j] B[j,k] contributes to C[i,k].
                # So T[i,j,k] = 1.
                
                val = 0
                for r in range(U.shape[1]):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if val != 1:
                    return False
    return True
