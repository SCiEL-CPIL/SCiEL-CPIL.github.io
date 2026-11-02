import numpy as np
import time

def _strassen_2x2():
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1; W[0,0]=1; W[3,0]=1
    U[2,1]=1; U[3,1]=1; V[0,1]=1; W[2,1]=1; W[3,1]=1
    U[0,2]=1; V[2,2]=1; V[3,2]=-1; W[1,2]=1; W[3,2]=1
    U[3,3]=1; V[1,3]=1; V[0,3]=-1; W[0,3]=1
    U[0,4]=1; U[2,4]=1; V[3,4]=1; W[0,4]=-1; W[2,4]=1
    U[1,5]=1; U[0,5]=-1; V[0,5]=1; V[2,5]=1; W[3,5]=1
    U[3,6]=1; U[1,6]=-1; V[1,6]=1; V[3,6]=1; W[0,6]=-1; W[1,6]=1
    return U, V, W

def _schoolbook_3x3():
    U = np.zeros((9, 9))
    V = np.zeros((9, 9))
    W = np.zeros((9, 9))
    r = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i*3+j, r] = 1
                V[j*3+k, r] = 1
                W[k*3+i, r] = 1
                r += 1
    return U, V, W

def _verify(n, m, p, U, V, W, tol=1e-9):
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                val = np.dot(U_arr[a,:]*V_arr[b,:], W_arr[c,:])
                if abs(val - 1.0) > tol:
                    return False
    np.random.seed(42)
    for _ in range(5000):
        a = np.random.randint(0, n*m)
        b = np.random.randint(0, m*p)
        c = np.random.randint(0, p*n)
        i1,j1 = divmod(a, m)
        j2,k1 = divmod(b, p)
        k2,i2 = divmod(c, n)
        if i1==i2 and j1==j2 and k1==k2:
            continue
        val = np.dot(U_arr[a,:]*V_arr[b,:], W_arr[c,:])
        if abs(val) > tol:
            return False
    return True

def construct():
    start = time.time()
    deadline = start + 110.0
    
    U1, V1, W1 = _strassen_2x2()  # shapes (4,7)
    U2, V2, W2 = _schoolbook_3x3()  # shapes (9,9)
    R1, R2 = 7, 9
    R_total = R1 * R2  # 63
    
    # Kronecker product construction
    # U: row = (br*3+ii)*6 + (bc*3+jj), col = r1*9+r2
    #   U[row, col] = U1[br, r1] * U2[ii*3+jj, r2]
    # V: row = (bc*3+jj)*6 + (bo*3+kk), col = r1*9+r2
    #   V[row, col] = V1[bc, r1] * V2[jj*3+kk, r2]
    # W: row = (bo*3+kk)*6 + (br*3+ii), col = r1*9+r2
    #   W[row, col] = W1[bo, r1] * W2[kk*3+ii, r2]
    
    U = np.zeros((36, R_total))
    V = np.zeros((36, R_total))
    W = np.zeros((36, R_total))
    
    for r1 in range(R1):
        for r2 in range(R2):
            r = r1 * R2 + r2
            for br in range(2):
                for bc in range(2):
                    for ii in range(3):
                        for jj in range(3):
                            u_row = (br*3 + ii) * 6 + (bc*3 + jj)
                            U[u_row, r] = U1[br, r1] * U2[ii*3 + jj, r2]
                    for kk in range(3):
                        for bo in range(2):
                            w_row = (bo*3 + kk) * 6 + (br*3 + ii)
                            W[w_row, r] = W1[bo, r1] * W2[kk*3 + ii, r2]
                    for jj in range(3):
                        for kk in range(3):
                            for bo in range(2):
                                v_row = (bc*3 + jj) * 6 + (bo*3 + kk)
                                V[v_row, r] = V1[bc, r1] * V2[jj*3 + kk, r2]
    
    if _verify(6, 6, 6, U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    U_sb = np.zeros((36, 216))
    V_sb = np.zeros((36, 216))
    W_sb = np.zeros((36, 216))
    col = 0
    for i in range(6):
        for j in range(6):
            for k in range(6):
                U_sb[i*6+j, col] = 1
                V_sb[j*6+k, col] = 1
                W_sb[k*6+i, col] = 1
                col += 1
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
