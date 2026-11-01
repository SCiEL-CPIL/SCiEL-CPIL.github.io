import numpy as np
import time

def construct():
    start = time.time()
    deadline = 145
    
    n, m, p = 2, 2, 13
    
    # Try 44-rank Strassen-extended decomposition
    total_rank = 44
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    col = 0
    for s in range(6):
        e_idx = 2*s
        f_idx = 2*s+1
        g_idx = 13+2*s
        h_idx = 13+2*s+1
        c00 = 4*s
        c01 = 4*s+1
        c10 = 4*s+2
        c11 = 4*s+3
        
        U[0,col]=1; U[3,col]=1; V[e_idx,col]=1; V[h_idx,col]=1; W[c00,col]=1; W[c11,col]=1; col+=1
        U[2,col]=1; U[3,col]=1; V[e_idx,col]=1; W[c10,col]=1; col+=1
        U[0,col]=1; V[f_idx,col]=1; V[g_idx,col]=-1; W[c01,col]=1; col+=1
        U[3,col]=1; V[e_idx,col]=1; V[f_idx,col]=-1; W[c00,col]=1; W[c10,col]=1; col+=1
        U[0,col]=1; U[1,col]=1; V[h_idx,col]=1; W[c00,col]=-1; W[c01,col]=1; col+=1
        U[2,col]=1; U[0,col]=-1; V[e_idx,col]=1; V[f_idx,col]=1; W[c11,col]=1; col+=1
        U[1,col]=1; U[3,col]=-1; V[g_idx,col]=1; V[h_idx,col]=1; W[c00,col]=1; col+=1
    
    U[0,col]=1; V[12,col]=1; W[24,col]=1; col+=1
    U[1,col]=1; V[25,col]=1; W[24,col]=1; col+=1
    U[2,col]=1; V[12,col]=1; W[25,col]=1; col+=1
    U[3,col]=1; V[25,col]=1; W[25,col]=1; col+=1
    
    # Verify
    error = 0.0
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    for kp in range(p):
                        for ip in range(n):
                            expected = 1.0 if (j==jp and i==ip and k==kp) else 0.0
                            actual = float(np.dot(U[i*m+j], V[jp*p+k] * W[kp*n+ip]))
                            error += abs(actual - expected)
    
    if error < 1e-10:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: schoolbook rank 52
    total_rank = 52
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
