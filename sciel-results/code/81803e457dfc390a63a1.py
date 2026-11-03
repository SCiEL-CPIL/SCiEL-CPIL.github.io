import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = 3, 3, 8
    schoolbook_rank = n * m * p  # 72
    
    # Schoolbook as fallback
    U_sb = [[0]*schoolbook_rank for _ in range(n*m)]
    V_sb = [[0]*schoolbook_rank for _ in range(m*p)]
    W_sb = [[0]*schoolbook_rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j][col] = 1
                V_sb[j*p+k][col] = 1
                W_sb[k*n+i][col] = 1
                col += 1
    best_rank = schoolbook_rank
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Try to find lower rank using optimization
    for target_rank in range(71, 55, -1):
        if time.time() > deadline - 40:
            break
        
        found = False
        for seed in range(50):
            if time.time() > deadline - 35:
                break
            
            np.random.seed(seed)
            U = np.random.randn(n*m, target_rank)
            V = np.random.randn(m*p, target_rank)
            W = np.random.randn(p*n, target_rank)
            
            # ALS iterations
            for it in range(5000):
                if time.time() > deadline - 30:
                    break
                
                # Update U
                for i in range(n):
                    for j in range(m):
                        A = np.array([V[j*p+k,:] * W[k*n+i,:] for k in range(p)])
                        try:
                            U[i*m+j,:] = np.linalg.lstsq(A, np.ones(p), rcond=None)[0]
                        except:
                            pass
                
                # Update V
                for j in range(m):
                    for k in range(p):
                        A = np.array([U[i*m+j,:] * W[k*n+i,:] for i in range(n)])
                        try:
                            V[j*p+k,:] = np.linalg.lstsq(A, np.ones(n), rcond=None)[0]
                        except:
                            pass
                
                # Update W
                for k in range(p):
                    for i in range(n):
                        A = np.array([U[i*m+j,:] * V[j*p+k,:] for j in range(m)])
                        try:
                            W[k*n+i,:] = np.linalg.lstsq(A, np.ones(m), rcond=None)[0]
                        except:
                            pass
                
                # Check convergence
                err = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U[i*m+j,:], V[j*p+k,:]*W[k*n+i,:])
                            err += (val-1)**2
                if err < 1e-12:
                    found = True
                    break
            
            if found:
                # Round to half-integers
                U_r = np.round(U*2)/2
                V_r = np.round(V*2)/2
                W_r = np.round(W*2)/2
                
                # Verify exactness
                exact = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U_r[i*m+j,:], V_r[j*p+k,:]*W_r[k*n+i,:])
                            if abs(val-1) > 1e-9:
                                exact = False
                                break
                    if not exact:
                        break
                if not exact:
                    break
                if exact:
                    best_rank = target_rank
                    best_U = U_r.tolist()
                    best_V = V_r.tolist()
                    best_W = W_r.tolist()
                    break
        
        if found:
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
