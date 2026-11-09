import numpy as np
import time

DIMENSIONS = [3, 3, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    
    def verify(U, V, W, tol=1e-6):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def to_half_int(mat):
        return np.round(mat * 2) / 2.0
    
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
        return U, V, W, rank
    
    best_U, best_V, best_W, best_rank = schoolbook()
    
    # Try to find lower rank
    for rank in range(24, n*m*p):
        if time.time() > deadline - 50:
            break
        
        found_valid = False
        for restart in range(15):
            if time.time() > deadline - 50:
                break
            
            U = np.random.randn(n*m, rank)
            V = np.random.randn(m*p, rank)
            W = np.random.randn(p*n, rank)
            
            best_err = float('inf')
            best_U_iter = U.copy()
            best_V_iter = V.copy()
            best_W_iter = W.copy()
            
            for it in range(150):
                if time.time() > deadline - 50:
                    break
                
                # Solve for W
                for i in range(n):
                    for k in range(p):
                        A_mat = np.zeros((m, rank))
                        for j in range(m):
                            A_mat[j, :] = U[i*m+j, :] * V[j*p+k, :]
                        W[k*n+i, :] = np.linalg.lstsq(A_mat, np.ones(m), rcond=None)[0]
                
                # Solve for U
                for i in range(n):
                    for j in range(m):
                        A_mat = np.zeros((p, rank))
                        for k in range(p):
                            A_mat[k, :] = V[j*p+k, :] * W[k*n+i, :]
                        U[i*m+j, :] = np.linalg.lstsq(A_mat, np.ones(p), rcond=None)[0]
                
                # Solve for V
                for j in range(m):
                    for k in range(p):
                        A_mat = np.zeros((n, rank))
                        for i in range(n):
                            A_mat[i, :] = U[i*m+j, :] * W[k*n+i, :]
                        V[j*p+k, :] = np.linalg.lstsq(A_mat, np.ones(n), rcond=None)[0]
                
                # Error
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                            err += (val - 1.0)**2
                
                if err < best_err:
                    best_err = err
                    best_U_iter = U.copy()
                    best_V_iter = V.copy()
                    best_W_iter = W.copy()
                
                if err < 1e-14:
                    break
            
            # Snap and verify
            U_s = to_half_int(best_U_iter)
            V_s = to_half_int(best_V_iter)
            W_s = to_half_int(best_W_iter)
            
            if verify(U_s, V_s, W_s, tol=1e-3):
                best_U, best_V, best_W, best_rank = U_s, V_s, W_s, rank
                found_valid = True
                break
        
        if found_valid:
            break
    
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
