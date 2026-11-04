import numpy as np
import time

def construct():
    n, m, p = 2, 5, 5
    start_time = time.time()
    deadline = start_time + 140
    
    def schoolbook():
        U = np.zeros((n*m, n*m*p))
        V = np.zeros((m*p, n*m*p))
        W = np.zeros((p*n, n*m*p))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {'U': U, 'V': V, 'W': W}
    
    def verify(U, V, W):
        n, m, p = 2, 5, 5
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1) > 1e-9:
                        return False
        return True
    
    def solve_for_U(V, W, r):
        U = np.zeros((n*m, r))
        for i in range(n):
            for j in range(m):
                A = np.zeros((p, r))
                b = np.ones(p)
                for k in range(p):
                    A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                try:
                    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    U[i*m+j, :] = x
                except:
                    U[i*m+j, :] = np.random.randn(r) * 0.1
        return U
    
    def solve_for_V(U, W, r):
        V = np.zeros((m*p, r))
        for j in range(m):
            for k in range(p):
                A = np.zeros((n, r))
                b = np.ones(n)
                for i in range(n):
                    A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                try:
                    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    V[j*p+k, :] = x
                except:
                    V[j*p+k, :] = np.random.randn(r) * 0.1
        return V
    
    def solve_for_W(U, V, r):
        W = np.zeros((p*n, r))
        for k in range(p):
            for i in range(n):
                A = np.zeros((m, r))
                b = np.ones(m)
                for j in range(m):
                    A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                try:
                    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    W[k*n+i, :] = x
                except:
                    W[k*n+i, :] = np.random.randn(r) * 0.1
        return W
    
    def iterative_solve(r, max_iters=500, seed=None):
        if seed is not None:
            np.random.seed(seed)
        
        U = np.random.randn(n*m, r) * 0.3
        V = np.random.randn(m*p, r) * 0.3
        W = np.random.randn(p*n, r) * 0.3
        
        for it in range(max_iters):
            if time.time() > deadline:
                return None
            
            U = solve_for_U(V, W, r)
            V = solve_for_V(U, W, r)
            W = solve_for_W(U, V, r)
            
            err = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                        err += (val - 1)**2
            
            if err < 1e-12:
                return U, V, W
        
        return None
    
    best_sol = None
    best_rank = 50
    
    for r in range(20, 50):
        if time.time() > deadline:
            break
        
        for attempt in range(15):
            if time.time() > deadline:
                break
            
            sol = iterative_solve(r, max_iters=300)
            if sol is not None:
                U, V, W = sol
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                if verify(U_rounded, V_rounded, W_rounded):
                    best_sol = {'U': U_rounded, 'V': V_rounded, 'W': W_rounded}
                    best_rank = r
                    break
                elif verify(U, V, W):
                    best_sol = {'U': U, 'V': V, 'W': W}
                    best_rank = r
                    break
        
        if best_sol is not None and best_rank == r:
            break
    
    if best_sol is not None:
        return best_sol
    
    return schoolbook()
