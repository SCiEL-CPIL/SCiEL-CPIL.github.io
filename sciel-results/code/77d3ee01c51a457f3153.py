import numpy as np
import time
import random

DIMENSIONS = [3, 3, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    nm = n * m  # 9
    mp = m * p  # 21
    pn = p * n  # 21
    
    # Schoolbook as fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W, rank
    
    best_U, best_V, best_W, best_rank = schoolbook()
    
    # Alternating minimization approach
    def alternating_minimize(r, max_iters=200, tol=1e-10, max_time=20):
        """Alternating least squares for tensor decomposition."""
        t0 = time.time()
        
        # Random initialization
        np.random.seed(42)
        U = np.random.randn(nm, r) * 0.5
        V = np.random.randn(mp, r) * 0.5
        W = np.random.randn(pn, r) * 0.5
        
        prev_obj = float('inf')
        
        for it in range(max_iters):
            if time.time() - t0 > max_time:
                break
            
            # Solve for U given V, W
            # For each (i,j): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for k=0..p-1
            for i in range(n):
                for j in range(m):
                    # Build linear system: A @ u = b where u = U[i*m+j, :]
                    # A[k, c] = V[j*p+k, c] * W[k*n+i, c]
                    A = np.zeros((p, r))
                    for k in range(p):
                        A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    b = np.ones(p)
                    # Solve least squares
                    u, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    U[i*m+j, :] = u
            
            # Solve for V given U, W
            # For each (j,k): sum_c V[j*p+k,c] * U[i*m+j,c] * W[k*n+i,c] = 1 for i=0..n-1
            for j in range(m):
                for k in range(p):
                    A = np.zeros((n, r))
                    for i in range(n):
                        A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    b = np.ones(n)
                    v, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    V[j*p+k, :] = v
            
            # Solve for W given U, V
            # For each (k,i): sum_c W[k*n+i,c] * U[i*m+j,c] * V[j*p+k,c] = 1 for j=0..m-1
            for k in range(p):
                for i in range(n):
                    A = np.zeros((m, r))
                    for j in range(m):
                        A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    b = np.ones(m)
                    w, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    W[k*n+i, :] = w
            
            # Compute objective
            obj = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                        obj += (val - 1.0)**2
            
            if abs(prev_obj - obj) < tol:
                break
            prev_obj = obj
        
        # Check if solution is valid
        if obj < 1e-6:
            # Try to round to half-integers
            U_round = np.round(U * 2) / 2
            V_round = np.round(V * 2) / 2
            W_round = np.round(W * 2) / 2
            
            obj_round = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U_round[i*m+j, :], V_round[j*p+k, :] * W_round[k*n+i, :])
                        obj_round += (val - 1.0)**2
            
            if obj_round < 1e-6:
                return U_round, V_round, W_round, r
            else:
                return U, V, W, r
        
        return None
    
    # Try ranks from low to high with multiple restarts
    for r in [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 42, 45, 48, 50, 54, 56, 60, 63]:
        if time.time() > deadline - 15:
            break
        
        found = False
        for restart in range(20):
            if time.time() > deadline - 10:
                break
            
            result = alternating_minimize(r, max_iters=150, max_time=min(5, deadline - time.time() - 5))
            if result is not None:
                U, V, W, rank = result
                if rank < best_rank:
                    best_U, best_V, best_W, best_rank = U, V, W, rank
                found = True
                break
        
        if found:
            break
    
    # Convert to lists
    return {
        "U": best_U.tolist() if hasattr(best_U, 'tolist') else best_U,
        "V": best_V.tolist() if hasattr(best_V, 'tolist') else best_V,
        "W": best_W.tolist() if hasattr(best_W, 'tolist') else best_W
    }
