"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    schoolbook_rank = n * m * p
    
    def check_decomposition(U, V, W, r):
        """Check if U, V, W of rank r exactly decompose the tensor."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(r):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(val - 1.0) > 1e-8:
                        return False
        return True
    
    def alternating_minimize(U, V, W, r, max_iter=1000, tol=1e-12):
        """Alternating minimization to find rank-r decomposition."""
        for iteration in range(max_iter):
            # Fix V, W, optimize U
            # For each (i,j), we need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all k
            # This is: sum_c U[i*m+j, c] * (V[j*p+k, c] * W[k*n+i, c]) = 1
            # Let A[i,j,k,c] = V[j*p+k, c] * W[k*n+i, c]
            # Then: sum_c U[i*m+j, c] * A[i,j,k,c] = 1 for all k
            # This is a linear system in U[i*m+j, :]
            
            new_U = U.copy()
            for i in range(n):
                for j in range(m):
                    # Build linear system: M * u = b where u = U[i*m+j, :]
                    # M[k, c] = V[j*p+k, c] * W[k*n+i, c]
                    # b[k] = 1
                    M = np.zeros((p, r))
                    for k in range(p):
                        for c in range(r):
                            M[k, c] = V[j*p+k, c] * W[k*n+i, c]
                    b = np.ones(p)
                    # Solve M * u = b in least squares sense
                    try:
                        u, residuals, rank_m, sv = np.linalg.lstsq(M, b, rcond=None)
                        new_U[i*m+j, :] = u
                    except:
                        pass
            
            # Fix U, W, optimize V
            new_V = V.copy()
            for j in range(m):
                for k in range(p):
                    # Need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i
                    # M[i, c] = U[i*m+j, c] * W[k*n+i, c]
                    # b[i] = 1
                    M = np.zeros((n, r))
                    for i in range(n):
                        for c in range(r):
                            M[i, c] = U[i*m+j, c] * W[k*n+i, c]
                    b = np.ones(n)
                    try:
                        v, residuals, rank_m, sv = np.linalg.lstsq(M, b, rcond=None)
                        new_V[j*p+k, :] = v
                    except:
                        pass
            
            # Fix U, V, optimize W
            new_W = W.copy()
            for k in range(p):
                for i in range(n):
                    # Need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all j
                    # M[j, c] = U[i*m+j, c] * V[j*p+k, c]
                    # b[j] = 1
                    M = np.zeros((m, r))
                    for j in range(m):
                        for c in range(r):
                            M[j, c] = U[i*m+j, c] * V[j*p+k, c]
                    b = np.ones(m)
                    try:
                        w, residuals, rank_m, sv = np.linalg.lstsq(M, b, rcond=None)
                        new_W[k*n+i, :] = w
                    except:
                        pass
            
            U, V, W = new_U, new_V, new_W
            
            # Check convergence
            max_change = max(np.max(np.abs(U - new_U)), np.max(np.abs(V - new_V)), np.max(np.abs(W - new_W)))
            if max_change < tol:
                break
        
        return U, V, W
    
    deadline = time.time() + 100
    
    # Try ranks from small to large
    for r in range(1, schoolbook_rank + 1):
        if time.time() > deadline - 30:
            break
        
        found = False
        for trial in range(30):
            if time.time() > deadline - 15:
                break
            
            np.random.seed(trial * 7919 + r * 104729)
            scale = 1.0 / np.sqrt(r)
            U_init = np.random.randn(nm, r) * scale
            V_init = np.random.randn(mp, r) * scale
            W_init = np.random.randn(pn, r) * scale
            
            U, V, W = alternating_minimize(U_init, V_init, W_init, r, max_iter=500)
            
            if check_decomposition(U, V, W, r):
                U_list = [[float(U[i,j]) for j in range(r)] for i in range(nm)]
                V_list = [[float(V[i,j]) for j in range(r)] for i in range(mp)]
                W_list = [[float(W[i,j]) for j in range(r)] for i in range(pn)]
                return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: schoolbook
    rank = schoolbook_rank
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][column] = 1
                V[j*p+k][column] = 1
                W[k*n+i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
