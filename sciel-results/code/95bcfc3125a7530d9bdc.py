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
    
    # Build target tensor T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    deadline = time.time() + 140  # 10s safety margin
    
    def check_decomposition(U, V, W, r, tol=1e-8):
        """Check if U, V, W of rank r exactly decompose the tensor."""
        # Compute reconstructed tensor: T_recon[i,j,k] = sum_c U[i,j,c]*V[j,k,c]*W[k,i,c]
        # Vectorized: for each c, outer product contribution
        T_recon = np.zeros((nm, mp, pn))
        for c in range(r):
            # U[:, c] is (nm,), V[:, c] is (mp,), W[:, c] is (pn,)
            # Need to build 3D array where T_recon[i,j,k] += U[i,c]*V[j,c]*W[k,c]
            # This is outer product of U[:,c], V[:,c], W[:,c]
            T_recon += np.einsum('a,b,c->abc', U[:, c], V[:, c], W[:, c])
        return np.allclose(T_recon, T, atol=tol)
    
    def alternating_minimize(U, V, W, r, max_iter=500, tol=1e-10):
        """Vectorized alternating minimization for rank-r tensor decomposition."""
        for iteration in range(max_iter):
            # Fix V, W, optimize U
            # For each (i,j): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all k
            # M[i*m+j, k, c] = V[j*p+k, c] * W[k*n+i, c]
            # Solve: for each i,j, M[i*m+j, :, :] @ U[i*m+j, :] = ones(p)
            new_U = U.copy()
            for i in range(n):
                for j in range(m):
                    # M shape: (p, r) where M[k, c] = V[j*p+k, c] * W[k*n+i, c]
                    M = np.zeros((p, r))
                    for k in range(p):
                        M[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    b = np.ones(p)
                    try:
                        u, _, _, _ = np.linalg.lstsq(M, b, rcond=None)
                        new_U[i*m+j, :] = u
                    except:
                        pass
            
            # Fix U, W, optimize V
            new_V = V.copy()
            for j in range(m):
                for k in range(p):
                    # M shape: (n, r) where M[i, c] = U[i*m+j, c] * W[k*n+i, c]
                    M = np.zeros((n, r))
                    for i in range(n):
                        M[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    b = np.ones(n)
                    try:
                        v, _, _, _ = np.linalg.lstsq(M, b, rcond=None)
                        new_V[j*p+k, :] = v
                    except:
                        pass
            
            # Fix U, V, optimize W
            new_W = W.copy()
            for k in range(p):
                for i in range(n):
                    # M shape: (m, r) where M[j, c] = U[i*m+j, c] * V[j*p+k, c]
                    M = np.zeros((m, r))
                    for j in range(m):
                        M[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    b = np.ones(m)
                    try:
                        w, _, _, _ = np.linalg.lstsq(M, b, rcond=None)
                        new_W[k*n+i, :] = w
                    except:
                        pass
            
            U, V, W = new_U, new_V, new_W
            
            # Check convergence
            max_change = max(np.max(np.abs(U - new_U)), np.max(np.abs(V - new_V)), np.max(np.abs(W - new_W)))
            if max_change < tol:
                break
        
        return U, V, W
    
    def polish(U, V, W, r, max_iter=2000, tol=1e-12):
        """High-precision polish pass with tighter tolerance."""
        return alternating_minimize(U, V, W, r, max_iter=max_iter, tol=tol)
    
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
    
    best_rank = schoolbook_rank
    best_U = [[0] * best_rank for _ in range(nm)]
    best_V = [[0] * best_rank for _ in range(mp)]
    best_W = [[0] * best_rank for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i*m+j][col] = 1
                best_V[j*p+k][col] = 1
                best_W[k*n+i][col] = 1
                col += 1
    
    # Try ranks from small to large
    for r in range(1, schoolbook_rank + 1):
        if time.time() > deadline - 30:
            break
        
        found = False
        # More trials for smaller ranks, fewer for larger
        n_trials = max(10, min(50, 200 // r))
        
        for trial in range(n_trials):
            if time.time() > deadline - 15:
                break
            
            np.random.seed(trial * 7919 + r * 104729)
            scale = 1.0 / np.sqrt(r)
            U_init = np.random.randn(nm, r) * scale
            V_init = np.random.randn(mp, r) * scale
            W_init = np.random.randn(pn, r) * scale
            
            U, V, W = alternating_minimize(U_init, V_init, W_init, r, max_iter=300)
            
            if check_decomposition(U, V, W, r, tol=1e-8):
                # Polish with tighter tolerance
                U, V, W = polish(U, V, W, r, max_iter=1000, tol=1e-12)
                if check_decomposition(U, V, W, r, tol=1e-10):
                    best_rank = r
                    best_U = [[float(U[i,j]) for j in range(r)] for i in range(nm)]
                    best_V = [[float(V[i,j]) for j in range(r)] for i in range(mp)]
                    best_W = [[float(W[i,j]) for j in range(r)] for i in range(pn)]
                    found = True
                    break
        
        if found:
            # Found a valid decomposition at this rank, try to go lower
            continue
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
