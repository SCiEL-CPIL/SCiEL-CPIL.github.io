"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 145.0  # safety margin
    
    n, m, p = DIMENSIONS
    # U: (n*m) x r, V: (m*p) x r, W: (p*n) x r
    # Tensor T: T[i*m+j, j*p+k, k*n+i] = 1 for valid (i,j,k)
    
    # Precompute non-zero indices
    # u = i*m + j
    # v = j*p + k
    # w = k*n + i
    
    nonzeros_U = [[] for _ in range(n*m)]
    nonzeros_V = [[] for _ in range(m*p)]
    nonzeros_W = [[] for _ in range(p*n)]
    
    T_flat = {}
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i*m + j
                v = j*p + k
                w = k*n + i
                nonzeros_U[u].append((v, w))
                nonzeros_V[v].append((u, w))
                nonzeros_W[w].append((u, v))
                T_flat[(u, v, w)] = 1.0

    best_rank = n*m*p  # schoolbook
    best_result = None
    
    # Try ranks from low to high
    # Start from a reasonable lower bound, e.g., 30 or 40
    # Schoolbook is 54.
    # Let's try 30 to 53.
    for r in range(30, n*m*p):
        if time.time() > deadline - 10:
            break
        
        success = False
        # Increase trials for lower ranks?
        # Lower ranks are harder.
        num_trials = 50 if r < 40 else 100
        
        for trial in range(num_trials):
            if time.time() > deadline - 5:
                break
            
            # Random initialization
            np.random.seed(trial * 1000 + r)
            U = np.random.randn(n*m, r) * 0.1
            V = np.random.randn(m*p, r)
            W = np.random.randn(p*n, r)
            
            # Alternating minimization
            for it in range(200):
                if time.time() > deadline - 5:
                    break
                
                # Update U
                for i in range(n*m):
                    nz = nonzeros_U[i]
                    if not nz:
                        continue
                    A_sub = np.array([V[v] * W[w] for v, w in nz])
                    b_sub = np.ones(len(nz))
                    # Solve least squares
                    U[i], residuals, rank_A, sv = np.linalg.lstsq(A_sub, b_sub, rcond=None)
                
                # Update V
                for j in range(m*p):
                    nz = nonzeros_V[j]
                    if not nz:
                        continue
                    A_sub = np.array([U[u] * W[w] for u, w in nz])
                    b_sub = np.ones(len(nz))
                    V[j], residuals, rank_A, sv = np.linalg.lstsq(A_sub, b_sub, rcond=None)
                
                # Update W
                for k in range(p*n):
                    nz = nonzeros_W[k]
                    if not nz:
                        continue
                    A_sub = np.array([U[u] * V[v] for u, v in nz])
                    b_sub = np.ones(len(nz))
                    W[k], residuals, rank_A, sv = np.linalg.lstsq(A_sub, b_sub, rcond=None)
            
            # Check if we have a good solution
            # Calculate error efficiently
            error = 0.0
            for i in range(n*m):
                for v, w in nonzeros_U[i]:
                    val = np.sum(U[i] * V[v] * W[w])
                    error += (val - 1.0)**2
            
            if error < 1e-6:
                # Round to half-integers
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                # Verify rounded solution
                error_rounded = 0.0
                for i in range(n*m):
                    for v, w in nonzeros_U[i]:
                        val = np.sum(U_rounded[i] * V_rounded[v] * W_rounded[w])
                        error_rounded += (val - 1.0)**2
                
                if error_rounded < 1e-6:
                    success = True
                    best_rank = r
                    best_result = (U_rounded, V_rounded, W_rounded)
                    break
        
        if success:
            break
    
    if best_result is None:
        # Fall back to schoolbook
        r = n*m*p
        U = np.zeros((n*m, r))
        V = np.zeros((m*p, r))
        W = np.zeros((p*n, r))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i*m + j
                    v = j*p + k
                    w = k*n + i
                    U[u, col] = 1
                    V[v, col] = 1
                    W[w, col] = 1
                    col += 1
        best_result = (U, V, W)
    
    U, V, W = best_result
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
