import numpy as np
import time
import random

DIMENSIONS = [2, 4, 4]

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m + j][c] * V[j*p + k][c] * W[k*n + i][c]
                if abs(s - 1.0) > tol:
                    return False
    # Also check that zero entries stay zero
    for a in range(n*m):
        for b in range(m*p):
            for c in range(p*n):
                # Check if this is a valid (i,j,k) triple
                i = a // m
                j = a % m
                j2 = b // p
                k = b % p
                k2 = c // n
                i2 = c % n
                if not (j == j2 and k == k2 and i == i2):
                    s = 0.0
                    for t in range(rank):
                        s += U[a][t] * V[b][t] * W[c][t]
                    if abs(s) > tol:
                        return False
    return True

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110
    
    # Schoolbook as fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m + j][col] = 1
                V_school[j*p + k][col] = 1
                W_school[k*n + i][col] = 1
                col += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school
    
    random.seed(42)
    np.random.seed(42)
    
    # Try to find lower rank decompositions
    # For 2x4 x 4x4, try ranks from 24 to 31
    for rank in range(24, best_rank):
        if time.time() > deadline:
            break
        
        # Multiple attempts for this rank
        max_attempts = 20
        found = False
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Initialize with structured approach
            # Use the fact that we have 2 rows in A
            # For each (j,k), we need to combine contributions from i=0 and i=1
            
            # Start with schoolbook and try to merge
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            W = np.zeros((p*n, rank))
            
            # Assign schoolbook terms to columns
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if col < rank:
                            U[i*m + j, col] = 1.0
                            V[j*p + k, col] = 1.0
                            W[k*n + i, col] = 1.0
                            col += 1
            
            # Now try to optimize: we have rank columns, 32 schoolbook terms
            # If rank < 32, some columns must serve multiple terms
            # Use random perturbation and optimization
            
            # Add random perturbation to unused or all columns
            if rank < 32:
                # We need to compress 32 terms into rank columns
                # Use random linear combinations
                for c in range(rank):
                    if c < 32:
                        # Keep base structure but add perturbation
                        U[:, c] += np.random.randn(n*m) * 0.1
                        V[:, c] += np.random.randn(m*p) * 0.1
                        W[:, c] += np.random.randn(p*n) * 0.1
                    else:
                        U[:, c] = np.random.randn(n*m) * 0.5
                        V[:, c] = np.random.randn(m*p) * 0.5
                        W[:, c] = np.random.randn(p*n) * 0.5
            
            # Optimize using scipy
            from scipy.optimize import minimize
            
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            size_U = n*m*rank
            size_V = m*p*rank
            size_W = p*n*rank
            
            def objective(x):
                Uf = x[:size_U].reshape(n*m, rank)
                Vf = x[size_U:size_U+size_V].reshape(m*p, rank)
                Wf = x[size_U+size_V:].reshape(p*n, rank)
                err = 0.0
                # Non-zero entries must equal 1
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = np.dot(Uf[i*m+j], Vf[j*p+k] * Wf[k*n+i])
                            err += (s - 1.0)**2
                # Zero entries must equal 0 (sample some to keep it fast)
                for _ in range(100):
                    a = random.randint(0, n*m-1)
                    b = random.randint(0, m*p-1)
                    c = random.randint(0, p*n-1)
                    i = a // m
                    j = a % m
                    j2 = b // p
                    k = b % p
                    k2 = c // n
                    i2 = c % n
                    if not (j == j2 and k == k2 and i == i2):
                        s = np.dot(Uf[a], Vf[b] * Wf[c])
                        err += s**2
                return err
            
            try:
                result = minimize(objective, x0, method='Nelder-Mead',
                                 options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-10})
            except:
                continue
            
            x_opt = result.x
            U_opt = x_opt[:size_U].reshape(n*m, rank)
            V_opt = x_opt[size_U:size_U+size_V].reshape(m*p, rank)
            W_opt = x_opt[size_U+size_V:].reshape(p*n, rank)
            
            # Round to half-integers
            U_r = np.round(U_opt * 2) / 2.0
            V_r = np.round(V_opt * 2) / 2.0
            W_r = np.round(W_opt * 2) / 2.0
            
            # Check if rounded version is exact
            exact = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = np.dot(U_r[i*m+j], V_r[j*p+k] * W_r[k*n+i])
                        if abs(s - 1.0) > 1e-9:
                            exact = False
                            break
                    if not exact:
                        break
                if not exact:
                    break
            
            if exact:
                # Check zero entries
                for a in range(n*m):
                    for b in range(m*p):
                        for c in range(p*n):
                            i = a // m
                            j = a % m
                            j2 = b // p
                            k = b % p
                            k2 = c // n
                            i2 = c % n
                            if not (j == j2 and k == k2 and i == i2):
                                s = np.dot(U_r[a], V_r[b] * W_r[c])
                                if abs(s) > 1e-9:
                                    exact = False
                                    break
                        if not exact:
                            break
                    if not exact:
                        break
            
            if exact:
                U_final = [[float(x) for x in row] for row in U_r.tolist()]
                V_final = [[float(x) for x in row] for row in V_r.tolist()]
                W_final = [[float(x) for x in row] for row in W_r.tolist()]
                if _verify(U_final, V_final, W_final, n, m, p):
                    best_rank = rank
                    best_result = {"U": U_final, "V": V_final, "W": W_final}
                    found = True
                    break
        
        if found:
            break
    
    return best_result
