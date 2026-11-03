import numpy as np
import time
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140  # Safety margin
    
    # Schoolbook rank
    R_school = n * m * p  # 30
    
    # Try to find a rank-22 decomposition
    # The tensor equation: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # for all i in [0,n), j in [0,m), k in [0,p)
    
    # Strategy: Use a structured approach based on the fact that
    # the matrix multiplication tensor has symmetries.
    
    # For 2x3 x 3x5, try to find rank 22
    # We'll use a combination of:
    # 1. A parametric family inspired by Strassen
    # 2. Newton's method to solve the resulting equations
    
    # Let's try a different approach: use the fact that we can
    # decompose the multiplication into smaller sub-problems.
    
    # Actually, let me try a direct search with better initialization.
    # Key insight: the solution should have many zeros (sparse structure).
    
    # Try rank 22 with a sparse initialization
    R = 22
    
    # Initialize with a structured pattern
    # Use the idea that each "term" in the decomposition should
    # correspond to a specific pattern of indices
    
    # Let's try: initialize U, V, W with values from {0, 1, -1}
    # and use a combinatorial search
    
    # Actually, let me try a more systematic approach:
    # Use the fact that the problem is equivalent to finding
    # a low-rank completion of a tensor.
    
    # Let me try: start from schoolbook and try to merge terms
    # by finding linear dependencies.
    
    # Schoolbook: 30 terms, each with U[i*m+j, r]=1, V[j*p+k, r]=1, W[k*n+i, r]=1
    # Can we combine some of these?
    
    # Let's try a different strategy: use ALS but with:
    # 1. Better initialization (sparse, structured)
    # 2. More iterations
    # 3. Exact verification at the end
    
    best_result = None
    
    # Try multiple ranks from 22 upward
    for R in range(22, 26):
        if time.time() - start_time > deadline:
            break
        
        # Try several random initializations with sparse structure
        num_trials = 200 if R == 22 else 100
        
        for trial in range(num_trials):
            if time.time() - start_time > deadline:
                break
            
            # Sparse random initialization: most entries are 0
            density = 0.3
            U = np.zeros((n*m, R))
            V = np.zeros((m*p, R))
            W = np.zeros((p*n, R))
            
            # Initialize with random sparse patterns
            for r in range(R):
                # Each column has a few non-zero entries
                nz_u = max(1, int(np.random.exponential(2)))
                nz_v = max(1, int(np.random.exponential(2)))
                nz_w = max(1, int(np.random.exponential(2)))
                
                u_idx = np.random.choice(n*m, min(nz_u, n*m), replace=False)
                v_idx = np.random.choice(m*p, min(nz_v, m*p), replace=False)
                w_idx = np.random.choice(p*n, min(nz_w, p*n), replace=False)
                
                U[u_idx, r] = np.random.choice([-1, 1])
                V[v_idx, r] = np.random.choice([-1, 1])
                W[w_idx, r] = np.random.choice([-1, 1])
            
            # ALS iterations
            converged = False
            for it in range(500):
                if time.time() - start_time > deadline:
                    break
                
                # Solve for U
                for i in range(n):
                    for j in range(m):
                        A = np.zeros((p, R))
                        for k in range(p):
                            A[k, :] = V[j*p + k, :] * W[k*n + i, :]
                        b = np.ones(p)
                        try:
                            U[i*m + j, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                        except:
                            pass
                
                # Solve for V
                for j in range(m):
                    for k in range(p):
                        A = np.zeros((n, R))
                        for i in range(n):
                            A[i, :] = U[i*m + j, :] * W[k*n + i, :]
                        b = np.ones(n)
                        try:
                            V[j*p + k, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                        except:
                            pass
                
                # Solve for W
                for k in range(p):
                    for i in range(n):
                        A = np.zeros((m, R))
                        for j in range(m):
                            A[j, :] = U[i*m + j, :] * V[j*p + k, :]
                        b = np.ones(m)
                        try:
                            W[k*n + i, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                        except:
                            pass
                
                # Check error
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U[i*m + j, :], V[j*p + k, :] * W[k*n + i, :])
                            err += (val - 1.0)**2
                
                if err < 1e-10:
                    converged = True
                    break
                
                if err > 1e6:
                    break  # Diverged
            
            if converged:
                best_result = (R, U, V, W)
                break
        
        if best_result is not None:
            break
    
    if best_result is not None:
        R, U, V, W = best_result
        # Round to nearest half-integer if close
        U = np.round(U * 2) / 2
        V = np.round(V * 2) / 2
        W = np.round(W * 2) / 2
        
        # Verify
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m + j, :], V[j*p + k, :] * W[k*n + i, :])
                    if abs(val - 1.0) > 1e-8:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fall back to schoolbook
    R = n * m * p
    U = [[0]*R for _ in range(n*m)]
    V = [[0]*R for _ in range(m*p)]
    W = [[0]*R for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
