import numpy as np
import time
import random

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit
    
    # Build the target tensor: T[i,j,k] = 1 for all valid (i,j,k)
    # In flattened form: T[u,v,w] where u=i*m+j, v=j*p+k, w=k*n+i
    # But actually, the tensor is defined on the consistent indices only.
    # For the decomposition, we need: for all i,j,k:
    #   sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    
    # Let's use a structured approach: start with schoolbook and try to reduce rank
    # by finding linear dependencies or using known decompositions.
    
    # First, let's try a systematic search for rank < 50
    # We'll use a greedy approach: build up terms one by one
    
    def verify(U, V, W):
        """Verify that the decomposition is exact."""
        r = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    def solve_for_W(U, V):
        """Given U and V, solve for W such that the decomposition is exact."""
        r = U.shape[1]
        W_sol = np.zeros((pn, r), dtype=float)
        
        for w in range(pn):
            k, i = divmod(w, n)
            # For this w, we need: for all j:
            #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
            # Let x_c = W[k*n+i, c]
            # Then: sum_c (U[i*m+j, c] * V[j*p+k, c]) * x_c = 1 for all j
            
            # Build the system: A * x = b
            # A[j, c] = U[i*m+j, c] * V[j*p+k, c]
            # b[j] = 1
            A = np.zeros((m, r), dtype=float)
            for j in range(m):
                for c in range(r):
                    A[j, c] = U[i * m + j, c] * V[j * p + k, c]
            b = np.ones(m, dtype=float)
            
            # Solve A * x = b
            try:
                x = np.linalg.lstsq(A, b, rcond=None)[0]
                W_sol[w, :] = x
            except:
                W_sol[w, :] = 0
        
        return W_sol
    
    def round_to_half_integer(X):
        X2 = np.round(X * 2)
        return X2 / 2.0
    
    # Try ranks from 40 up to 49
    # Lower ranks are better (higher score)
    for r in range(40, 50):
        if time.time() > deadline - 5:
            break
        
        max_trials = 5000 if r < 45 else 2000
        
        for trial in range(max_trials):
            if time.time() > deadline:
                break
            
            # Sparse random initialization
            U = np.zeros((nm, r))
            V = np.zeros((mp, r))
            
            for t in range(r):
                # Pick random non-zero entries
                nz_u = np.random.choice(nm, size=min(3, nm), replace=False)
                nz_v = np.random.choice(mp, size=min(3, mp), replace=False)
                U[nz_u, t] = np.random.choice([-1, 1], size=len(nz_u))
                V[nz_v, t] = np.random.choice([-1, 1], size=len(nz_v))
            
            if np.any(np.all(U == 0, axis=0)) or np.any(np.all(V == 0, axis=0)):
                continue
            
            W_float = solve_for_W(U, V)
            W_r = round_to_half_integer(W_float)
            
            if verify(U, V, W_r):
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W_r.tolist()
                }
    
    # Fallback: Schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
