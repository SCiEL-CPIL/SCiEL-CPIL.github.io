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
    
    # Try ranks from 25 up to 49 (lower is better)
    # Use smarter initialization based on structure
    
    for r in range(25, 50):
        if time.time() > deadline - 5:
            break
        
        max_trials = 2000 if r < 35 else 1000
        
        for trial in range(max_trials):
            if time.time() > deadline:
                break
            
            # Strategy 1: Structured initialization with block patterns
            U = np.zeros((nm, r))
            V = np.zeros((mp, r))
            
            if trial % 3 == 0:
                # Block-structured: assign each term to a specific (i,j,k) pattern
                for t in range(r):
                    # Spread entries across different i,j,k
                    i_val = t % n
                    j_val = (t // n) % m
                    k_val = (t // (n * m)) % p
                    
                    # U: nonzero at positions involving (i_val, j_val)
                    for j in range(m):
                        if random.random() < 0.3:
                            U[i_val * m + j, t] = random.choice([-1, 1])
                    
                    # V: nonzero at positions involving (j_val, k_val)
                    for j in range(m):
                        if random.random() < 0.3:
                            V[j * p + k_val, t] = random.choice([-1, 1])
                    
                    # Ensure at least one nonzero in U and V for this term
                    if np.all(U[:, t] == 0):
                        U[i_val * m + j_val, t] = 1
                    if np.all(V[:, t] == 0):
                        V[j_val * p + k_val, t] = 1
                        
            elif trial % 3 == 1:
                # Random sparse with higher density
                for t in range(r):
                    nz_u = np.random.choice(nm, size=random.randint(2, 5), replace=False)
                    nz_v = np.random.choice(mp, size=random.randint(2, 5), replace=False)
                    U[nz_u, t] = np.random.choice([-1, 1], size=len(nz_u))
                    V[nz_v, t] = np.random.choice([-1, 1], size=len(nz_v))
            else:
                # Very sparse: 1-2 nonzeros per term
                for t in range(r):
                    nz_u = np.random.choice(nm, size=random.randint(1, 2), replace=False)
                    nz_v = np.random.choice(mp, size=random.randint(1, 2), replace=False)
                    U[nz_u, t] = np.random.choice([-1, 1], size=len(nz_u))
                    V[nz_v, t] = np.random.choice([-1, 1], size=len(nz_v))
            
            # Skip if any term is completely zero
            if np.any(np.all(U == 0, axis=0)) or np.any(np.all(V == 0, axis=0)):
                continue
            
            W_float = solve_for_W(U, V)
            
            # Try rounding to integers first, then half-integers
            W_int = np.round(W_float)
            if verify(U, V, W_int):
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W_int.tolist()
                }
            
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
