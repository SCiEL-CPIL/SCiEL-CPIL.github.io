import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    # Matrix multiplication tensor: C = A * B
    # A is n x m, B is m x p, C is n x p
    # T[i, j, k] = 1 if we're multiplying A[i,j] * B[j,k] to contribute to C[i,k]
    # U index: i*m + j  (row n*m)
    # V index: j*p + k  (row m*p)
    # W index: k*n + i  (row p*n) -- per the trace convention W[k*n + i, r]
    
    # The tensor equation for rank R:
    # For each (i, j, k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Trivial decomposition: R = n*m*p = 30
    # Each term (i,j,k) gets its own rank slot.
    
    # We try to find a lower rank solution.
    # Known: for 2x3 * 3x5, the rank is at most 30 (trivial) and at least... 
    # Let's try to find rank 29 or lower.
    
    # Strategy: Use a deterministic search with small integer/half-integer values.
    # For rank R, we need to find U (n*m x R), V (m*p x R), W (p*n x R)
    # such that for all (i,j,k): sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
    
    # Given the complexity, let's first verify the trivial solution works,
    # then attempt a structured search for rank 29.
    
    # The trivial rank-30 solution:
    R = 30
    U = np.zeros((n*m, R), dtype=float)
    V = np.zeros((m*p, R), dtype=float)
    W = np.zeros((p*n, R), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1.0
                V[j*p + k, col] = 1.0
                W[k*n + i, col] = 1.0
                col += 1
    
    # Verify the trivial solution
    def verify(U, V, W):
        n, m, p = DIMENSIONS
        R = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for r in range(R):
                        total += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(total - 1.0) > 1e-9:
                        return False
        return True
    
    if verify(U, V, W):
        # Check if we can find a rank 29 solution
        # Try a simple perturbation: can we merge two of the rank-30 terms?
        # In the trivial solution, each (i,j,k) triple has its own rank dimension.
        # To reduce rank by 1, we'd need to express one tensor fiber as a combination
        # of the others, which generally isn't possible for the identity-like structure.
        
        # However, let's try a random search for rank 29 with small coefficients.
        # Given the 120s timeout, we'll do a limited search.
        
        import random
        random.seed(42)
        
        found = False
        best_rank = R
        
        for trial_R in [29, 28, 27, 26, 25, 24, 23, 22, 21, 20]:
            for attempt in range(2000):
                # Generate random factor matrices with small entries
                scale = 1.0
                U_rand = np.random.randint(-2, 3, size=(n*m, trial_R)).astype(float) * scale
                V_rand = np.random.randint(-2, 3, size=(m*p, trial_R)).astype(float) * scale
                W_rand = np.random.randint(-2, 3, size=(p*n, trial_R)).astype(float) * scale
                
                # Verify
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            total = 0.0
                            for r in range(trial_R):
                                total += U_rand[i*m+j, r] * V_rand[j*p+k, r] * W_rand[k*n+i, r]
                            if abs(total - 1.0) > 1e-9:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                
                if valid:
                    U = U_rand
                    V = V_rand
                    W = W_rand
                    R = trial_R
                    found = True
                    break
            
            if found:
                break
        
        if found:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Return the trivial rank-30 solution as fallback
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
