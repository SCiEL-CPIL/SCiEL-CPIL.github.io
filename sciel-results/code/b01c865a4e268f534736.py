"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    def build_tensor():
        """Build the full tensor T of shape (n*m, m*p, p*n)."""
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i  # trace convention
                    T[a, b, c] = 1.0
        return T
    
    def verify(U, V, W, T, tol=1e-6):
        """Check if U, V, W decompose T."""
        r = U.shape[1]
        # Compute sum_r U[a,r]*V[b,r]*W[c,r] for all a,b,c
        # This is a tensor contraction
        # Efficient: for each r, outer product U[:,r] @ V[:,r] @ W[:,r]
        result = np.zeros(T.shape)
        for r_idx in range(r):
            result += np.einsum('a,b,c->abc', U[:, r_idx], V[:, r_idx], W[:, r_idx])
        return np.allclose(result, T, atol=tol)
    
    def round_to_half_int(arr):
        """Round array to nearest half-integer."""
        return np.round(arr * 2) / 2.0
    
    def try_rank(r, T, max_iters=2000, seed=42):
        """Try to find rank-r decomposition."""
        rng = np.random.RandomState(seed)
        nm, mp, pn = n*m, m*p, p*n
        
        for attempt in range(3):
            if time.time() > deadline:
                return None
            
            # Initialize with random half-integer values
            U = rng.randint(-2, 3, size=(nm, r)).astype(float) / 2.0
            V = rng.randint(-2, 3, size=(mp, r)).astype(float) / 2.0
            W = rng.randint(-2, 3, size=(pn, r)).astype(float) / 2.0
            
            # Try to refine using iterative method
            # We want: sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
            # This is nonlinear. Use alternating minimization or gradient descent.
            
            # Simple approach: try a few random perturbations and check
            # For now, just check if random init works (unlikely)
            if verify(U, V, W, T):
                return U, V, W
        
        return None
    
    # Try to find rank < 50
    T = build_tensor()
    
    # Start from rank 49 and go down
    for r in range(49, 30, -1):
        if time.time() > deadline - 10:
            break
        result = try_rank(r, T, seed=r)
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: schoolbook rank 50
    U_cols = []
    V_cols = []
    W_cols = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = [0] * (n * m)
                u[i * m + j] = 1
                v = [0] * (m * p)
                v[j * p + k] = 1
                w = [0] * (p * n)
                w[k * n + i] = 1
                U_cols.append(u)
                V_cols.append(v)
                W_cols.append(w)
    
    rank = len(U_cols)
    U = np.array(U_cols).T
    V = np.array(V_cols).T
    W = np.array(W_cols).T
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
