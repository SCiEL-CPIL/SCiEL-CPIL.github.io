"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    nm = n * m  # 6
    mp = m * p  # 24
    pn = p * n  # 16
    
    schoolbook_rank = n * m * p  # 48
    
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin
    
    # Build target tensor for verification
    def check_decomposition(U, V, W, r, tol=1e-9):
        """Check if U, V, W of rank r exactly decompose the tensor."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(r):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def check_decomposition_fast(U, V, W, r):
        """Vectorized check."""
        T_recon = np.zeros((nm, mp, pn))
        for c in range(r):
            T_recon += np.einsum('a,b,c->abc', U[:, c], V[:, c], W[:, c])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if abs(T_recon[i*m+j, j*p+k, k*n+i] - 1.0) > 1e-9:
                        return False
        return True
    
    def alternating_minimize(U, V, W, r, max_iter=200, tol=1e-10):
        """Alternating minimization."""
        for iteration in range(max_iter):
            if time.time() > deadline - 20:
                break
            
            # Fix V, W, optimize U
            new_U = U.copy()
            for i in range(n):
                for j in range(m):
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
            
            max_change = max(
                np.max(np.abs(U - new_U)) if r > 0 else 0,
                np.max(np.abs(V - new_V)) if r > 0 else 0,
                np.max(np.abs(W - new_W)) if r > 0 else 0
            )
            if max_change < tol:
                break
        
        return U, V, W
    
    # Schoolbook decomposition as fallback
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
    
    # Try structured decompositions based on algebraic structure
    # Key insight: T[i*m+j, j*p+k, k*n+i] = 1
    # For fixed j, we need sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,k
    # This is a bilinear form in (i,k) for each j
    
    # Strategy: Try to decompose using shared structure across k values
    # For each j, the (i,k) grid is n x p = 2 x 8 = 16 entries
    # We can try to represent this as a sum of rank-1 matrices in the (i,k) space
    
    # Approach: For each j, find a decomposition of the 2x8 all-ones matrix
    # into rank-1 terms, and use those to build the overall decomposition
    
    # The all-ones 2x8 matrix has rank 1, so for each j we need only 1 term!
    # But the coupling across j means we need to coordinate.
    
    # Let's try: U[i*m+j, c] depends on (i,j,c), V[j*p+k,c] on (j,k,c), W[k*n+i,c] on (k,i,c)
    
    # Simple structured approach: use Fourier-like basis for k dimension
    # Since p=8, we can use 8 frequency components
    
    # Try rank = n * m * p / some_factor
    # Theoretical lower bound: max(n*m, m*p, p*n) = max(6, 24, 16) = 24
    
    best_rank = schoolbook_rank
    best_U, best_V, best_W = schoolbook()
    
    # Try ranks from lower bound up
    lower_bound = max(n*m, m*p, p*n)  # 24
    
    for r in range(lower_bound, schoolbook_rank + 1):
        if time.time() > deadline - 30:
            break
        
        found = False
        
        # Multiple initialization strategies
        n_trials = max(5, min(30, 100 // max(1, r - lower_bound)))
        
        for trial in range(n_trials):
            if time.time() > deadline - 15:
                break
            
            np.random.seed(trial * 1000003 + r * 7919)
            
            # Different initialization strategies
            if trial % 3 == 0:
                # Random
                U_init = np.random.randn(nm, r) * 0.5
                V_init = np.random.randn(mp, r) * 0.5
                W_init = np.random.randn(pn, r) * 0.5
            elif trial % 3 == 1:
                # Structured: based on Kronecker products
                U_init = np.zeros((nm, r))
                V_init = np.zeros((mp, r))
                W_init = np.zeros((pn, r))
                # Try to assign columns to (i,k) pairs with shared j
                col = 0
                for i in range(n):
                    for k in range(p):
                        if col >= r:
                            break
                        # Distribute across j
                        for j in range(m):
                            U_init[i*m+j, col] = 1.0 / np.sqrt(m)
                            V_init[j*p+k, col] = 1.0 / np.sqrt(m)
                            W_init[k*n+i, col] = 1.0 / np.sqrt(m)
                        col += 1
                # Add random perturbation
                U_init += np.random.randn(nm, r) * 0.1
                V_init += np.random.randn(mp, r) * 0.1
                W_init += np.random.randn(pn, r) * 0.1
            else:
                # Sparse structured
                U_init = np.random.randn(nm, r) * 0.3
                V_init = np.random.randn(mp, r) * 0.3
                W_init = np.random.randn(pn, r) * 0.3
                # Make some entries exactly zero to encourage sparsity
                for idx in np.random.choice(nm * r, size=nm * r // 3, replace=False):
                    U_init.flat[idx] = 0
                for idx in np.random.choice(mp * r, size=mp * r // 3, replace=False):
                    V_init.flat[idx] = 0
                for idx in np.random.choice(pn * r, size=pn * r // 3, replace=False):
                    W_init.flat[idx] = 0
            
            U, V, W = alternating_minimize(U_init, V_init, W_init, r, max_iter=150)
            
            if check_decomposition_fast(U, V, W, r):
                # Polish
                U, V, W = alternating_minimize(U, V, W, r, max_iter=500, tol=1e-12)
                if check_decomposition_fast(U, V, W, r):
                    best_rank = r
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                    found = True
                    break
        
        if found:
            continue
    
    # Convert to list format
    U_list = [[float(best_U[i,j]) for j in range(best_rank)] for i in range(nm)]
    V_list = [[float(best_V[i,j]) for j in range(best_rank)] for i in range(mp)]
    W_list = [[float(best_W[i,j]) for j in range(best_rank)] for i in range(pn)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
