import numpy as np
import time

DIMENSIONS = [3, 3, 4]

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i,j,k] = 1 for all valid indices."""
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
    return T

def _tensor_to_matrices(U, V, W):
    """Given factor matrices, compute the decomposed tensor."""
    n, m, p = DIMENSIONS
    T = np.zeros((n, m, p))
    rank = U.shape[1]
    for c in range(rank):
        u = U[:, c].reshape(n, m)
        v = V[:, c].reshape(m, p)
        w = W[:, c].reshape(p, n)
        # The decomposition: T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
        # Wait, I need to be careful about the index conventions.
        # U has shape (n*m, rank), indexed by [i*m + j, c]
        # V has shape (m*p, rank), indexed by [j*p + k, c]  
        # W has shape (p*n, rank), indexed by [k*n + i, c]
        # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += U[i*m + j, c] * V[j*p + k, c] * W[k*n + i, c]
    return T

def _verify_decomposition(U, V, W, T, tol=1e-6):
    """Check if the decomposition exactly reconstructs T."""
    n, m, p = DIMENSIONS
    T_rec = np.zeros((n, m, p))
    rank = U.shape[1]
    for c in range(rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_rec[i, j, k] += U[i*m + j, c] * V[j*p + k, c] * W[k*n + i, c]
    return np.max(np.abs(T_rec - T)) < tol

def _try_rank(r, n, m, p, T, max_iters=500, tol=1e-8):
    """Try to find a rank-r decomposition using ALS."""
    np.random.seed(42)
    U = np.random.randn(n*m, r) * 0.5
    V = np.random.randn(m*p, r) * 0.5
    W = np.random.randn(p*n, r) * 0.5
    
    for it in range(max_iters):
        # Update U
        for c in range(r):
            # U[:,c] should minimize ||T - sum_c U_c ⊗ V_c ⊗ W_c||
            # This is equivalent to: for each (i,j), U[i,j,c] * (V_c ⊗ W_c)[j,k,i] 
            # Actually, let's use the standard ALS for tensor decomposition.
            # Fix V and W, solve for U.
            # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
            # For fixed c, U[i*m+j, c] appears multiplied by V[j*p+k, c] * W[k*n+i, c]
            # This is a linear system in U[:,c]
            
            # Build the linear system for U[:,c]
            # Residual: R = T - sum_{c' != c} U[:,c'] ⊗ V[:,c'] ⊗ W[:,c']
            # Then we need: R[i,j,k] = U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
            # This is still nonlinear in U because V and W are fixed but the product structure...
            
            # Actually, ALS for CP decomposition:
            # T = sum_c U_c ⊗ V_c ⊗ W_c
            # To update U_c: U_c = argmin ||T - sum_{c'≠c} U_{c'}⊗V_{c'}⊗W_{c'} - U_c⊗V_c⊗W_c||
            # This is a linear least squares problem.
            
            pass  # This is getting complex, let me use a simpler approach
    
    # Use a simpler approach: random restarts with local search
    return None

def construct():
    n, m, p = DIMENSIONS
    T = _build_tensor(n, m, p)
    
    # Start with schoolbook as fallback
    rank_school = n * m * p
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m + j, col] = 1
                V_school[j*p + k, col] = 1
                W_school[k*n + i, col] = 1
                col += 1
    
    best_rank = rank_school
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Try to find lower rank decompositions
    for r in range(20, rank_school):
        if time.time() > 130:  # Safety margin
            break
        
        found = False
        for attempt in range(10):
            np.random.seed(r * 100 + attempt)
            U = np.random.randn(n*m, r)
            V = np.random.randn(m*p, r)
            W = np.random.randn(p*n, r)
            
            # Simple gradient-free local search
            best_err = np.inf
            for step in range(200):
                T_rec = _tensor_to_matrices(U, V, W)
                err = np.sum((T_rec - T)**2)
                if err < best_err:
                    best_err = err
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
                if err < 1e-10:
                    found = True
                    break
                
                # Perturb
                U += np.random.randn(n*m, r) * 0.1
                V += np.random.randn(m*p, r) * 0.1
                W += np.random.randn(p*n, r) * 0.1
            
            if found:
                best_rank = r
                best_U, best_V, best_W = U, V, W
                break
        
        if found:
            break
    
    # Convert to lists with half-integer rounding
    def to_half_int_list(arr):
        return [[round(x * 2) / 2 for x in row] for row in arr]
    
    U_list = to_half_int_list(best_U)
    V_list = to_half_int_list(best_V)
    W_list = to_half_int_list(best_W)
    
    return {"U": U_list, "V": V_list, "W": W_list}
