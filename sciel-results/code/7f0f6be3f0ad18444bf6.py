"""Find rank-minimal exact tensor decomposition for 3x4x5 matrix multiplication."""
import time
import numpy as np
from scipy.optimize import minimize
from itertools import product

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 145.0
    
    # Build the target tensor T with shape (n*m, m*p, p*n)
    # T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for valid i,j,k
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Schoolbook solution as fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
        return U, V, W, rank
    
    U_sb, V_sb, W_sb, rank_sb = schoolbook()
    best_U, best_V, best_W = U_sb.copy(), V_sb.copy(), W_sb.copy()
    best_rank = rank_sb
    
    def verify(U, V, W, tol=1e-8):
        """Check if decomposition is exact."""
        r = U.shape[1]
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        return np.max(np.abs(T_approx - T)) < tol
    
    def polish(U, V, W, tol=1e-10):
        """High-precision polish: refine values and try rounding to half-integers."""
        r = U.shape[1]
        
        def flatten_params(U, V, W):
            return np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        def unflatten_params(x):
            U = x[:n*m*r].reshape(n*m, r)
            V = x[n*m*r:n*m*r+m*p*r].reshape(m*p, r)
            W = x[n*m*r+m*p*r:].reshape(p*n, r)
            return U, V, W
        
        def objective(x):
            U, V, W = unflatten_params(x)
            T_approx = np.einsum('ar,br,cr->abc', U, V, W)
            return np.sum((T_approx - T)**2)
        
        x0 = flatten_params(U, V, W)
        
        # Try multiple optimization passes with tightening tolerances
        for ftol, gtol, maxiter in [(1e-10, 1e-8, 200), (1e-12, 1e-10, 300)]:
            try:
                result = minimize(objective, x0, method='L-BFGS-B',
                                 options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol})
                x0 = result.x
            except Exception:
                break
        
        U_opt, V_opt, W_opt = unflatten_params(x0)
        
        # Try rounding to half-integers
        U_rounded = np.round(U_opt * 2) / 2
        V_rounded = np.round(V_opt * 2) / 2
        W_rounded = np.round(W_opt * 2) / 2
        
        if verify(U_rounded, V_rounded, W_rounded, tol=1e-6):
            return U_rounded, V_rounded, W_rounded
        
        # Try integer rounding
        U_int = np.round(U_opt)
        V_int = np.round(V_opt)
        W_int = np.round(W_opt)
        if verify(U_int, V_int, W_int, tol=1e-6):
            return U_int, V_int, W_int
        
        # Return polished real values if they verify
        if verify(U_opt, V_opt, W_opt, tol=1e-8):
            return U_opt, V_opt, W_opt
        
        return None
    
    def try_rank_structured(rank, max_patterns=200, seed=0):
        """Try structured sparsity patterns for rank-r decomposition."""
        nonlocal best_U, best_V, best_W, best_rank
        
        if rank >= best_rank:
            return False
        
        np.random.seed(seed)
        
        # Generate candidate sparsity patterns
        patterns = []
        
        # Pattern type 1: Random sparse patterns
        for _ in range(max_patterns // 3):
            U_mask = np.zeros((n*m, rank), dtype=bool)
            V_mask = np.zeros((m*p, rank), dtype=bool)
            W_mask = np.zeros((p*n, rank), dtype=bool)
            
            # Each row has 1-2 nonzeros
            for i in range(n*m):
                nnz = np.random.choice([1, 2], p=[0.7, 0.3])
                cols = np.random.choice(rank, nnz, replace=False)
                U_mask[i, cols] = True
            
            for i in range(m*p):
                nnz = np.random.choice([1, 2], p=[0.7, 0.3])
                cols = np.random.choice(rank, nnz, replace=False)
                V_mask[i, cols] = True
            
            for i in range(p*n):
                nnz = np.random.choice([1, 2], p=[0.7, 0.3])
                cols = np.random.choice(rank, nnz, replace=False)
                W_mask[i, cols] = True
            
            patterns.append((U_mask, V_mask, W_mask))
        
        # Pattern type 2: Block-structured patterns (Coppersmith-Winograd inspired)
        for _ in range(max_patterns // 3):
            U_mask = np.zeros((n*m, rank), dtype=bool)
            V_mask = np.zeros((m*p, rank), dtype=bool)
            W_mask = np.zeros((p*n, rank), dtype=bool)
            
            # Divide into blocks
            block_size = max(1, rank // 4)
            for i in range(n*m):
                block = i // block_size
                cols = np.arange(block * block_size, min((block + 1) * block_size, rank))
                if len(cols) > 0:
                    # Pick 1-2 from block
                    nnz = min(len(cols), np.random.choice([1, 2]))
                    cols = np.random.choice(cols, nnz, replace=False)
                    U_mask[i, cols] = True
            
            for i in range(m*p):
                block = i // block_size
                cols = np.arange(block * block_size, min((block + 1) * block_size, rank))
                if len(cols) > 0:
                    nnz = min(len(cols), np.random.choice([1, 2]))
                    cols = np.random.choice(cols, nnz, replace=False)
                    V_mask[i, cols] = True
            
            for i in range(p*n):
                block = i // block_size
                cols = np.arange(block * block_size, min((block + 1) * block_size, rank))
                if len(cols) > 0:
                    nnz = min(len(cols), np.random.choice([1, 2]))
                    cols = np.random.choice(cols, nnz, replace=False)
                    W_mask[i, cols] = True
            
            patterns.append((U_mask, V_mask, W_mask))
        
        # Pattern type 3: Diagonal-like patterns
        for _ in range(max_patterns // 3):
            U_mask = np.zeros((n*m, rank), dtype=bool)
            V_mask = np.zeros((m*p, rank), dtype=bool)
            W_mask = np.zeros((p*n, rank), dtype=bool)
            
            shift_u = np.random.randint(0, rank)
            shift_v = np.random.randint(0, rank)
            shift_w = np.random.randint(0, rank)
            
            for i in range(n*m):
                col = (i + shift_u) % rank
                U_mask[i, col] = True
                if rank > 1:
                    col2 = (i + shift_u + 1) % rank
                    if np.random.random() < 0.3:
                        U_mask[i, col2] = True
            
            for i in range(m*p):
                col = (i + shift_v) % rank
                V_mask[i, col] = True
                if rank > 1:
                    col2 = (i + shift_v + 1) % rank
                    if np.random.random() < 0.3:
                        V_mask[i, col2] = True
            
            for i in range(p*n):
                col = (i + shift_w) % rank
                W_mask[i, col] = True
                if rank > 1:
                    col2 = (i + shift_w + 1) % rank
                    if np.random.random() < 0.3:
                        W_mask[i, col2] = True
            
            patterns.append((U_mask, V_mask, W_mask))
        
        # Try each pattern
        for pattern_idx, (U_mask, V_mask, W_mask) in enumerate(patterns):
            if time.time() > deadline:
                return False
            
            # Initialize values only where mask is True
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            W = np.zeros((p*n, rank))
            
            U[U_mask] = np.random.randn(np.sum(U_mask)) * 0.5
            V[V_mask] = np.random.randn(np.sum(V_mask)) * 0.5
            W[W_mask] = np.random.randn(np.sum(W_mask)) * 0.5
            
            # Optimize only free variables
            free_U = np.where(U_mask)
            free_V = np.where(V_mask)
            free_W = np.where(W_mask)
            
            def flatten_free(x):
                return np.concatenate([x[0], x[1], x[2]])
            
            def unflatten_free(x):
                nU = len(free_U[0])
                nV = len(free_V[0])
                U_vals = x[:nU]
                V_vals = x[nU:nU+nV]
                W_vals = x[nU+nV:]
                return U_vals, V_vals, W_vals
            
            def objective(x):
                U_vals, V_vals, W_vals = unflatten_free(x)
                U_curr = np.zeros((n*m, rank))
                V_curr = np.zeros((m*p, rank))
                W_curr = np.zeros((p*n, rank))
                U_curr[free_U] = U_vals
                V_curr[free_V] = V_vals
                W_curr[free_W] = W_vals
                T_approx = np.einsum('ar,br,cr->abc', U_curr, V_curr, W_curr)
                return np.sum((T_approx - T)**2)
            
            x0 = np.concatenate([U[free_U], V[free_V], W[free_W]])
            
            try:
                result = minimize(objective, x0, method='L-BFGS-B',
                                 options={'maxiter': 150, 'ftol': 1e-10, 'gtol': 1e-8})
                
                if result.fun < 1e-6:
                    U_vals, V_vals, W_vals = unflatten_free(result.x)
                    U_opt = np.zeros((n*m, rank))
                    V_opt = np.zeros((m*p, rank))
                    W_opt = np.zeros((p*n, rank))
                    U_opt[free_U] = U_vals
                    V_opt[free_V] = V_vals
                    W_opt[free_W] = W_vals
                    
                    polished = polish(U_opt, V_opt, W_opt)
                    if polished is not None:
                        U_rounded, V_rounded, W_rounded = polished
                        if verify(U_rounded, V_rounded, W_rounded, tol=1e-6):
                            best_U, best_V, best_W = U_rounded.copy(), V_rounded.copy(), W_rounded.copy()
                            best_rank = rank
                            return True
            except Exception:
                pass
        
        return False
    
    def try_rank_random(rank, max_trials=30, seed=0):
        """Try random initialization for rank-r decomposition."""
        nonlocal best_U, best_V, best_W, best_rank
        
        if rank >= best_rank:
            return False
        
        np.random.seed(seed)
        
        for trial in range(max_trials):
            if time.time() > deadline:
                return False
            
            # Random initialization
            U = np.random.randn(n*m, rank) * 0.5
            V = np.random.randn(m*p, rank) * 0.5
            W = np.random.randn(p*n, rank) * 0.5
            
            def flatten_params(U, V, W):
                return np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def unflatten_params(x):
                U = x[:n*m*rank].reshape(n*m, rank)
                V = x[n*m*rank:n*m*rank+m*p*rank].reshape(m*p, rank)
                W = x[n*m*rank+m*p*rank:].reshape(p*n, rank)
                return U, V, W
            
            def objective(x):
                U, V, W = unflatten_params(x)
                T_approx = np.einsum('ar,br,cr->abc', U, V, W)
                return np.sum((T_approx - T)**2)
            
            x0 = flatten_params(U, V, W)
            
            try:
                result = minimize(objective, x0, method='L-BFGS-B',
                                 options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8})
                
                if result.fun < 1e-6:
                    U_opt, V_opt, W_opt = unflatten_params(result.x)
                    
                    polished = polish(U_opt, V_opt, W_opt)
                    if polished is not None:
                        U_rounded, V_rounded, W_rounded = polished
                        if verify(U_rounded, V_rounded, W_rounded, tol=1e-6):
                            best_U, best_V, best_W = U_rounded.copy(), V_rounded.copy(), W_rounded.copy()
                            best_rank = rank
                            return True
            except Exception:
                pass
        
        return False
    
    # Try ranks from lower bound upward
    lower_bound = max(n*p, m*p, n*m)  # = 20
    for rank in range(lower_bound, min(best_rank, lower_bound + 15)):
        if time.time() > deadline:
            break
        
        # First try structured patterns
        if try_rank_structured(rank, max_patterns=100, seed=rank*100):
            break
        
        # Then try random initialization
        if try_rank_random(rank, max_trials=20, seed=rank*100+1):
            break
    
    # Convert to lists
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
