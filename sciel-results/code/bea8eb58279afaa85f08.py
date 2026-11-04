import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 96
    
    start_time = time.time()
    deadline = start_time + 140  # 140s deadline, 10s margin
    
    # Build target tensor T
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Schoolbook fallback
    U_school = np.zeros((n * m, rank_schoolbook))
    V_school = np.zeros((m * p, rank_schoolbook))
    W_school = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    best_result = {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    best_rank = rank_schoolbook
    
    def verify(U, V, W, atol=1e-9):
        """Check if the decomposition exactly reconstructs T."""
        S = np.einsum('ir,jr,kr->ijk', U, V, W)
        return np.allclose(S, T, atol=atol)
    
    def try_als(U, V, W, max_iters=300, tol=1e-12):
        """Alternating Least Squares for tensor decomposition with convergence tracking."""
        U = U.copy().astype(float)
        V = V.copy().astype(float)
        W = W.copy().astype(float)
        rank = U.shape[1]
        
        best_residual = np.inf
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Update U: fix V, W
            M_U = np.einsum('jr,kr->rjk', V, W).reshape(rank, m * p * p * n)
            T_U = T.reshape(n * m, m * p * p * n)
            try:
                U_new = np.linalg.lstsq(M_U.T, T_U.T, rcond=None)[0].T
                U = U_new
            except:
                break
            
            # Update V: fix U, W
            M_V = np.einsum('ir,kr->rik', U, W).reshape(rank, n * m * p * n)
            T_V = T.transpose(1, 0, 2).reshape(m * p, n * m * p * n)
            try:
                V_new = np.linalg.lstsq(M_V.T, T_V.T, rcond=None)[0].T
                V = V_new
            except:
                break
            
            # Update W: fix U, V
            M_W = np.einsum('ir,jr->rij', U, V).reshape(rank, n * m * m * p)
            T_W = T.transpose(2, 0, 1).reshape(p * n, n * m * m * p)
            try:
                W_new = np.linalg.lstsq(M_W.T, T_W.T, rcond=None)[0].T
                W = W_new
            except:
                break
            
            # Check convergence periodically
            if it % 5 == 0:
                S = np.einsum('ir,jr,kr->ijk', U, V, W)
                residual = np.linalg.norm(T - S)
                if residual < best_residual:
                    best_residual = residual
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                if residual < tol:
                    break
        
        # Final check
        S = np.einsum('ir,jr,kr->ijk', U, V, W)
        if np.allclose(S, T, atol=1e-8):
            return U, V, W
        # Also check the best found
        S_best = np.einsum('ir,jr,kr->ijk', best_U, best_V, best_W)
        if np.allclose(S_best, T, atol=1e-8):
            return best_U, best_V, best_W
        return None
    
    def try_hogwild(U, V, W, max_iters=200, tol=1e-12):
        """Hogwild-style asynchronous ALS with random coordinate updates."""
        U = U.copy().astype(float)
        V = V.copy().astype(float)
        W = W.copy().astype(float)
        rank = U.shape[1]
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Randomly pick which factor to update
            factor = np.random.randint(3)
            
            if factor == 0:
                # Update U
                M_U = np.einsum('jr,kr->rjk', V, W).reshape(rank, m * p * p * n)
                T_U = T.reshape(n * m, m * p * p * n)
                try:
                    U = np.linalg.lstsq(M_U.T, T_U.T, rcond=None)[0].T
                except:
                    pass
            elif factor == 1:
                # Update V
                M_V = np.einsum('ir,kr->rik', U, W).reshape(rank, n * m * p * n)
                T_V = T.transpose(1, 0, 2).reshape(m * p, n * m * p * n)
                try:
                    V = np.linalg.lstsq(M_V.T, T_V.T, rcond=None)[0].T
                except:
                    pass
            else:
                # Update W
                M_W = np.einsum('ir,jr->rij', U, V).reshape(rank, n * m * m * p)
                T_W = T.transpose(2, 0, 1).reshape(p * n, n * m * m * p)
                try:
                    W = np.linalg.lstsq(M_W.T, T_W.T, rcond=None)[0].T
                except:
                    pass
            
            if it % 10 == 0:
                S = np.einsum('ir,jr,kr->ijk', U, V, W)
                if np.allclose(S, T, atol=1e-8):
                    return U, V, W
        
        S = np.einsum('ir,jr,kr->ijk', U, V, W)
        if np.allclose(S, T, atol=1e-8):
            return U, V, W
        return None
    
    # Try various ranks with multiple initialization strategies
    # Start from lower ranks and work up
    target_ranks = [48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92]
    
    for rank in target_ranks:
        if time.time() > deadline:
            break
        
        if rank >= best_rank:
            continue
        
        # Multiple initialization strategies
        n_attempts = 40
        
        for attempt in range(n_attempts):
            if time.time() > deadline:
                break
            
            np.random.seed(1000 + rank * 100 + attempt)
            
            # Strategy 1: Random dense
            if attempt % 4 == 0:
                U = np.random.randn(n * m, rank) * 0.1
                V = np.random.randn(m * p, rank) * 0.1
                W = np.random.randn(p * n, rank) * 0.1
                
            # Strategy 2: Sparse with random signs
            elif attempt % 4 == 1:
                U = np.zeros((n * m, rank))
                V = np.zeros((m * p, rank))
                W = np.zeros((p * n, rank))
                
                for r in range(rank):
                    nnz_u = max(1, int(np.random.uniform(0.3, 0.8) * n * m))
                    nnz_v = max(1, int(np.random.uniform(0.3, 0.8) * m * p))
                    nnz_w = max(1, int(np.random.uniform(0.3, 0.8) * p * n))
                    
                    idx_u = np.random.choice(n * m, nnz_u, replace=False)
                    idx_v = np.random.choice(m * p, nnz_v, replace=False)
                    idx_w = np.random.choice(p * n, nnz_w, replace=False)
                    
                    U[idx_u, r] = np.random.choice([-1, 1], size=nnz_u)
                    V[idx_v, r] = np.random.choice([-1, 1], size=nnz_v)
                    W[idx_w, r] = np.random.choice([-1, 1], size=nnz_w)
                    
            # Strategy 3: Structured based on block structure
            elif attempt % 4 == 2:
                U = np.zeros((n * m, rank))
                V = np.zeros((m * p, rank))
                W = np.zeros((p * n, rank))
                
                # Use the fact that n=3, m=4, p=8
                # Try to exploit the cyclic structure
                for r in range(rank):
                    # Random phase
                    phase = 2 * np.pi * np.random.rand()
                    
                    # U has structure related to i (row of A)
                    for i in range(n):
                        for j in range(m):
                            U[i * m + j, r] = np.cos(phase * (i + j / m) + np.random.randn() * 0.5)
                    
                    # V has structure related to j (col of A, row of B)
                    for j in range(m):
                        for k in range(p):
                            V[j * p + k, r] = np.cos(phase * (j + k / p) + np.random.randn() * 0.5)
                    
                    # W has structure related to k (col of B) and i (row of output)
                    for k in range(p):
                        for i in range(n):
                            W[k * n + i, r] = np.cos(phase * (k + i / n) + np.random.randn() * 0.5)
                    
                    # Scale
                    scale = np.random.uniform(0.5, 2.0)
                    U[:, r] *= scale
                    V[:, r] *= scale
                    W[:, r] *= scale
                    
            # Strategy 4: Very sparse
            else:
                U = np.zeros((n * m, rank))
                V = np.zeros((m * p, rank))
                W = np.zeros((p * n, rank))
                
                for r in range(rank):
                    # Each term touches very few elements
                    nnz = max(2, rank // 10)
                    idx_u = np.random.choice(n * m, min(nnz, n * m), replace=False)
                    idx_v = np.random.choice(m * p, min(nnz, m * p), replace=False)
                    idx_w = np.random.choice(p * n, min(nnz, p * n), replace=False)
                    
                    U[idx_u, r] = np.random.choice([-1, 1], size=len(idx_u))
                    V[idx_v, r] = np.random.choice([-1, 1], size=len(idx_v))
                    W[idx_w, r] = np.random.choice([-1, 1], size=len(idx_w))
                
                # Add some noise to break symmetry
                U += np.random.randn(n * m, rank) * 0.01
                V += np.random.randn(m * p, rank) * 0.01
                W += np.random.randn(p * n, rank) * 0.01
            
            # Try ALS
            result = try_als(U, V, W, max_iters=150)
            if result is not None:
                U_f, V_f, W_f = result
                
                # Try to round to integers or half-integers
                # First try integers
                U_r = np.round(U_f)
                V_r = np.round(V_f)
                W_r = np.round(W_f)
                
                if verify(U_r, V_r, W_r, atol=1e-8):
                    return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                
                # Try half-integers
                U_r = np.round(U_f * 2) / 2
                V_r = np.round(V_f * 2) / 2
                W_r = np.round(W_f * 2) / 2
                
                if verify(U_r, V_r, W_r, atol=1e-8):
                    return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                
                # Return unrounded if close enough
                if verify(U_f, V_f, W_f, atol=1e-6):
                    return {"U": U_f.tolist(), "V": V_f.tolist(), "W": W_f.tolist()}
        
        # If we haven't found a solution for this rank, try hogwild with best initialization
        if time.time() > deadline:
            break
        
        # Try a few more attempts with hogwild
        for attempt in range(5):
            if time.time() > deadline:
                break
            
            np.random.seed(5000 + rank * 100 + attempt)
            U = np.random.randn(n * m, rank) * 0.1
            V = np.random.randn(m * p, rank) * 0.1
            W = np.random.randn(p * n, rank) * 0.1
            
            result = try_hogwild(U, V, W, max_iters=100)
            if result is not None:
                U_f, V_f, W_f = result
                U_r = np.round(U_f * 2) / 2
                V_r = np.round(V_f * 2) / 2
                W_r = np.round(W_f * 2) / 2
                if verify(U_r, V_r, W_r, atol=1e-8):
                    return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                if verify(U_f, V_f, W_f, atol=1e-6):
                    return {"U": U_f.tolist(), "V": V_f.tolist(), "W": W_f.tolist()}
    
    return best_result
# EVOLVE-BLOCK-END
