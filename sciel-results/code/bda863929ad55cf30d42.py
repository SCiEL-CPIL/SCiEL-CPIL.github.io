import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    start_time = time.time()
    deadline = start_time + 115
    
    # Construct target tensor T with shape (n*m, m*p, p*n)
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Schoolbook decomposition as fallback
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
    
    best_rank = rank_schoolbook
    best_result = {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
    
    def verify(U, V, W, tol=1e-9):
        S = np.einsum('ir,jr,kr->ijk', U, V, W)
        return np.allclose(S, T, atol=tol)
    
    def round_to_half_int(U, V, W):
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        if verify(U_r, V_r, W_r, tol=1e-9):
            return U_r, V_r, W_r
        return None
    
    def try_als_fast(rank, U_init, V_init, W_init, max_iters=30, tol=1e-10):
        U = U_init.copy()
        V = V_init.copy()
        W = W_init.copy()
        
        for it in range(max_iters):
            if time.time() > deadline:
                return None
            
            # Update U
            M_U = np.einsum('jr,kr->rjk', V, W).reshape(rank, m * p * p * n)
            T_flat_U = T.reshape(n * m, m * p * p * n)
            try:
                U_new = np.linalg.lstsq(M_U.T, T_flat_U.T, rcond=None)[0].T
                U = U_new
            except:
                return None
            
            # Update V
            M_V = np.einsum('ir,kr->rik', U, W).reshape(rank, n * m * p * n)
            T_perm = np.transpose(T, (1, 0, 2))
            T_flat_V = T_perm.reshape(m * p, n * m * p * n)
            try:
                V_new = np.linalg.lstsq(M_V.T, T_flat_V.T, rcond=None)[0].T
                V = V_new
            except:
                return None
            
            # Update W
            M_W = np.einsum('ir,jr->rij', U, V).reshape(rank, n * m * m * p)
            T_perm_w = np.transpose(T, (2, 0, 1))
            T_flat_W = T_perm_w.reshape(p * n, n * m * m * p)
            try:
                W_new = np.linalg.lstsq(M_W.T, T_flat_W.T, rcond=None)[0].T
                W = W_new
            except:
                return None
            
            if verify(U, V, W, tol=1e-12):
                return (U, V, W)
        
        if verify(U, V, W, tol=1e-10):
            return (U, V, W)
        return None
    
    def try_structured_decomposition(rank):
        """Try structured decompositions based on matrix multiplication structure."""
        if time.time() > deadline:
            return None
        
        # Strategy: exploit the chain structure i->j->k->i
        # Try to find rank-r decomposition where factors have block structure
        
        for seed in range(15):
            if time.time() > deadline:
                break
            
            np.random.seed(seed * 7 + 13)
            
            # Random initialization with structure
            U_init = np.random.randn(n * m, rank) * 0.5
            V_init = np.random.randn(m * p, rank) * 0.5
            W_init = np.random.randn(p * n, rank) * 0.5
            
            result = try_als_fast(rank, U_init, V_init, W_init, max_iters=20)
            if result is not None:
                U, V, W = result
                rounded = round_to_half_int(U, V, W)
                if rounded is not None:
                    return rounded
                return (U, V, W)
        
        return None
    
    def try_kronecker_like(rank):
        """Try Kronecker-product-like structured decompositions."""
        if time.time() > deadline:
            return None
        
        # For 3x4x8, try to decompose using tensor products
        # 3 = 3, 4 = 2*2, 8 = 2*2*2
        # Try rank = 3 * 4 * 8 / factor
        
        for seed in range(10):
            if time.time() > deadline:
                break
            
            np.random.seed(seed * 11 + 5)
            
            U_init = np.random.randn(n * m, rank) * 0.3
            V_init = np.random.randn(m * p, rank) * 0.3
            W_init = np.random.randn(p * n, rank) * 0.3
            
            result = try_als_fast(rank, U_init, V_init, W_init, max_iters=25)
            if result is not None:
                U, V, W = result
                rounded = round_to_half_int(U, V, W)
                if rounded is not None:
                    return rounded
                return (U, V, W)
        
        return None
    
    # Try ranks in increasing order from lower bounds
    # Lower bound: max(12, 32, 24) = 32
    # Try: 32, 36, 40, 44, 48, 52, 56, 60, 64, 72, 80, 88, 96
    
    candidate_ranks = []
    for r in range(32, 97):
        if time.time() > deadline:
            break
        candidate_ranks.append(r)
    
    # Prioritize ranks that are multiples of small numbers or have special structure
    priority_ranks = [32, 36, 40, 48, 56, 64, 72, 80, 88, 96]
    candidate_ranks = priority_ranks + [r for r in candidate_ranks if r not in priority_ranks]
    
    for rank in candidate_ranks:
        if time.time() > deadline:
            break
        
        if rank >= best_rank:
            continue
        
        # Try multiple strategies
        for strategy in ['structured', 'kronecker']:
            if time.time() > deadline:
                break
            
            if strategy == 'structured':
                result = try_structured_decomposition(rank)
            else:
                result = try_kronecker_like(rank)
            
            if result is not None:
                U, V, W = result
                if verify(U, V, W, tol=1e-9):
                    rounded = round_to_half_int(U, V, W)
                    if rounded is not None:
                        U, V, W = rounded
                    best_rank = rank
                    best_result = {
                        "U": U.tolist(),
                        "V": V.tolist(),
                        "W": W.tolist()
                    }
                    break  # Found a valid decomposition at this rank
    
    return best_result
