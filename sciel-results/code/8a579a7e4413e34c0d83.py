"""Program for matrix multiplication 4x4 * 4x9 tensor rank search.

Strategy:
1. Build the schoolbook decomposition (rank 144) as a guaranteed valid baseline.
2. Attempt to find a lower-rank decomposition via iterative rank-reduction:
   - Start with a random initialization of factor matrices of a target rank.
   - Use ALS (Alternating Least Squares) to refine the decomposition.
   - If the residual norm falls below a tolerance, accept the lower rank.
3. Return the best (lowest-rank) valid decomposition found.

The trace convention for W is W[k*n + i, r], i.e., W is indexed by (output_col, output_row).
"""

import time
import numpy as np

DIMENSIONS = [4, 4, 9]
_DEADLINE_FRACTION = 0.90
_WALL_BUDGET_S = 150.0


# EVOLVE-BLOCK-START
def construct():
    """Return {'U','V','W'} exact decomposition of (4x4)*(4x9) tensor.
    
    Attempts to find a rank-minimal exact tensor decomposition.
    Falls back to schoolbook (rank 144) if no lower rank is found.
    """
    _t0 = time.time()
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    def elapsed():
        return time.time() - _t0
    
    def remaining():
        return _WALL_BUDGET_S * _DEADLINE_FRACTION - elapsed()
    
    # Build the multiplication tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
    # All other entries are 0.
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    def build_schoolbook():
        """Build the schoolbook decomposition with rank n*m*p = 144."""
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=np.float64)
        V = np.zeros((m * p, rank), dtype=np.float64)
        W = np.zeros((p * n, rank), dtype=np.float64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W
    
    def verify(U, V, W, tol=1e-8):
        """Check if the decomposition is exact."""
        # Reconstruct tensor and compare
        # T_reconstructed[i, j, k] = sum_r U[i,r] * V[j,r] * W[k,r]
        # For efficiency, we can check via matrix operations
        # But for small sizes, direct computation is fine.
        # Actually, let's check by computing the implied multiplication
        # For random A and B, check if A*B matches.
        # But that's probabilistic. Let's do exact tensor comparison.
        
        # Efficient check: for each r, the outer product U[:,r] ⊗ V[:,r] ⊗ W[:,r]
        # Sum over r and compare to T.
        
        # For small dimensions, just do it directly.
        T_rec = np.zeros_like(T)
        for r in range(U.shape[1]):
            T_rec += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
        
        residual = np.max(np.abs(T_rec - T))
        return residual < tol, residual
    
    def als_decompose(target_rank, max_iters=50, tol=1e-10, seed=42):
        """Try to find a rank-target_rank decomposition using ALS.
        
        Returns (U, V, W, success) where success is True if residual < tol.
        """
        rng = np.random.RandomState(seed)
        
        # Initialize
        U = rng.randn(n * m, target_rank) * 0.1
        V = rng.randn(m * p, target_rank) * 0.1
        W = np.ones((p * n, target_rank))  # Start with ones for stability
        
        best_residual = np.inf
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iters):
            if remaining() < 5.0:
                break
            
            # Update U: fix V and W
            # For each i, U[i, :] = (V^T * W^T flattened appropriately) ... 
            # Actually, the ALS update for U given V, W:
            # T[i, :, :] = sum_r U[i, r] * V[:, r] * W[:, r]
            # So for fixed i, we need to solve for U[i, :] such that
            # sum_r U[i, r] * (V[:, r] * W[:, r]) = T[i, :, :]
            # This is a linear system: U[i, :] @ M = T[i, :, :].ravel()
            # where M[r, (j,k)] = V[j, r] * W[k, r]
            
            # Build M: shape (target_rank, m*p * p*n)
            M = np.zeros((target_rank, m * p * p * n))
            for r in range(target_rank):
                M[r, :] = (V[:, r][:, np.newaxis] * W[:, r][np.newaxis, :]).ravel()
            
            # Solve for each i
            for i in range(n * m):
                rhs = T[i, :, :].ravel()
                # Solve U[i, :] @ M = rhs
                # i.e., M^T @ U[i, :]^T = rhs
                try:
                    sol, _, _, _ = np.linalg.lstsq(M.T, rhs, rcond=None)
                    U[i, :] = sol
                except:
                    pass
            
            # Update V similarly
            M2 = np.zeros((target_rank, n * m * p * n))
            for r in range(target_rank):
                M2[r, :] = (U[:, r][:, np.newaxis] * W[:, r][np.newaxis, :]).ravel()
            
            for j in range(m * p):
                rhs = T[:, j, :].ravel()
                try:
                    sol, _, _, _ = np.linalg.lstsq(M2.T, rhs, rcond=None)
                    V[j, :] = sol
                except:
                    pass
            
            # Update W
            M3 = np.zeros((target_rank, n * m * m * p))
            for r in range(target_rank):
                M3[r, :] = (U[:, r][:, np.newaxis] * V[:, r][np.newaxis, :]).ravel()
            
            for k in range(p * n):
                rhs = T[:, :, k].ravel()
                try:
                    sol, _, _, _ = np.linalg.lstsq(M3.T, rhs, rcond=None)
                    W[k, :] = sol
                except:
                    pass
            
            # Check convergence
            success, residual = verify(U, V, W, tol=1e-6)
            if residual < best_residual:
                best_residual = residual
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if success:
                return U, V, W, True
        
        return best_U, best_V, best_W, best_residual < tol
    
    # First, get the schoolbook solution
    U_school, V_school, W_school = build_schoolbook()
    
    # Try to find lower-rank decompositions
    best_rank = n * m * p
    best_factors = (U_school, V_school, W_school)
    
    # Try ranks from low to high
    # Theoretical lower bounds: rank >= max(n*m, m*p, p*n) = max(16, 36, 36) = 36
    # Also rank >= max(n, m, p) = 9
    # Let's try a few candidate ranks
    
    candidate_ranks = [36, 40, 44, 48, 52, 56, 60, 64, 72, 80, 96, 108, 120, 132]
    
    for target_rank in candidate_ranks:
        if remaining() < 10.0:
            break
        
        if target_rank >= best_rank:
            continue
        
        # Try a few different seeds
        for seed in [42, 123, 777, 2024]:
            if remaining() < 5.0:
                break
            
            U, V, W, success = als_decompose(target_rank, max_iters=30, tol=1e-10, seed=seed)
            
            if success:
                # Verify with stricter tolerance
                success_strict, _ = verify(U, V, W, tol=1e-8)
                if success_strict:
                    best_rank = target_rank
                    best_factors = (U, V, W)
                    break
            else:
                # Check if the residual is close enough
                _, residual = verify(U, V, W, tol=1e-3)
                if residual < 1e-6:
                    # Do a polish step
                    U, V, W, success = als_decompose(target_rank, max_iters=50, tol=1e-12, seed=seed)
                    if success:
                        success_strict, _ = verify(U, V, W, tol=1e-8)
                        if success_strict:
                            best_rank = target_rank
                            best_factors = (U, V, W)
                            break
    
    U, V, W = best_factors
    
    # Convert to lists as required by the interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
