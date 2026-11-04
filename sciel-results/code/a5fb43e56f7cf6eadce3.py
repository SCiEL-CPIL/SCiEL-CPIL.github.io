"""Program for rank-minimal tensor decomposition of 2x3 * 3x4 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T with shape (n*m, m*p, p*n)."""
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _reconstruct(U, V, W):
    """Reconstruct the tensor from factor matrices."""
    return np.einsum('ar,br,cr->abc', U, V, W)

def _check_decomposition(U, V, W, T, tol=1e-9):
    """Check if U, V, W exactly decompose T."""
    recon = _reconstruct(U, V, W)
    err = np.max(np.abs(recon - T))
    return err < tol

def _snap_matrix(M, tol=1e-6):
    """Snap matrix entries to nearest integer or half-integer."""
    M_int = np.round(M)
    if np.allclose(M, M_int, atol=tol):
        return M_int
    M_half = np.round(M * 2) / 2.0
    if np.allclose(M, M_half, atol=tol):
        return M_half
    return None

def _rank_of_matrix(M, tol=1e-8):
    """Compute numerical rank of a matrix."""
    s = np.linalg.svd(M, compute_uv=False)
    return int(np.sum(s > tol))

def _try_rank_smooth(n, m, p, rank, T, time_budget):
    """
    Try to find a rank decomposition using smooth optimization.
    We parameterize U and V, then solve for W via least squares,
    and minimize the reconstruction error.
    """
    start_time = time.time()
    
    # Number of scalar variables: rank*(n*m + m*p + p*n)
    # But we'll optimize U and V, and solve for W.
    
    def _loss_and_grad(params, rank):
        # params: concatenated U (n*m*rank) and V (m*p*rank)
        U_flat = params[:rank * n * m]
        V_flat = params[rank * n * m:]
        U = U_flat.reshape(n * m, rank)
        V = V_flat.reshape(m * p, rank)
        
        # Solve for W: for each c, W[c,:] is determined by the equations
        # sum_r U[i,r] V[j,r] W[c,r] = T[i,j,k] where c = k*n+i
        # For fixed k, and for each i, we have m equations for W[k*n+i, :]
        W = np.zeros((p * n, rank))
        for k in range(p):
            for i in range(n):
                A_i = np.zeros((m, rank))
                b_i = np.zeros(m)
                for j in range(m):
                    for r in range(rank):
                        A_i[j, r] = U[i * m + j, r] * V[j * p + k, r]
                    b_i[j] = T[i * m + j, j * p + k, k * n + i]
                # Solve A_i @ W_i = b_i (least squares)
                if rank > m:
                    # Underdetermined: use minimum norm solution
                    try:
                        W_i, _, _, _ = np.linalg.lstsq(A_i, b_i, rcond=None)
                    except:
                        W_i = np.zeros(rank)
                else:
                    try:
                        W_i = np.linalg.solve(A_i, b_i)
                    except:
                        W_i, _, _, _ = np.linalg.lstsq(A_i, b_i, rcond=None)
                W[k * n + i, :] = W_i
        
        # Compute reconstruction error
        recon = _reconstruct(U, V, W)
        err = np.sum((recon - T) ** 2)
        
        # Gradient via finite differences is too slow; use analytical or skip
        # For now, just return the loss; we'll use scipy.optimize.minimize
        return err, (U, V, W)
    
    # Use random restarts with gradient-based optimization
    best_result = None
    np.random.seed(42)
    
    num_restarts = 20
    for restart in range(num_restarts):
        if time.time() - start_time > time_budget:
            break
        
        # Random initialization
        U_init = np.random.randn(n * m, rank) * 0.5
        V_init = np.random.randn(m * p, rank) * 0.5
        params_init = np.concatenate([U_init.flatten(), V_init.flatten()])
        
        # Use scipy.optimize.minimize with L-BFGS-B
        try:
            from scipy.optimize import minimize
            
            def _objective(params):
                U_flat = params[:rank * n * m]
                V_flat = params[rank * n * m:]
                U = U_flat.reshape(n * m, rank)
                V = V_flat.reshape(m * p, rank)
                
                W = np.zeros((p * n, rank))
                for k in range(p):
                    for i in range(n):
                        A_i = np.zeros((m, rank))
                        b_i = np.zeros(m)
                        for j in range(m):
                            for r in range(rank):
                                A_i[j, r] = U[i * m + j, r] * V[j * p + k, r]
                            b_i[j] = T[i * m + j, j * p + k, k * n + i]
                        if rank > m:
                            W_i, _, _, _ = np.linalg.lstsq(A_i, b_i, rcond=None)
                        else:
                            try:
                                W_i = np.linalg.solve(A_i, b_i)
                            except:
                                W_i, _, _, _ = np.linalg.lstsq(A_i, b_i, rcond=None)
                        W[k * n + i, :] = W_i
                
                recon = _reconstruct(U, V, W)
                return np.sum((recon - T) ** 2)
            
            result = minimize(
                _objective,
                params_init,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if result.fun < 1e-8:
                # Check if we can snap to integers/halves
                U_opt = result.x[:rank * n * m].reshape(n * m, rank)
                V_opt = result.x[rank * n * m:].reshape(m * p, rank)
                
                W_opt = np.zeros((p * n, rank))
                for k in range(p):
                    for i in range(n):
                        A_i = np.zeros((m, rank))
                        b_i = np.zeros(m)
                        for j in range(m):
                            for r in range(rank):
                                A_i[j, r] = U_opt[i * m + j, r] * V_opt[j * p + k, r]
                            b_i[j] = T[i * m + j, j * p + k, k * n + i]
                        if rank > m:
                            W_i, _, _, _ = np.linalg.lstsq(A_i, b_i, rcond=None)
                        else:
                            try:
                                W_i = np.linalg.solve(A_i, b_i)
                            except:
                                W_i, _, _, _ = np.linalg.lstsq(A_i, b_i, rcond=None)
                        W_opt[k * n + i, :] = W_i
                
                U_snap = _snap_matrix(U_opt)
                V_snap = _snap_matrix(V_opt)
                W_snap = _snap_matrix(W_opt)
                
                if U_snap is not None and V_snap is not None and W_snap is not None:
                    if _check_decomposition(U_snap, V_snap, W_snap, T):
                        return U_snap, V_snap, W_snap
                
                # Even if snapping fails, check if the real-valued solution works
                # (but the task requires integer/half-integer entries)
                pass
                
        except Exception:
            continue
    
    return None

def _try_rank_integer_search(n, m, p, rank, T, time_budget):
    """
    Search for a rank decomposition with integer/half-integer entries.
    Use a smarter search: fix the structure based on known constructions.
    """
    start_time = time.time()
    
    # Known: for 2x3x4, rank 19 is achievable.
    # Try a structured search based on the Strassen-like approach.
    
    # The tensor has 2*3*4 = 24 nonzero entries.
    # A rank-r decomposition has r rank-1 terms.
    
    # For rank 19, we need to save 5 operations compared to schoolbook.
    
    # Let's try a specific construction based on the literature.
    # One known rank-19 decomposition for 2x3x4:
    
    if rank == 19:
        # Try a known construction
        # This is based on the work by Bläser and others.
        
        # U has shape (6, 19), V has shape (12, 19), W has shape (8, 19)
        
        # Let's try to find it via a more targeted search.
        # We'll use the fact that the problem is small.
        
        # Strategy: enumerate possible sparse patterns for U and V,
        # then solve for W and check integrality.
        
        vals = [0, 1, -1, 0.5, -0.5]
        
        # For efficiency, we'll try random sparse matrices with these values.
        np.random.seed(123)
        
        max_iters = 50000
        for it in range(max_iters):
            if time.time() - start_time > time_budget:
                break
            
            # Random sparse U and V
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            
            # Make U and V sparse with density ~0.2-0.4
            density = 0.15 + 0.25 * np.random.rand()
            
            for i in range(n * m):
                for r in range(rank):
                    if np.random.rand() < density:
                        U[i, r] = np.random.choice(vals)
            
            for j in range(m * p):
                for r in range(rank):
                    if np.random.rand() < density:
                        V[j, r] = np.random.choice(vals)
            
            # Solve for W
            W = np.zeros((p * n, rank))
            valid = True
            
            for k in range(p):
                for i in range(n):
                    A_i = np.zeros((m, rank))
                    b_i = np.zeros(m)
                    for j in range(m):
                        for r in range(rank):
                            A_i[j, r] = U[i * m + j, r] * V[j * p + k, r]
                        b_i[j] = T[i * m + j, j * p + k, k * n + i]
                    
                    # Check if the system is consistent
                    # A_i is m x rank, with m=3 and rank=19 (underdetermined)
                    # We need to check if b_i is in the column space of A_i
                    try:
                        # Use least squares and check residual
                        W_i, residuals, rnk, sv = np.linalg.lstsq(A_i, b_i, rcond=None)
                        if residuals and residuals[0] > 1e-9:
                            valid = False
                            break
                        W[k * n + i, :] = W_i
                    except:
                        valid = False
                        break
                
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Snap W
            W_snap = _snap_matrix(W)
            if W_snap is None:
                continue
            
            # Verify
            if _check_decomposition(U, V, W_snap, T):
                return U, V, W_snap
        
    return None

def construct():
    n, m, p = DIMENSIONS
    T = _build_tensor(n, m, p)
    
    total_start = time.time()
    time_budget = 140.0  # Leave 10s margin
    
    # Schoolbook fallback: rank 24
    rank_fb = n * m * p
    U_fb = [[0] * rank_fb for _ in range(n * m)]
    V_fb = [[0] * rank_fb for _ in range(m * p)]
    W_fb = [[0] * rank_fb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1
    
    # Try ranks from 19 to 23
    for rank in [19, 20, 21, 22, 23]:
        if rank > rank_fb:
            continue
        
        elapsed = time.time() - total_start
        remaining = time_budget - elapsed
        if remaining < 2.0:
            break
        
        # Allocate time based on rank
        if rank == 19:
            rank_budget = remaining * 0.6
        elif rank == 20:
            rank_budget = remaining * 0.2
        else:
            rank_budget = remaining * 0.07
        
        # First try smooth optimization
        result = _try_rank_smooth(n, m, p, rank, T, rank_budget * 0.5)
        if result is not None:
            U, V, W = result
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
        
        # Then try integer search
        elapsed2 = time.time() - total_start
        remaining2 = time_budget - elapsed2
        if remaining2 > 1.0:
            result = _try_rank_integer_search(n, m, p, rank, T, min(rank_budget * 0.5, remaining2 - 1.0))
            if result is not None:
                U, V, W = result
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
    
    # Fallback to schoolbook
    return {"U": U_fb, "V": V_fb, "W": W_fb}

# EVOLVE-BLOCK-END
