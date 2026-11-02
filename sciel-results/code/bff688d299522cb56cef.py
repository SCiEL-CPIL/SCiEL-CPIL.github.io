"""Matrix multiplication 3x3 x 3x7 rank-minimal decomposition search."""
import time
import numpy as np

DIMENSIONS = [3, 3, 7]

def _build_tensor():
    """Build the matrix multiplication tensor T[i,j,k] = 1 for all valid (i,j,k).
    
    Tensor shape: (n*m, m*p, p*n) in the flattened index convention.
    T[a, b, c] = 1 where a = i*m+j, b = j*p+k, c = k*n+i
    This is the standard matrix multiplication tensor.
    """
    n, m, p = DIMENSIONS
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _verify_decomposition(U, V, W, T, tol=1e-9):
    """Check if U @ diag @ V^T @ diag @ W^T reconstructs T."""
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(T_recon, T, atol=tol, rtol=tol)

def _polish_decomposition(U, V, W, T, max_iters=200, tol=1e-12, time_limit=5.0):
    """Polish a near-solution using alternating least squares (ALS)."""
    start_time = time.time()
    rank = U.shape[1]
    
    U = U.copy().astype(np.float64)
    V = V.copy().astype(np.float64)
    W = W.copy().astype(np.float64)
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        # Check convergence
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        error = np.linalg.norm(T_recon - T)
        if error < tol:
            return U, V, W, True
        
        # ALS: alternate between fixing two factors and solving for the third
        # We'll do 3 sweeps per iteration
        
        # Fix V and W, solve for U
        for r in range(rank):
            # Compute residual without factor r
            T_recon = np.einsum('ar,br,cr->abc', U, V, W)
            T_recon -= np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
            residual = T - T_recon
            
            # For each (a, c), we want U[a, r] * V[b, r] * W[c, r] to fit residual[a, b, c]
            # Sum over b: U[a, r] * sum_b (residual[a, b, c] * V[b, r]) = W[c, r] * U[a, r]
            # This is a linear system: for each a, U[a, r] * (sum_b residual[a,b,c] * V[b,r] * W[c,r]) = 0
            # Actually, let's use a simpler approach:
            # The optimal U[:, r] satisfies:
            # U[:, r] = (V[:, r]^T * W[:, r]^T * residual_reshaped) / (V[:, r]^T * W[:, r]^T * V[:, r] * W[:, r])
            # But this is getting complicated. Let's use a gradient-based update instead.
            
            # Compute gradient for U[:, r]
            grad_U_r = -2 * np.einsum('abc,br,cr->a', residual, V[:, r], W[:, r])
            
            # Compute Hessian approximation (diagonal)
            hess_diag = 2 * np.einsum('b,c->', V[:, r], V[:, r]) * np.einsum('c,c->', W[:, r], W[:, r])
            
            # Update
            U[:, r] -= grad_U_r / (hess_diag + 1e-8)
        
        # Fix U and W, solve for V
        for r in range(rank):
            T_recon = np.einsum('ar,br,cr->abc', U, V, W)
            T_recon -= np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
            residual = T - T_recon
            
            grad_V_r = -2 * np.einsum('abc,ar,cr->b', residual, U[:, r], W[:, r])
            hess_diag = 2 * np.einsum('a,a->', U[:, r], U[:, r]) * np.einsum('c,c->', W[:, r], W[:, r])
            
            V[:, r] -= grad_V_r / (hess_diag + 1e-8)
        
        # Fix U and V, solve for W
        for r in range(rank):
            T_recon = np.einsum('ar,br,cr->abc', U, V, W)
            T_recon -= np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
            residual = T - T_recon
            
            grad_W_r = -2 * np.einsum('abc,ar,br->c', residual, U[:, r], V[:, r])
            hess_diag = 2 * np.einsum('a,a->', U[:, r], U[:, r]) * np.einsum('b,b->', V[:, r], V[:, r])
            
            W[:, r] -= grad_W_r / (hess_diag + 1e-8)
    
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    error = np.linalg.norm(T_recon - T)
    return U, V, W, error < 1e-9

def _try_random_rank(rank, T, max_iterations=500, time_limit=10.0):
    """Try to find a rank-`rank` decomposition using random initialization and Adam optimization."""
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    start_time = time.time()
    best_error = float('inf')
    best_U = best_V = best_W = None
    
    # Multiple initialization strategies
    init_strategies = ['random', 'random_large', 'structured']
    
    for strategy in init_strategies:
        if time.time() - start_time > time_limit:
            break
        
        # Initialize parameters based on strategy
        if strategy == 'random':
            U = np.random.randn(nm, rank) * 0.1
            V = np.random.randn(mp, rank) * 0.1
            W = np.random.randn(pn, rank) * 0.1
        elif strategy == 'random_large':
            U = np.random.randn(nm, rank) * 0.5
            V = np.random.randn(mp, rank) * 0.5
            W = np.random.randn(pn, rank) * 0.5
        elif strategy == 'structured':
            # Initialize with some structure based on the tensor
            U = np.zeros((nm, rank))
            V = np.zeros((mp, rank))
            W = np.zeros((pn, rank))
            
            # Try to initialize based on the support of the tensor
            for r in range(rank):
                # Pick random indices
                idx_a = np.random.choice(nm, size=min(nm, 3), replace=False)
                idx_b = np.random.choice(mp, size=min(mp, 3), replace=False)
                idx_c = np.random.choice(pn, size=min(pn, 3), replace=False)
                U[idx_a, r] = 1.0
                V[idx_b, r] = 1.0
                W[idx_c, r] = 1.0
        
        # Adam optimizer
        mU = np.zeros_like(U)
        mV = np.zeros_like(V)
        mW = np.zeros_like(W)
        vU = np.zeros_like(U)
        vV = np.zeros_like(V)
        vW = np.zeros_like(W)
        
        b1 = 0.9
        b2 = 0.999
        eps = 1e-8
        lr = 0.01
        
        for sub_it in range(300):
            if time.time() - start_time > time_limit:
                break
            
            # Compute current reconstruction
            T_recon = np.einsum('ar,br,cr->abc', U, V, W)
            error = np.linalg.norm(T_recon - T)
            if error < best_error:
                best_error = error
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
                if error < 1e-9:
                    # Try to polish
                    U_pol, V_pol, W_pol, success = _polish_decomposition(U, V, W, T, time_limit=min(3.0, time_limit - (time.time() - start_time)))
                    if success:
                        return U_pol, V_pol, W_pol, True
                    return U.copy(), V.copy(), W.copy(), True
            
            # Compute gradients
            residual = T - T_recon
            grad_U = -2 * np.einsum('abc,br,cr->ar', residual, V, W)
            grad_V = -2 * np.einsum('abc,ar,cr->br', residual, U, W)
            grad_W = -2 * np.einsum('abc,ar,br->cr', residual, U, V)
            
            # Adam update
            t = sub_it + 1
            mU = b1 * mU + (1 - b1) * grad_U
            mV = b1 * mV + (1 - b1) * grad_V
            mW = b1 * mW + (1 - b1) * grad_W
            vU = b2 * vU + (1 - b2) * (grad_U ** 2)
            vV = b2 * vV + (1 - b2) * (grad_V ** 2)
            vW = b2 * vW + (1 - b2) * (grad_W ** 2)
            
            mU_hat = mU / (1 - b1 ** t)
            mV_hat = mV / (1 - b1 ** t)
            mW_hat = mW / (1 - b1 ** t)
            vU_hat = vU / (1 - b2 ** t)
            vV_hat = vV / (1 - b2 ** t)
            vW_hat = vW / (1 - b2 ** t)
            
            U += lr * mU_hat / (np.sqrt(vU_hat) + eps)
            V += lr * mV_hat / (np.sqrt(vV_hat) + eps)
            W += lr * mW_hat / (np.sqrt(vW_hat) + eps)
    
    # Check best and try to polish if close
    if best_U is not None:
        T_recon = np.einsum('ar,br,cr->abc', best_U, best_V, best_W)
        error = np.linalg.norm(T_recon - T)
        if error < 1e-6:
            # Try to polish
            U_pol, V_pol, W_pol, success = _polish_decomposition(best_U, best_V, best_W, T, time_limit=min(5.0, time_limit - (time.time() - start_time)))
            if success:
                return U_pol, V_pol, W_pol, True
        if error < 1e-9:
            return best_U, best_V, best_W, True
    
    return None, None, None, False

def _schoolbook():
    """Return the schoolbook (trivial) decomposition of rank n*m*p."""
    n, m, p = DIMENSIONS
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

def _to_list_of_lists(arr):
    """Convert numpy array to list of lists of floats."""
    return arr.tolist()

def construct():
    """Find a rank-minimal exact tensor decomposition for 3x3 * 3x7 matrix multiplication."""
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 63
    
    # Start with schoolbook as fallback
    fallback = _schoolbook()
    
    # Time budget management
    start_time = time.time()
    total_budget = 150.0
    margin = 15.0
    effective_budget = total_budget - margin
    
    # Build tensor
    T = _build_tensor()
    
    # Theoretical lower bound: max(n*m, m*p, p*n) = max(9, 21, 21) = 21
    lower_bound = max(n * m, m * p, p * n)
    
    best_result = None
    best_rank = schoolbook_rank
    
    # Try candidate ranks with adaptive time allocation
    candidate_ranks = sorted(set([
        lower_bound, lower_bound + 1, lower_bound + 2, lower_bound + 3,
        lower_bound + 5, lower_bound + 7, lower_bound + 10,
        25, 27, 30, 35, 40
    ]))
    candidate_ranks = [r for r in candidate_ranks if lower_bound <= r < schoolbook_rank]
    
    for rank in candidate_ranks:
        if time.time() - start_time > effective_budget:
            break
        
        remaining = effective_budget - (time.time() - start_time)
        if remaining < 5.0:
            break
        
        # Give more time to lower ranks
        rank_factor = max(0.5, 1.0 - (rank - lower_bound) * 0.05)
        time_per_rank = min(15.0, remaining * 0.3 * rank_factor)
        time_per_rank = max(2.0, time_per_rank)
        
        result = _try_random_rank(rank, T, max_iterations=300, time_limit=time_per_rank)
        if result[3]:
            U, V, W, _ = result
            best_rank = rank
            best_result = {"U": _to_list_of_lists(U), "V": _to_list_of_lists(V), "W": _to_list_of_lists(W)}
            
            # If we found a low rank, try to find even lower
            if rank <= lower_bound + 3:
                for lower_rank in range(lower_bound, rank):
                    if time.time() - start_time > effective_budget - 5.0:
                        break
                    result_lower = _try_random_rank(lower_rank, T, max_iterations=200, time_limit=5.0)
                    if result_lower[3]:
                        U_l, V_l, W_l, _ = result_lower
                        best_rank = lower_rank
                        best_result = {"U": _to_list_of_lists(U_l), "V": _to_list_of_lists(V_l), "W": _to_list_of_lists(W_l)}
                        break
            break
    
    if best_result is not None:
        return best_result
    
    return fallback
