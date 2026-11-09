"""Find a low-rank exact tensor decomposition for 2x4 * 4x6 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 4, 6]

def _build_target(n, m, p):
    nm, mp, pn = n*m, m*p, p*n
    target = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i*m+j, j*p+k, k*n+i] = 1.0
    return target

def _als_solve(M, target_flat, rank):
    """Solve for U such that U @ M.T ≈ target_flat (row-wise least squares).
    M: (mp*pn, rank), target_flat: (nm, mp*pn)
    Returns U: (nm, rank)
    """
    try:
        MMt = M.T @ M
        M_t = M.T @ target_flat.T
        U_new = np.linalg.solve(MMt, M_t).T
        return U_new
    except np.linalg.LinAlgError:
        try:
            U_new = np.linalg.lstsq(M, target_flat.T, rcond=None)[0].T
            return U_new
        except:
            return None

def _als_step(U, V, W, target, nm, mp, pn, rank, lam=1e-8):
    """One ALS iteration. Returns (U_new, V_new, W_new) or None on failure."""
    # Update U
    M = (V[:, :, None] * W[None, :, :]).reshape(mp * pn, rank)
    target_2d = target.reshape(nm, mp * pn)
    U_new = _als_solve(M, target_2d, rank)
    if U_new is None:
        return None
    
    # Update V
    N = (U_new[:, :, None] * W[None, :, :]).reshape(nm * pn, rank)
    target_2d_v = target.transpose(1, 0, 2).reshape(mp, nm * pn)
    V_new = _als_solve(N, target_2d_v, rank)
    if V_new is None:
        return None
    
    # Update W
    P = (U_new[:, :, None] * V_new[None, :, :]).reshape(nm * mp, rank)
    target_2d_w = target.transpose(2, 0, 1).reshape(pn, nm * mp)
    W_new = _als_solve(P, target_2d_w, rank)
    if W_new is None:
        return None
    
    return U_new, V_new, W_new

def _compute_error(U, V, W, target):
    decomposed = np.einsum('ar,br,cr->abc', U, V, W)
    err = np.linalg.norm(decomposed - target) / np.linalg.norm(target)
    return err

def _gradient_polish(U, V, W, target, nm, mp, pn, rank, lr=0.01, max_iter=50, tol=1e-12):
    """Polish solution using gradient descent on the objective ||T - sum_r U_r V_r W_r||^2."""
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    for it in range(max_iter):
        # Compute current decomposition
        decomposed = np.einsum('ar,br,cr->abc', U, V, W)
        diff = decomposed - target  # (nm, mp, pn)
        
        # Gradients
        # dL/dU[a,r] = sum_{b,c} diff[a,b,c] * V[b,r] * W[c,r]
        grad_U = np.einsum('abc,br,cr->ar', diff, V, W)
        grad_V = np.einsum('abc,ar,cr->br', diff, U, W)
        grad_W = np.einsum('abc,ar,br->cr', diff, U, V)
        
        # Update
        U -= lr * grad_U
        V -= lr * grad_V
        W -= lr * grad_W
        
        err = np.linalg.norm(decomposed - target)
        if err < tol:
            break
    
    return U, V, W

def _try_rank(rank, target, nm, mp, pn, n, m, p, rng, max_attempts, deadline):
    """Try to find a decomposition of given rank using ALS with various initializations."""
    for attempt in range(max_attempts):
        if time.time() > deadline:
            return None
        
        # Different initialization strategies
        if attempt < max_attempts // 3:
            # Random Gaussian
            U = rng.randn(nm, rank)
            V = rng.randn(mp, rank)
            W = rng.randn(pn, rank)
        elif attempt < 2 * max_attempts // 3:
            # Sparse random (some entries zero)
            U = rng.randn(nm, rank)
            V = rng.randn(mp, rank)
            W = rng.randn(pn, rank)
            # Zero out ~30% of entries
            mask_u = rng.rand(nm, rank) < 0.3
            mask_v = rng.rand(mp, rank) < 0.3
            mask_w = rng.rand(pn, rank) < 0.3
            U[mask_u] = 0
            V[mask_v] = 0
            W[mask_w] = 0
            # Ensure no column is all zero
            for r in range(rank):
                if np.allclose(U[:, r], 0):
                    U[rng.randint(nm), r] = rng.randn()
                if np.allclose(V[:, r], 0):
                    V[rng.randint(mp), r] = rng.randn()
                if np.allclose(W[:, r], 0):
                    W[rng.randint(pn), r] = rng.randn()
        else:
            # Perturbed schoolbook-like structure
            U = np.zeros((nm, rank))
            V = np.zeros((mp, rank))
            W = np.zeros((pn, rank))
            # Initialize with random permutations of schoolbook patterns
            for r in range(rank):
                i = rng.randint(n)
                j = rng.randint(m)
                k = rng.randint(p)
                U[i*m+j, r] = 1.0 + 0.1 * rng.randn()
                V[j*p+k, r] = 1.0 + 0.1 * rng.randn()
                W[k*n+i, r] = 1.0 + 0.1 * rng.randn()
            # Add small random noise to all entries
            U += 0.01 * rng.randn(nm, rank)
            V += 0.01 * rng.randn(mp, rank)
            W += 0.01 * rng.randn(pn, rank)
        
        # Run ALS
        best_err = np.inf
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        stuck_count = 0
        for it in range(300):
            if time.time() > deadline:
                break
            
            result = _als_step(U, V, W, target, nm, mp, pn, rank)
            if result is None:
                break
            U, V, W = result
            
            if it % 3 == 2:
                err = _compute_error(U, V, W, target)
                if err < best_err:
                    best_err = err
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    if err < 1e-10:
                        break
                    stuck_count = 0
                else:
                    stuck_count += 1
                    if stuck_count > 8 and err > best_err * 1.5:
                        break
        
        # Use best solution found
        U, V, W = best_U, best_V, best_W
        
        # Gradient polish
        if best_err < 0.5:  # Only polish if ALS made progress
            U, V, W = _gradient_polish(U, V, W, target, nm, mp, pn, rank, lr=0.05, max_iter=100, tol=1e-12)
            final_err = _compute_error(U, V, W, target)
            if final_err < best_err:
                best_err = final_err
        
        # Final check
        if best_err < 1e-8:
            decomposed = np.einsum('ar,br,cr->abc', U, V, W)
            if np.allclose(decomposed, target, atol=1e-6, rtol=1e-6):
                # Try to find clean values
                # Round to nearest 0.5
                U_half = np.round(U * 2) / 2
                V_half = np.round(V * 2) / 2
                W_half = np.round(W * 2) / 2
                decomp_half = np.einsum('ar,br,cr->abc', U_half, V_half, W_half)
                if np.allclose(decomp_half, target, atol=1e-4, rtol=1e-4):
                    return {"U": U_half.tolist(), "V": V_half.tolist(), "W": W_half.tolist()}
                
                # Round to nearest integer
                U_int = np.round(U)
                V_int = np.round(V)
                W_int = np.round(W)
                decomp_int = np.einsum('ar,br,cr->abc', U_int, V_int, W_int)
                if np.allclose(decomp_int, target, atol=1e-4, rtol=1e-4):
                    return {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
                
                # Use float values
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None

def construct():
    start_time = time.time()
    deadline = 145.0  # safety margin before 150s
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    schoolbook_rank = n * m * p  # 48
    
    # Generate schoolbook solution as fallback
    U_sb = np.zeros((nm, schoolbook_rank))
    V_sb = np.zeros((mp, schoolbook_rank))
    W_sb = np.zeros((pn, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1.0
                V_sb[j*p+k, col] = 1.0
                W_sb[k*n+i, col] = 1.0
                col += 1
    best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    best_rank = schoolbook_rank
    
    # Build target tensor
    target = _build_target(n, m, p)
    
    rng = np.random.RandomState(42)
    
    # Try ranks from lower bound up to schoolbook
    # Lower bound: max(n*m, m*p, p*n) = 24
    lower_bound = max(nm, mp, pn)
    
    for rank in range(lower_bound, schoolbook_rank + 1):
        if time.time() > deadline:
            break
        
        # More attempts for lower ranks (harder to find)
        if rank < 30:
            max_attempts = 200
        elif rank < 40:
            max_attempts = 300
        else:
            max_attempts = 400
        
        result = _try_rank(rank, target, nm, mp, pn, n, m, p, rng, max_attempts, deadline)
        
        if result is not None:
            best_rank = rank
            best_result = result
            break
    
    return best_result
