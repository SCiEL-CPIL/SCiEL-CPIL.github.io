"""Matrix multiplication 2x3x5 tensor decomposition - rank minimization."""
import numpy as np
import time
from scipy.optimize import least_squares, minimize

DIMENSIONS = [2, 3, 5]

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i,j,k] as a 3D array."""
    T = np.zeros((n*m, m*p, n*p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    return T

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that sum_c U[:,c] ⊗ V[:,c] ⊗ W[:,c] == T."""
    rank = U.shape[1]
    T_recon = np.zeros((n*m, m*p, n*p))
    for c in range(rank):
        T_recon += np.einsum('a,b,c->abc', U[:,c], V[:,c], W[:,c])
    T_true = _build_tensor(n, m, p)
    return np.max(np.abs(T_recon - T_true)) < tol

def _get_active_indices(n, m, p):
    """Get active (a, b, c) indices for the tensor."""
    active = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                active.append((a, b, c))
    return active

def _get_inactive_constraints(n, m, p):
    """Get inactive (a, b, c) indices - those that should be zero."""
    nm, mp, pn = n*m, m*p, p*n
    active_set = set(_get_active_indices(n, m, p))
    inactive = []
    for a in range(nm):
        for b in range(mp):
            for c in range(pn):
                if (a, b, c) not in active_set:
                    inactive.append((a, b, c))
    return inactive

def _make_residuals(n, m, p, active, inactive):
    """Create a residual function for least_squares."""
    nm, mp, pn = n*m, m*p, p*n
    
    def residuals(params, R):
        U = params[:R*nm].reshape(nm, R)
        V = params[R*nm:R*nm+R*mp].reshape(mp, R)
        W = params[R*nm+R*mp:].reshape(pn, R)
        
        res = np.empty(len(active) + len(inactive))
        
        # Active constraints: sum should equal 1
        for idx, (a, b, c) in enumerate(active):
            res[idx] = np.sum(U[a] * V[b] * W[c]) - 1.0
        
        # Inactive constraints: sum should equal 0
        offset = len(active)
        for idx, (a, b, c) in enumerate(inactive):
            res[offset + idx] = np.sum(U[a] * V[b] * W[c])
        
        return res
    
    return residuals

def _polish(U, V, W, n, m, p, active, inactive, max_iters=50, tol=1e-12):
    """High-precision polish pass using Newton-like steps."""
    nm, mp, pn = n*m, m*p, p*n
    R = U.shape[1]
    
    for it in range(max_iters):
        # Compute current residual
        recon = np.zeros((nm, mp, pn))
        for c in range(R):
            recon += np.einsum('a,b,c->abc', U[:,c], V[:,c], W[:,c])
        
        # Build residual vector
        res = np.zeros(len(active) + len(inactive))
        for idx, (a, b, c) in enumerate(active):
            res[idx] = recon[a, b, c] - 1.0
        offset = len(active)
        for idx, (a, b, c) in enumerate(inactive):
            res[offset + idx] = recon[a, b, c]
        
        if np.max(np.abs(res)) < tol:
            break
        
        # Compute Jacobian and solve
        # This is a large system, so use least squares for the step
        # For efficiency, use a simplified approach: direct update
        
        # Gradient-based step
        lr = 0.05
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        for c in range(R):
            # dU[:,c] = sum over (b,c) where residual is nonzero
            for a, b, cc in active:
                if cc == c:
                    dU[a, c] += (recon[a, b, c] - 1.0) * V[b, c] * W[cc, c]
            for a, b, cc in inactive:
                if cc == c:
                    dU[a, c] += recon[a, b, c] * V[b, c] * W[cc, c]
            
            for a, b, cc in active:
                if cc == c:
                    dV[b, c] += (recon[a, b, c] - 1.0) * U[a, c] * W[cc, c]
            for a, b, cc in inactive:
                if cc == c:
                    dV[b, c] += recon[a, b, c] * U[a, c] * W[cc, c]
            
            for a, b, cc in active:
                if cc == c:
                    dW[cc, c] += (recon[a, b, c] - 1.0) * U[a, c] * V[b, c]
            for a, b, cc in inactive:
                if cc == c:
                    dW[cc, c] += recon[a, b, c] * U[a, c] * V[b, c]
        
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
        
        # Normalize columns to prevent scale drift
        for c in range(R):
            nu = np.linalg.norm(U[:, c])
            nv = np.linalg.norm(V[:, c])
            nw = np.linalg.norm(W[:, c])
            if nu > 1e-15 and nv > 1e-15 and nw > 1e-15:
                target = (nu * nv * nw) ** (1/3)
                U[:, c] *= target / nu
                V[:, c] *= target / nv
                W[:, c] *= target / nw
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Schoolbook as fallback
    rank_school = n * m * p
    U_s = np.zeros((nm, rank_school))
    V_s = np.zeros((mp, rank_school))
    W_s = np.zeros((pn, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_s[i*m+j, col] = 1
                V_s[j*p+k, col] = 1
                W_s[k*n+i, col] = 1
                col += 1
    best_result = {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
    best_rank = rank_school
    
    active = _get_active_indices(n, m, p)
    inactive = _get_inactive_constraints(n, m, p)
    
    def _try_rank_least_squares(rank, n_restarts=5, max_nfev=500):
        """Try to find a rank-decomposition using least_squares."""
        nonlocal best_result, best_rank
        
        residuals = _make_residuals(n, m, p, active, inactive)
        
        for restart in range(n_restarts):
            if time.time() > deadline:
                return False
            
            np.random.seed(restart * 1000 + rank)
            x0 = np.random.randn(rank * (nm + mp + pn)) * 0.05
            
            try:
                result = least_squares(
                    residuals, x0, args=(rank,),
                    method='lm', max_nfev=max_nfev
                )
                
                if result.cost < 1e-10:
                    # Extract solution
                    params = result.x
                    U = params[:rank*nm].reshape(nm, rank)
                    V = params[rank*nm:rank*nm+rank*mp].reshape(mp, rank)
                    W = params[rank*nm+rank*mp:].reshape(pn, rank)
                    
                    # Polish for higher precision
                    U, V, W = _polish(U, V, W, n, m, p, active, inactive, max_iters=100)
                    
                    # Verify
                    if _verify(U, V, W, n, m, p, tol=1e-9):
                        if rank < best_rank:
                            best_rank = rank
                            best_result = {
                                "U": U.tolist(),
                                "V": V.tolist(),
                                "W": W.tolist()
                            }
                        return True
                        
            except Exception:
                continue
        
        return False
    
    def _try_rank_gradient(rank, max_iters=2000, n_restarts=3):
        """Try to find a rank-decomposition using gradient descent."""
        nonlocal best_result, best_rank
        
        T = _build_tensor(n, m, p)
        
        for restart in range(n_restarts):
            if time.time() > deadline:
                return False
            
            np.random.seed(restart * 1000 + rank + 7)
            U = np.random.randn(nm, rank) * 0.1
            V = np.random.randn(mp, rank) * 0.1
            W = np.random.randn(pn, rank) * 0.1
            
            lr = 0.01
            for it in range(max_iters):
                if time.time() > deadline:
                    return False
                
                # Compute residual
                recon = np.zeros((nm, mp, pn))
                for c in range(rank):
                    recon += np.einsum('a,b,c->abc', U[:,c], V[:,c], W[:,c])
                residual = recon - T
                err = np.sum(residual**2)
                
                if err < 1e-12:
                    if rank < best_rank:
                        best_rank = rank
                        best_result = {
                            "U": U.tolist(),
                            "V": V.tolist(),
                            "W": W.tolist()
                        }
                    return True
                
                # Compute gradients
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                for c in range(rank):
                    dU[:,c] = np.einsum('abc,bc->a', residual, np.einsum('b,c->bc', V[:,c], W[:,c]))
                    dV[:,c] = np.einsum('abc,ac->b', residual, np.einsum('a,c->ac', U[:,c], W[:,c]))
                    dW[:,c] = np.einsum('abc,ab->c', residual, np.einsum('a,b->ab', U[:,c], V[:,c]))
                
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
                
                # Scale invariance fix
                for c in range(rank):
                    norm_u = np.linalg.norm(U[:,c])
                    norm_v = np.linalg.norm(V[:,c])
                    norm_w = np.linalg.norm(W[:,c])
                    if norm_u > 1e-15 and norm_v > 1e-15 and norm_w > 1e-15:
                        target = (norm_u * norm_v * norm_w) ** (1/3)
                        U[:,c] *= target / norm_u
                        V[:,c] *= target / norm_v
                        W[:,c] *= target / norm_w
                
                if it % 50 == 0 and err > 1.0:
                    lr *= 0.9
                elif it % 50 == 0 and err < 0.1:
                    lr = min(lr * 1.1, 0.05)
            
            # Final polish
            U, V, W = _polish(U, V, W, n, m, p, active, inactive, max_iters=50)
            if _verify(U, V, W, n, m, p, tol=1e-9):
                if rank < best_rank:
                    best_rank = rank
                    best_result = {
                        "U": U.tolist(),
                        "V": V.tolist(),
                        "W": W.tolist()
                    }
                return True
        
        return False
    
    # Known lower bound: rank >= n*m + m*p + n*p - n - m - p + 1 = 22
    lower_bound = n*m + m*p + n*p - n - m - p + 1  # = 22
    
    # Try ranks from lower bound upward
    for rank in range(lower_bound, best_rank):
        if time.time() > deadline:
            break
        
        # Try least squares first (more robust)
        success = _try_rank_least_squares(rank, n_restarts=3, max_nfev=300)
        if success:
            break
        
        # If least squares fails, try gradient descent
        success = _try_rank_gradient(rank, max_iters=500, n_restarts=2)
        if success:
            break
    
    return best_result

# EVOLVE-BLOCK-END
