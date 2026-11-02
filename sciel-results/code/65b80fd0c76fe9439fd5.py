import time
import numpy as np
import scipy.optimize as opt

DIMENSIONS = [3, 5, 6]

def _schoolbook(n, m, p):
    """Return schoolbook decomposition with rank n*m*p."""
    nm, mp, pn = n * m, m * p, p * n
    rank = n * m * p
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _build_target_tensor(n, m, p):
    """Build the dense target tensor for matrix multiplication."""
    nm, mp, pn = n * m, m * p, p * n
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                d = k * n + i
                T[a, b, d] = 1.0
    return T

def _optimize_rank(n, m, p, rank, T_target, deadline):
    """
    Attempt to find a decomposition of the given rank using gradient descent.
    Returns (U, V, W) if successful (low error), else None.
    """
    nm, mp, pn = n * m, m * p, p * n
    n_params = rank * (nm + mp + pn)
    
    def loss(params):
        if time.time() > deadline:
            return float('inf')
        
        # Reshape parameters into U, V, W
        U = params[:rank*nm].reshape(nm, rank)
        V = params[rank*nm:rank*(nm+mp)].reshape(mp, rank)
        W = params[rank*(nm+mp):].reshape(pn, rank)
        
        # Compute tensor contraction
        # T_calc[a,b,d] = sum_r U[a,r] V[b,r] W[d,r]
        T_calc = np.einsum('ar,br,dr->abd', U, V, W)
        
        # Compute squared error
        diff = T_calc - T_target
        return np.sum(diff * diff)
    
    # Try a few random initializations
    for trial in range(3): # 3 restarts per rank
        if time.time() > deadline:
            return None
        
        # Random initialization
        # Scale matters. Schoolbook entries are 0/1.
        # Random normal with small std might work.
        x0 = np.random.randn(n_params) * 0.1
        
        # Optimize
        try:
            result = opt.minimize(loss, x0, method='L-BFGS-B', 
                                  options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-15, 'maxls': 50})
            
            if result.success or result.fun < 1e-10:
                # Extract U, V, W
                U = result.x[:rank*nm].reshape(nm, rank)
                V = result.x[rank*nm:rank*(nm+mp)].reshape(mp, rank)
                W = result.x[rank*(nm+mp):].reshape(pn, rank)
                
                # Final verification
                T_calc = np.einsum('ar,br,dr->abd', U, V, W)
                if np.allclose(T_calc, T_target, atol=1e-9):
                    return U, V, W
        except Exception:
            pass
            
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    schoolbook_rank = n * m * p
    
    # Get schoolbook as fallback
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    
    # Build target tensor
    T_target = _build_target_tensor(n, m, p)
    
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = schoolbook_rank
    
    # Search for lower ranks
    # Start from schoolbook_rank - 1 and go down
    # Limit the search to a reasonable range to fit in time
    # e.g., down to 80 or until time runs out
    min_rank_to_try = max(30, schoolbook_rank - 20) # Try ranks 90 down to 70
    
    for rank in range(schoolbook_rank - 1, min_rank_to_try - 1, -1):
        if time.time() > deadline - 10.0: # Stop if less than 10s left
            break
            
        result = _optimize_rank(n, m, p, rank, T_target, deadline)
        if result is not None:
            U, V, W = result
            best_U, best_V, best_W = U, V, W
            best_rank = rank
            # If we find a significantly better rank, we might stop or continue
            # Let's continue to see if we can go even lower, but be cautious with time
            if best_rank <= 85:
                 break # Found a good improvement, stop to save time for verification? 
                 # Actually, let's just break if we find ANY improvement to ensure we return it quickly
                 # Or continue? The score is seed/value. Lower value is better.
                 # So we want the lowest rank.
                 # But finding lower rank is harder.
                 # Let's break if we find one to ensure we return a valid result.
                 # If we continue, we might run out of time and return the schoolbook (if we don't save best).
                 # We are saving best. So we can continue.
                 # But if we run out of time, we return best.
                 # So we should continue searching if time allows.
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
