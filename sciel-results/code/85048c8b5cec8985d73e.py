import time
import numpy as np
from fractions import Fraction
from scipy.optimize import minimize

DIMENSIONS = [3, 4, 4]

def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    """Fast vectorized verification using einsum."""
    try:
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        if U_arr.shape[1] != V_arr.shape[1] or U_arr.shape[1] != W_arr.shape[1]:
            return False, float('inf')
            
        rank = U_arr.shape[1]
        
        U_3d = U_arr.reshape(n, m, rank)
        V_3d = V_arr.reshape(m, p, rank)
        W_3d = W_arr.reshape(p, n, rank)
        
        T = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
        
        max_err = np.max(np.abs(T - 1.0))
        return max_err < tol, max_err
    except Exception:
        return False, float('inf')

def _verify_exact(U, V, W, n, m, p):
    """Verify using exact Fraction arithmetic."""
    try:
        U_frac = [[Fraction(x) for x in row] for row in U]
        V_frac = [[Fraction(x) for x in row] for row in V]
        W_frac = [[Fraction(x) for x in row] for row in W]
        
        rank = len(U[0])
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = Fraction(0)
                    for c in range(rank):
                        val += U_frac[i*m+j][c] * V_frac[j*p+k][c] * W_frac[k*n+i][c]
                    if val != 1:
                        return False
        return True
    except Exception:
        return False

def _try_polish(U, V, W, n, m, p, max_denom=20):
    """Try to snap to small rationals and verify."""
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    for denom in [1, 2, 3, 4, 5, 10, 20]:
        U_frac = [[Fraction(x).limit_denominator(denom) for x in row] for row in U_list]
        V_frac = [[Fraction(x).limit_denominator(denom) for x in row] for row in V_list]
        W_frac = [[Fraction(x).limit_denominator(denom) for x in row] for row in W_list]
        
        if _verify_exact(U_frac, V_frac, W_frac, n, m, p):
            return U_frac, V_frac, W_frac
    return None

def _objective_and_grad(x, n, m, p, r):
    """
    Objective function and gradient for L-BFGS-B.
    x is a flattened array of [U, V, W]
    """
    # Unpack
    size_U = n * m * r
    size_V = m * p * r
    size_W = p * n * r
    
    U_flat = x[:size_U]
    V_flat = x[size_U:size_U + size_V]
    W_flat = x[size_U + size_V:]
    
    U = U_flat.reshape(n, m, r)
    V = V_flat.reshape(m, p, r)
    W = W_flat.reshape(p, n, r)
    
    # Compute T
    T = np.einsum('ijc,jkc,kic->ijk', U, V, W)
    diff = T - 1.0
    loss = np.sum(diff ** 2)
    
    # Compute Gradients
    # dL/dU = 2 * einsum('ijk,jkc,kic->ijc', diff, V, W)
    gU = 2.0 * np.einsum('ijk,jkc,kic->ijc', diff, V, W)
    # dL/dV = 2 * einsum('ijk,ijc,kic->jkc', diff, U, W)
    gV = 2.0 * np.einsum('ijk,ijc,kic->jkc', diff, U, W)
    # dL/dW = 2 * einsum('ijk,ijc,jkc->kic', diff, U, V)
    gW = 2.0 * np.einsum('ijk,ijc,jkc->kic', diff, U, V)
    
    grad = np.concatenate([gU.flatten(), gV.flatten(), gW.flatten()])
    
    return loss, grad

def _optimize_rank(n, m, p, r, max_time=10, seed=42):
    """
    Use L-BFGS-B to find a rank-r decomposition.
    """
    start_time = time.time()
    rng = np.random.RandomState(seed)
    
    # Initialization: Random
    U = rng.randn(n * m, r)
    V = rng.randn(m * p, r)
    W = rng.randn(p * n, r)
    
    # Normalize to start in a reasonable region
    U = U / np.linalg.norm(U) * np.sqrt(n*m)
    V = V / np.linalg.norm(V) * np.sqrt(m*p)
    W = W / np.linalg.norm(W) * np.sqrt(p*n)
    
    x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    size_U = n * m * r
    size_V = m * p * r
    size_W = p * n * r
    
    # We will run L-BFGS-B
    # maxiter is limited by time, but L-BFGS-B doesn't have a direct time limit.
    # We can use a callback to stop if time is up.
    
    state = {'stop': False}
    
    def callback(xk):
        if time.time() - start_time > max_time:
            state['stop'] = True
            return True
        return False
    
    # Run optimization
    # Tolerance for L-BFGS-B
    res = minimize(
        _objective_and_grad, 
        x0, 
        args=(n, m, p, r),
        jac=True, 
        method='L-BFGS-B', 
        options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 10000, 'disp': False},
        callback=callback
    )
    
    if state['stop']:
        # Use the last x if stopped? Minimize returns the last one if it stopped early?
        # Actually, if callback returns True, it stops. res.x is the last point.
        pass
        
    if res.success or (time.time() - start_time > max_time):
        x_opt = res.x
        U_opt = x_opt[:size_U].reshape(n, m, r)
        V_opt = x_opt[size_U:size_U + size_V].reshape(m, p, r)
        W_opt = x_opt[size_U + size_V:].reshape(p, n, r)
        
        # Check error
        T = np.einsum('ijc,jkc,kic->ijk', U_opt, V_opt, W_opt)
        err = np.max(np.abs(T - 1.0))
        
        if err < 1e-6:
            # Try to polish
            U_list = U_opt.tolist()
            V_list = V_opt.tolist()
            W_list = W_opt.tolist()
            
            polished = _try_polish(U_list, V_list, W_list, n, m, p)
            if polished is not None:
                U_f, V_f, W_f = polished
                return {"U": U_f, "V": V_f, "W": W_f}
            
            # If polish fails, check if it's close enough to integers/half-integers manually?
            # _try_polish already does this.
            # If we can't polish, we can't return it as "exact" unless we force it.
            # But the task requires exact decomposition.
            # So we discard if not polished.
            
    return None

def _schoolbook():
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

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Start with the known valid schoolbook solution
    result = _schoolbook()
    best_rank = n * m * p
    
    # Try to find lower rank solutions
    # Theoretical min is 16.
    # We try ranks from 16 upwards.
    # We allocate time based on remaining budget.
    
    # Strategy:
    # Try ranks 16 to 47.
    # For each rank, try a few seeds.
    # If we find a valid one, update result and best_rank.
    # Continue to lower ranks? No, we go UP from 16.
    # If we find rank 20, we continue to 21? No, we want MINIMAL.
    # So we should iterate r from 16 to 47.
    # If we find a solution for r=20, we record it.
    # Then we continue to r=21? No, we want the minimal.
    # So if we find a solution for r, we can stop?
    # No, it's possible r=19 works but r=20 was found first?
    # We iterate r from 16 to 47 in increasing order.
    # The first one we find is the minimal (among those we search).
    # However, search is stochastic. We might fail r=16 and find r=17.
    # We should try each rank with multiple seeds to increase confidence.
    
    # Time budget: ~140s.
    # Let's try ranks 16 to 30 first, as lower ranks are more likely to be the answer if they exist.
    # If we find nothing by 30, we try up to 47.
    
    ranks_to_try = list(range(16, 48))
    
    for r in ranks_to_try:
        if time.time() > deadline - 5:
            break
            
        # Determine how many seeds to try for this rank
        # We want to spend roughly equal time per rank? Or more on lower ranks?
        # Let's say 5 seconds per rank on average.
        # 140s / 32 ranks ~ 4s per rank.
        # We can try 2-3 seeds of 1-2s each? Or 1 seed of 4s?
        # L-BFGS-B is fast. Let's try 2 seeds of 2s each.
        
        found = False
        for seed in range(5):
            if time.time() > deadline - 2:
                break
            
            # Short time per attempt to allow more attempts
            max_time_attempt = 2.0 
            res = _optimize_rank(n, m, p, r, max_time=max_time_attempt, seed=seed)
            
            if res is not None:
                # Found a valid decomposition of rank r
                if r < best_rank:
                    best_rank = r
                    result = res
                    found = True
                    # Since we are iterating r in increasing order, 
                    # if we find a solution for r, it is the minimal *found so far*.
                    # But we should check if we can find r-1?
                    # We already tried r-1. If we failed, we might have missed it.
                    # Should we restart lower ranks?
                    # Given time constraints, let's just accept the first found minimal rank.
                    # Or, we can break and return, as we are searching from low to high.
                    # If we find rank 20, and we already tried 16-19 without success, 
                    # it's likely 20 is the answer (or we missed lower ones).
                    # To be safe, we can continue searching lower ranks with more seeds?
                    # No, we already did.
                    # Let's break and return the best found.
                    # Wait, if we find rank 20, we should stop?
                    # Yes, because we are iterating 16, 17, 18, 19, 20.
                    # If we found 20, 16-19 failed.
                    # It's possible 19 works with a different seed.
                    # But we used 5 seeds for 19.
                    # Let's assume if we fail 5 seeds, it's unlikely.
                    # So we can break.
                    break
        
        if found:
            # We found a solution for this rank r.
            # Since we iterate r increasing, this is the minimal rank found.
            # We can stop searching higher ranks.
            # Should we try to improve this rank? No, rank is fixed.
            # We can try to find a *different* solution of the same rank? No need.
            # We break the outer loop.
            break

    return result
