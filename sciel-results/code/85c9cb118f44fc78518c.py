"""Root-finder using Newton's method with adaptive step size and global search.
Uses a call counter to stay within budget and a time deadline for safety."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def find_root_system(F, x0):
    """
    Find x in R^2 such that ||F(x)|| <= 1e-8.
    
    Strategy:
    1. Newton's method with adaptive step size (uses Jacobian estimation).
    2. Global search: if local Newton fails, try multiple starting points.
    
    Call budget: stay well under 500 calls.
    Time budget: ~140 seconds.
    """
    start_time = time.time()
    deadline = start_time + 140.0
    
    x0_arr = np.array(x0, dtype=float).copy()
    best_x = x0_arr.copy()
    best_norm = float('inf')
    
    calls = 0
    max_calls = 490
    
    def safe_F(x):
        nonlocal calls
        if time.time() > deadline:
            raise TimeoutError
        if calls >= max_calls:
            raise RuntimeError("Call limit exceeded")
        try:
            Fx = F(x)
            Fx = np.asarray(Fx, dtype=float)
            if Fx.shape != (2,):
                Fx = Fx.reshape(2)
            if not np.all(np.isfinite(Fx)):
                return None
            calls += 1
            return Fx
        except Exception:
            return None

    def is_valid(x):
        return np.all(np.isfinite(x)) and x.shape == (2,)

    def compute_jacobian(x, h=1e-6):
        """Estimate Jacobian using finite differences."""
        Fx0 = safe_F(x)
        if Fx0 is None:
            return None, None
        J = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            Fx_j = safe_F(x + e)
            if Fx_j is None:
                return None, Fx0
            J[:, j] = (Fx_j - Fx0) / h
        return J, Fx0

    def try_newton(x_start, max_iter=100, tol=1e-10):
        """
        Solves F(x) = 0 using Newton's method.
        Returns (x, norm, success).
        """
        nonlocal best_x, best_norm
        
        x = x_start.copy()
        
        # Initial evaluation
        J, Fx = compute_jacobian(x)
        if Fx is None:
            return x, float('inf'), False
        norm = np.linalg.norm(Fx)
        
        if norm < best_norm:
            best_norm = norm
            best_x = x.copy()
            
        if norm <= tol:
            return x, norm, True
        
        for _ in range(max_iter):
            if time.time() > deadline or calls >= max_calls - 5:
                break
            
            # Solve J * dx = -Fx
            try:
                dx = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                try:
                    dx = np.linalg.lstsq(J, -Fx, rcond=None)[0]
                except Exception:
                    break
            
            if not np.all(np.isfinite(dx)):
                break
            
            # Adaptive step size with backtracking
            alpha = 1.0
            step_accepted = False
            
            for _ in range(10):
                x_try = x + alpha * dx
                if not is_valid(x_try):
                    alpha *= 0.5
                    continue
                
                J_try, Fx_try = compute_jacobian(x_try)
                if Fx_try is None:
                    alpha *= 0.5
                    continue
                
                norm_try = np.linalg.norm(Fx_try)
                
                if norm_try < best_norm:
                    best_norm = norm_try
                    best_x = x_try.copy()
                
                if norm_try < norm * 0.9:
                    x = x_try
                    J = J_try
                    Fx = Fx_try
                    norm = norm_try
                    step_accepted = True
                    break
                else:
                    alpha *= 0.5
            
            if not step_accepted:
                break
            
            if norm <= tol:
                return x, norm, True
        
        return x, norm, norm <= tol

    # --- Phase 1: Local Newton from x0 ---
    x_result, norm_result, success = try_newton(x0_arr, max_iter=100, tol=1e-10)
    if success:
        return x_result
    
    # --- Phase 2: Global Search ---
    # Generate candidate starting points
    candidates = [x0_arr.copy()]
    
    # Deterministic offsets
    for scale in [0.5, 1.0, 2.0, 5.0, 10.0]:
        for sign in [-1, 1]:
            for axis in range(2):
                x_cand = x0_arr.copy()
                x_cand[axis] += sign * scale * (abs(x0_arr[axis]) + 1.0)
                candidates.append(x_cand)
    
    # Gradient-descent-like direction
    Fx0 = safe_F(x0_arr)
    if Fx0 is not None:
        norm0 = np.linalg.norm(Fx0)
        if norm0 > 0:
            grad_dir = -Fx0 / norm0
            for scale in [1.0, 2.0, 5.0]:
                x_cand = x0_arr + scale * grad_dir
                candidates.append(x_cand)
    
    # Random candidates
    rng = np.random.RandomState(42)
    for _ in range(20):
        x_cand = x0_arr + rng.randn(2) * (np.linalg.norm(x0_arr) + 1.0)
        candidates.append(x_cand)
    
    # Try each candidate
    for x_cand in candidates:
        if time.time() > deadline - 10 or calls >= max_calls - 10:
            break
        if not is_valid(x_cand):
            continue
        
        x_result, norm_result, success = try_newton(x_cand, max_iter=50, tol=1e-10)
        if success:
            return x_result
    
    return best_x
# EVOLVE-BLOCK-END
