import numpy as np
import time
from scipy.optimize import minimize

def arithmetic_kakeya_distribution() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 600.0
    safety_margin = 30.0
    
    n = 10
    # Create coordinate arrays
    x = np.arange(n)  # x values 0..9
    y = np.arange(n)  # y values 0..9
    X, Y = np.meshgrid(x, y, indexing='xy')  # X[i,j] = j, Y[i,j] = i
    # P[y, x] so P[i, j] corresponds to Y=i, X=j
    # X[i,j] = j, Y[i,j] = i
    
    # Compute projection values
    proj_m1 = X - Y      # x - y, range -9..9
    proj_0 = X           # x, range 0..9
    proj_1 = X + Y       # x + y, range 0..18
    proj_2 = X + 2 * Y   # x + 2y, range 0..27
    
    def compute_entropies(P):
        """Compute entropies for all projections given probability array P."""
        # P has shape (10, 10), P[y, x]
        # Flatten and compute projections
        p_flat = P.ravel()
        
        # For each projection, pool probabilities by value
        def entropy_from_projection(proj_vals, probs):
            # proj_vals: array of projection values (same shape as probs)
            # probs: probability array
            unique_vals, inverse = np.unique(proj_vals, return_inverse=True)
            # Sum probabilities for each unique value
            pooled = np.zeros(len(unique_vals))
            np.add.at(pooled, inverse, probs)
            # Shannon entropy in bits, ignoring zero probabilities
            mask = pooled > 1e-15
            if not np.any(mask):
                return 0.0
            p = pooled[mask]
            return -np.sum(p * np.log2(p))
        
        H_m1 = entropy_from_projection(proj_m1.ravel(), p_flat)
        H_0 = entropy_from_projection(proj_0.ravel(), p_flat)
        H_1 = entropy_from_projection(proj_1.ravel(), p_flat)
        H_2 = entropy_from_projection(proj_2.ravel(), p_flat)
        
        return H_m1, H_0, H_1, H_2
    
    def soft_max(values, temperature=0.01):
        """Soft max approximation for smooth optimization."""
        if temperature <= 0:
            return np.max(values)
        shifted = values / temperature
        max_val = np.max(shifted)
        exp_shifted = np.exp(shifted - max_val)
        log_sum_exp = max_val + np.log(np.sum(exp_shifted))
        return log_sum_exp * temperature
    
    def objective_and_gradient(params):
        """
        params: flattened array of size 100, representing log-weights
        We use exp(params) / sum(exp(params)) to get probabilities (softmax parameterization)
        """
        elapsed = time.time() - start_time
        if elapsed > deadline - safety_margin:
            return None, None
        
        # Convert params to probabilities using softmax
        params_clipped = np.clip(params, -50, 50)
        exp_params = np.exp(params_clipped)
        P = exp_params / np.sum(exp_params)
        
        # Compute entropies
        H_m1, H_0, H_1, H_2 = compute_entropies(P)
        
        # M = max(H_0, H_1, H_2), use soft max for smoothness
        M_soft = soft_max([H_0, H_1, H_2], temperature=0.01)
        
        # Objective: maximize D/M, equivalent to minimize -D/M
        # For numerical stability, use log: minimize -(log(D) - log(M))
        if H_m1 <= 1e-15 or M_soft <= 1e-15:
            return 1e10, None
        
        obj = -(np.log(H_m1) - np.log(M_soft))
        
        # Compute gradient using finite differences for simplicity
        # Actually, let's use analytical gradient
        # d/dP of H_r where H_r = -sum p_i log2(p_i)
        # dH_r/dp_j = -log2(p_j) - 1/ln(2)
        
        # For the ratio D/M, gradient is complex. Use finite differences.
        eps = 1e-7
        grad = np.zeros_like(params)
        for i in range(len(params)):
            params_plus = params.copy()
            params_plus[i] += eps
            exp_plus = np.exp(np.clip(params_plus, -50, 50))
            P_plus = exp_plus / np.sum(exp_plus)
            H_m1_plus, H_0_plus, H_1_plus, H_2_plus = compute_entropies(P_plus)
            M_soft_plus = soft_max([H_0_plus, H_1_plus, H_2_plus], temperature=0.01)
            if H_m1_plus <= 1e-15 or M_soft_plus <= 1e-15:
                obj_plus = 1e10
            else:
                obj_plus = -(np.log(H_m1_plus) - np.log(M_soft_plus))
            
            grad[i] = (obj_plus - obj) / eps
        
        return obj, grad
    
    # Initial guess: uniform distribution
    # In log-space, uniform means all params equal
    params0 = np.zeros(n * n)
    
    # Try multiple initializations
    best_score = -np.inf
    best_P = None
    
    # First, try the uniform distribution
    P_uniform = np.ones((n, n)) / (n * n)
    H_m1, H_0, H_1, H_2 = compute_entropies(P_uniform)
    M = max(H_0, H_1, H_2)
    if H_m1 > 1e-15 and M > 1e-15:
        score_uniform = H_m1 / M
        best_score = score_uniform
        best_P = P_uniform.copy()
    
    # Try optimization from uniform
    elapsed = time.time() - start_time
    if elapsed < deadline - safety_margin - 10:
        try:
            result = minimize(
                objective_and_gradient,
                params0,
                method='SLSQP',
                jac=True,
                options={'maxiter': 200, 'ftol': 1e-10, 'disp': False}
            )
            if result.success and result.fun < 1e9:
                params_opt = result.x
                exp_params = np.exp(np.clip(params_opt, -50, 50))
                P_opt = exp_params / np.sum(exp_params)
                P_opt = P_opt.reshape(n, n)
                P_opt = np.clip(P_opt, 0, None)
                P_opt = P_opt / np.sum(P_opt)
                
                H_m1_opt, H_0_opt, H_1_opt, H_2_opt = compute_entropies(P_opt)
                M_opt = max(H_0_opt, H_1_opt, H_2_opt)
                if H_m1_opt > 1e-15 and M_opt > 1e-15:
                    score_opt = H_m1_opt / M_opt
                    if score_opt > best_score:
                        best_score = score_opt
                        best_P = P_opt.copy()
        except Exception:
            pass
    
    # Try a structured initialization: put mass on diagonal x=y+k for various k
    elapsed = time.time() - start_time
    if elapsed < deadline - safety_margin - 10:
        # Try putting mass on cells where x - y is spread out
        # For example, put mass on cells (y, x) where x = y + c for c in some set
        # This makes x-y take few values, which is bad for D
        # Instead, we want x-y to take many values
        
        # Try: put mass on corners and edges
        P_struct = np.zeros((n, n))
        # Put mass on cells that maximize x-y diversity
        # x-y ranges from -9 to 9
        # To get high H(x-y), spread mass over many x-y values
        # But to get low M, concentrate x, x+y, or x+2y
        
        # Strategy: put mass on cells where x is concentrated but x-y is spread
        # E.g., fix x=0, vary y: x-y = -y, so y=0..9 gives x-y=0,-1,...,-9
        # Then H(x-y) = log2(10) = 3.32, H(x) = 0, so M >= 0 but H(x+y)=H(y)=3.32, H(x+2y)=H(2y)=3.32
        # So M = 3.32, D = 3.32, ratio = 1
        
        # Better: put mass on a line x = a*y + b
        # Try different lines
        for a in [0, 1, 2, -1, 0.5, -0.5]:
            for b in range(-5, 15):
                P_line = np.zeros((n, n))
                count = 0
                for yy in range(n):
                    xx = int(round(a * yy + b))
                    if 0 <= xx < n:
                        P_line[yy, xx] = 1.0
                        count += 1
                if count > 5:
                    P_line = P_line / count
                    H_m1_l, H_0_l, H_1_l, H_2_l = compute_entropies(P_line)
                    M_l = max(H_0_l, H_1_l, H_2_l)
                    if H_m1_l > 1e-15 and M_l > 1e-15:
                        score_l = H_m1_l / M_l
                        if score_l > best_score:
                            best_score = score_l
                            best_P = P_line.copy()
    
    # Try 2D structured patterns
    elapsed = time.time() - start_time
    if elapsed < deadline - safety_margin - 10:
        # Try putting mass on cells where (x mod k, y mod k) are specific
        # Or use a 2D pattern
        for pattern in ['checkerboard', 'diagonals', 'cross']:
            P_pat = np.zeros((n, n))
            if pattern == 'checkerboard':
                for i in range(n):
                    for j in range(n):
                        if (i + j) % 2 == 0:
                            P_pat[i, j] = 1.0
            elif pattern == 'diagonals':
                for i in range(n):
                    for j in range(n):
                        if (i - j) % 3 == 0:
                            P_pat[i, j] = 1.0
            elif pattern == 'cross':
                for i in range(n):
                    for j in range(n):
                        if i == n // 2 or j == n // 2:
                            P_pat[i, j] = 1.0
            total = P_pat.sum()
            if total > 0:
                P_pat = P_pat / total
                H_m1_p, H_0_p, H_1_p, H_2_p = compute_entropies(P_pat)
                M_p = max(H_0_p, H_1_p, H_2_p)
                if H_m1_p > 1e-15 and M_p > 1e-15:
                    score_p = H_m1_p / M_p
                    if score_p > best_score:
                        best_score = score_p
                        best_P = P_pat.copy()
    
    # Final optimization from best found
    elapsed = time.time() - start_time
    if elapsed < deadline - safety_margin - 5 and best_P is not None:
        try:
            # Convert best_P to log-space params
            P_safe = np.clip(best_P, 1e-15, None)
            P_norm = P_safe / np.sum(P_safe)
            params_init = np.log(P_norm.ravel())
            
            result2 = minimize(
                objective_and_gradient,
                params_init,
                method='SLSQP',
                jac=True,
                options={'maxiter': 100, 'ftol': 1e-10, 'disp': False}
            )
            if result2.success and result2.fun < 1e9:
                params_opt2 = result2.x
                exp_params2 = np.exp(np.clip(params_opt2, -50, 50))
                P_opt2 = exp_params2 / np.sum(exp_params2)
                P_opt2 = P_opt2.reshape(n, n)
                P_opt2 = np.clip(P_opt2, 0, None)
                P_opt2 = P_opt2 / np.sum(P_opt2)
                
                H_m1_2, H_0_2, H_1_2, H_2_2 = compute_entropies(P_opt2)
                M_2 = max(H_0_2, H_1_2, H_2_2)
                if H_m1_2 > 1e-15 and M_2 > 1e-15:
                    score_2 = H_m1_2 / M_2
                    if score_2 > best_score:
                        best_score = score_2
                        best_P = P_opt2.copy()
        except Exception:
            pass
    
    # Ensure validity
    if best_P is None:
        best_P = np.ones((n, n)) / (n * n)
    
    best_P = np.clip(best_P, 0, None)
    total = best_P.sum()
    if total <= 1e-9:
        best_P = np.ones((n, n)) / (n * n)
    else:
        best_P = best_P / total
    
    return best_P
