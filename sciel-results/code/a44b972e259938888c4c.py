# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize as opt

def calc_prob(support, weights):
    """
    Calculate P[X1 + X2 + X3 < 2*X4]
    support: 1D array of shape (n,)
    weights: 1D array of shape (n,) (normalized)
    """
    n = len(support)
    # Weights should be normalized
    w_sum = np.sum(weights)
    if w_sum <= 1e-9:
        return 0.0
    w = weights / w_sum
    
    # Precompute all pairwise sums s_i + s_j
    # And the threshold 2*s_l - s_k
    # P = sum_k sum_l w_k w_l ( sum_{i,j: s_i+s_j < 2*s_l - s_k} w_i w_j )
    
    # To make it efficient, we can iterate over k, l
    # n is small (3 or 4), so O(n^4) is fine.
    p_val = 0.0
    for k in range(n):
        for l in range(n):
            threshold = 2.0 * support[l] - support[k]
            # Sum of w_i w_j for s_i + s_j < threshold
            inner_sum = 0.0
            for i in range(n):
                for j in range(n):
                    if support[i] + support[j] < threshold:
                        inner_sum += w[i] * w[j]
            p_val += w[k] * w[l] * inner_sum
    return p_val

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    start_time = time.time()
    deadline = start_time + 550  # Leave 50s margin
    
    best_score = -1.0
    best_support = None
    best_weights = None
    
    # Try different support sizes
    for n in [3, 4, 5]:
        if time.time() > deadline:
            break
            
        # We fix s_0 = 0 and s_{n-1} = 1 to handle scale invariance.
        # We optimize intermediate support points and weights.
        # Variables:
        # x_1, ..., x_{n-2} (intermediate support points)
        # w_1, ..., w_{n-2} (intermediate weights)
        # w_0 is determined by 1 - sum(w_1...w_{n-1})? 
        # Actually, we have n weights. Let's optimize w_1..w_{n-1} and set w_0 = 1 - sum.
        # But w_0 must be > 0.
        
        # Let's use a transformation to ensure positivity and order.
        # Support: 0 < x_1 < ... < x_{n-2} < 1
        # We can parameterize x_i via cumulative sums of positive variables.
        # Let z_1, ..., z_{n-2} > 0.
        # x_1 = z_1 / (1 + sum(z)) ? No, simpler to just use bounds and check order in objective.
        
        # Simpler: Optimize raw parameters and penalize violations.
        # Or use 'L-BFGS-B' with bounds.
        
        # Let's define the parameter vector:
        # [x_1, ..., x_{n-2}, w_1, ..., w_{n-1}]
        # Constraints:
        # 0 < x_1 < ... < x_{n-2} < 1
        # w_i > 0, sum(w) = 1 => w_0 = 1 - sum(w_1..w_{n-1}) > 0 => sum(w_1..w_{n-1}) < 1.
        
        # To handle the sum constraint and positivity easily, 
        # let's optimize log-weights for w_1..w_{n-1} and use a softmax-like mapping?
        # No, just optimize w_1..w_{n-1} with bounds [eps, 1-eps] and ensure sum < 1-eps.
        # This is hard with simple bounds.
        
        # Alternative: Optimize u_1..u_{n-1} for weights using softmax.
        # w_i = exp(u_i) / sum(exp(u)).
        # This ensures w_i > 0 and sum=1.
        # So we optimize u_1..u_{n-1} (unbounded) and x_1..x_{n-2} (bounded 0,1).
        # But we also need to enforce order x_1 < x_2 ...
        # We can sort the support after optimization? 
        # If we sort, the mapping from params to support is non-smooth.
        # Better to enforce order via parameters:
        # Let y_1, ..., y_{n-2} be the gaps.
        # x_1 = y_1
        # x_2 = y_1 + y_2
        # ...
        # x_{n-2} = sum(y_1..y_{n-2})
        # We need x_{n-2} < 1. So sum(y) < 1.
        # Let y_i = exp(v_i). Then sum(exp(v_i)) < 1.
        # This is also a constraint.
        
        # Let's just use a generic multivariate optimizer with a penalty function for constraints.
        # Objective: -P + Penalty
        # Penalty: 
        #   - If any x_i < 0 or x_i > 1
        #   - If order is violated
        #   - If any w_i <= 0
        
        # Actually, for n=3, it's simple enough to solve analytically or with a 2D grid search + local opt.
        # Let's do a grid search for initial guesses, then local optimization.
        
        pass # Placeholder for logic below

    # Implement the logic for n=3 specifically first as it's most likely to yield a quick win
    # And n=4.
    
    def solve_for_n(n):
        # Parameters:
        # x_1 ... x_{n-2}
        # u_1 ... u_{n-1} (log weights for indices 1..n-1)
        # w_0 is derived.
        
        # For n=3:
        # x_1 (1 param)
        # u_1, u_2 (2 params)
        # w_1 = exp(u_1) / (1 + exp(u_1) + exp(u_2)) ? 
        # No, if we use softmax on u_1..u_{n-1}, we get weights for indices 1..n-1.
        # But we also have index 0.
        # Let's use softmax on u_0..u_{n-1}.
        # Then w_i = exp(u_i) / sum(exp(u)).
        # This is clean.
        # So we optimize u_0..u_{n-1} and x_1..x_{n-2}.
        # Total params: n (for u) + n-2 (for x) = 2n - 2.
        
        def objective(params):
            # Unpack
            u = params[:n]
            x_raw = params[n:]
            
            # Weights
            # Shift for numerical stability
            u_max = np.max(u)
            exp_u = np.exp(u - u_max)
            w = exp_u / np.sum(exp_u)
            
            # Support
            # x_raw are the intermediate points.
            # We need to ensure 0 < x_1 < ... < x_{n-2} < 1.
            # Let's use a softplus or sigmoid mapping to ensure they are in (0,1)?
            # And to ensure order, we can sort them? 
            # If we sort, the gradient might be 0 at ties.
            # Let's just penalize if order is violated or out of bounds.
            
            x = np.zeros(n)
            x[0] = 0.0
            x[-1] = 1.0
            
            # Map x_raw to (0,1) using sigmoid
            x_int = 1.0 / (1.0 + np.exp(-x_raw))
            
            # Check order and bounds
            penalty = 0.0
            # Check if x_int are sorted
            for i in range(len(x_int)-1):
                if x_int[i] >= x_int[i+1]:
                    penalty += 100.0 * (x_int[i] - x_int[i+1])
            
            # Assign
            x[1:-1] = x_int
            
            # Check bounds (sigmoid ensures (0,1), so bounds are ok)
            
            if penalty > 0:
                return penalty - 1000.0 # Return large positive value (minimizing -P)
            
            p_val = calc_prob(x, w)
            return -p_val + penalty
        
        # Initial guess
        # x_raw: 0 (maps to 0.5)
        # u: 0 (uniform weights)
        x0 = np.zeros(2*n - 2)
        
        # Run optimization
        res = opt.minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6})
        
        if res.success or res.fun < -0.5: # If we found a decent score
            u_opt = res.x[:n]
            x_raw_opt = res.x[n:]
            
            # Reconstruct
            u_max = np.max(u_opt)
            exp_u = np.exp(u_opt - u_max)
            w_opt = exp_u / np.sum(exp_u)
            
            x_opt = np.zeros(n)
            x_opt[0] = 0.0
            x_opt[-1] = 1.0
            x_int = 1.0 / (1.0 + np.exp(-x_raw_opt))
            # Sort x_int to ensure valid support
            x_int = np.sort(x_int)
            x_opt[1:-1] = x_int
            
            # Check validity
            if np.all(x_opt >= 0) and np.all(w_opt >= 0) and np.sum(w_opt) > 1e-9:
                p_val = calc_prob(x_opt, w_opt)
                return x_opt, w_opt, p_val
        return None, None, -1.0

    # Try n=3
    s3, w3, p3 = solve_for_n(3)
    if p3 > best_score:
        best_score = p3
        best_support = s3
        best_weights = w3
        
    # Try n=4
    if time.time() < deadline:
        s4, w4, p4 = solve_for_n(4)
        if p4 > best_score:
            best_score = p4
            best_support = s4
            best_weights = w4
            
    # Try n=5
    if time.time() < deadline:
        s5, w5, p5 = solve_for_n(5)
        if p5 > best_score:
            best_score = p5
            best_support = s5
            best_weights = w5

    # Fallback to baseline if optimization failed
    if best_support is None:
        n = 32
        best_support = np.arange(n, dtype=np.float64)
        best_weights = np.ones(n, dtype=np.float64) / n
    else:
        # Scale back? No, scale invariance means [0..1] is fine.
        # But maybe larger values are better for float precision?
        # The metric uses float64. [0..1] is fine.
        pass

    return best_support, best_weights

# EVOLVE-BLOCK-END
