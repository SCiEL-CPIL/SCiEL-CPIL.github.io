"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100
import numpy as np
import time
import math
from scipy.optimize import minimize

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize

def objective(h):
    """
    Compute the evaluation value:
    value = max(np.correlate(h, 1 - h, mode="full")) / len(h) * 2
    """
    corr = np.correlate(h, 1.0 - h, mode="full")
    return np.max(corr) / N * 2.0

def construct():
    """
    Uses Deterministic Gradient-Based Optimization (SLSQP) on a Smoothed Objective.
    
    Strategy:
    1. The raw objective max(C_k) is non-smooth. We replace it with a Log-Sum-Exp (LSE) approximation:
       f_smooth(h) = (1/lambda) * log( sum( exp(lambda * C_k(h)) ) )
       As lambda increases, this approaches max(C_k).
    2. We minimize f_smooth(h) subject to:
       - sum(h) = 50
       - 0 <= h_i <= 1
    3. We use SLSQP because it handles equality constraints efficiently.
    4. We use multiple restarts with different lambda values (annealing lambda from low to high) 
       and different initial points to avoid local minima.
    
    This is materially different from PSO (swarm) and SA (stochastic acceptance) as it is a 
    deterministic, gradient-based local optimizer that exploits the smooth structure of the 
    LSE approximation to navigate the landscape more efficiently than random walks.
    """
    start_time = time.time()
    time_limit = 140.0  # Safety margin

    def compute_corrs(h):
        """Compute all correlation terms C_k(h) = sum h_i (1 - h_{i+k})"""
        # np.correlate(h, 1-h, 'full') gives 2N-1 terms.
        # The evaluator uses max of these.
        # Note: The correlation is defined as sum h[i] * (1-h[i+k]).
        # np.correlate(a, v) computes sum a[i+k] * v[i]. 
        # Here a=h, v=1-h.
        # So it computes sum h[i+k] * (1-h[i])? 
        # Let's check np.correlate definition: c_k = sum_n a_{n+k} * v_n^*
        # So np.correlate(h, 1-h, 'full') at index k (offset) is sum h[i] * (1-h[i + offset])?
        # Actually, np.correlate(h, 1-h, mode='full') returns c_k where c_k = sum_i h_{i+k} (1-h)_i.
        # This is equivalent to sum_i h_{i+k} (1-h_i).
        # By symmetry of correlation (for real signals, corr(x,y) = corr(y,x) with sign flip in lag? No, corr is not symmetric in lag unless x=y).
        # However, the max over all lags is what matters.
        # The evaluator computes: max(np.correlate(h, 1 - h, mode="full"))
        # So we just need to match this computation exactly.
        return np.correlate(h, 1.0 - h, mode="full")

    def lse_objective(h, lam):
        """
        Log-Sum-Exp approximation of max(C_k).
        Includes a large penalty for constraint violations to help SLSQP if it struggles,
        though SLSQP handles constraints natively. We rely on SLSQP constraints.
        """
        corrs = compute_corrs(h)
        # Shift and scale to avoid overflow in exp
        max_c = np.max(corrs)
        # If lam is large, exp(lam * (c - max_c)) is stable
        shifted = corrs - max_c
        # Use log-sum-exp trick: log(sum(exp(x))) = max(x) + log(sum(exp(x - max(x))))
        lse_val = max_c + np.log(np.sum(np.exp(lam * shifted))) / lam
        return lse_val

    def lse_gradient(h, lam):
        """
        Gradient of the LSE objective.
        d/dh [ (1/lam) log(sum exp(lam C_k)) ] = (1/lam) * (1/sum exp(lam C_k)) * sum (exp(lam C_k) * lam * dC_k/dh)
        = sum( w_k * dC_k/dh ) where w_k = exp(lam C_k) / sum(exp(lam C_k))
        """
        corrs = compute_corrs(h)
        max_c = np.max(corrs)
        shifted = corrs - max_c
        exp_terms = np.exp(lam * shifted)
        weights = exp_terms / np.sum(exp_terms)
        
        # Gradient of C_k(h) = sum_i h_i (1 - h_{i+k}) 
        # Wait, the correlation computed by np.correlate(h, 1-h) is sum h_{i+k}(1-h_i).
        # Let's derive gradient of C_k = sum_i h_{i+k} (1-h_i) = sum_i h_{i+k} - sum_i h_{i+k} h_i.
        # d/dh_j C_k = delta_{j, i+k} - (delta_{j, i+k} h_i + h_{i+k} delta_{j, i})
        # This is complex to vectorize perfectly without matching np.correlate's exact index convention.
        # 
        # Alternative: Use numerical gradient? SLSQP can use finite differences.
        # For N=100, numerical gradient is 100 evaluations. 
        # If we do 100 iterations, that's 10,000 evaluations. 
        # Each evaluation is O(N) for correlation? No, np.correlate is O(N^2) or FFT.
        # For N=100, it's fast.
        # Let's let SLSQP compute gradients numerically to ensure correctness with the exact np.correlate definition.
        # We will return None for jac to trigger finite differences.
        return None

    def constraints_func(h):
        # Return value for equality constraint: sum(h) - 50 = 0
        return np.sum(h) - N / 2.0

    # Constraint definition for SLSQP
    constraints = [
        {
            'type': 'eq',
            'fun': constraints_func
        }
    ]

    # Bounds
    bounds = [(0.0, 1.0) for _ in range(N)]

    best_h = np.full(N, 0.5)
    best_val = objective(best_h)

    # Multi-start with annealed lambda
    # Start with low lambda (smoother, easier to navigate) and increase (sharper, closer to true max)
    lambdas = [5.0, 10.0, 20.0, 50.0, 100.0]
    
    # Initial points:
    # 1. Uniform 0.5
    # 2. Alternating 0.0/1.0 (repaired)
    # 3. Random Gaussian (repaired)
    
    def repair(h):
        h = np.clip(h, 0.0, 1.0)
        diff = N / 2.0 - np.sum(h)
        if abs(diff) > 1e-9:
            # Water filling
            for _ in range(50):
                h = np.clip(h, 0.0, 1.0)
                current_sum = np.sum(h)
                rem = N / 2.0 - current_sum
                if abs(rem) < 1e-9:
                    break
                if rem > 0:
                    mask = h < 1.0
                else:
                    mask = h > 0.0
                if np.sum(mask) == 0:
                    break
                h[mask] += rem / np.sum(mask)
        return h

    initial_points = []
    # 1. Uniform
    initial_points.append(np.full(N, 0.5))
    
    # 2. Alternating
    h_alt = np.zeros(N)
    h_alt[1::2] = 1.0
    initial_points.append(repair(h_alt))
    
    # 3. Random
    for _ in range(3):
        h_rand = np.random.uniform(0, 1, N)
        initial_points.append(repair(h_rand))

    for lam in lambdas:
        for h0 in initial_points:
            if time.time() - start_time > time_limit - 5.0:
                break
            
            # SLSQP
            try:
                res = minimize(
                    lse_objective,
                    h0,
                    args=(lam,),
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints,
                    options={'maxiter': 200, 'ftol': 1e-8, 'disp': False}
                )
                
                h_opt = res.x
                # Repair to ensure strict validity
                h_opt = repair(h_opt)
                
                # Evaluate with the RAW objective (max, not LSE)
                val = objective(h_opt)
                
                if val < best_val:
                    best_val = val
                    best_h = h_opt.copy()
                    # If we found a very good value, we can update initial points for next lambda
                    # to stay in the neighborhood of the best solution
                    initial_points[0] = best_h.copy()
                    
            except Exception:
                pass

    # Final check
    if not np.all(np.isfinite(best_h)) or np.abs(np.sum(best_h) - N/2.0) > 1e-9 * N:
        return np.full(N, 0.5)
        
    return best_h

# EVOLVE-BLOCK-END
