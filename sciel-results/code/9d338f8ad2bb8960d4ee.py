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
    Uses Convex Optimization with Auxiliary Variable t.
    
    Strategy:
    1. The objective is to minimize max_k C_k(h).
    2. We introduce an auxiliary variable t such that we minimize t
       subject to C_k(h) <= t for all k.
    3. This transforms the problem into a standard convex optimization problem
       (minimize linear function subject to quadratic inequality constraints, 
       though C_k(h) = sum h_i - sum h_i h_{i+k} is quadratic. 
       Wait, is it convex? 
       C_k(h) = sum_{i} h_{i+k}(1-h_i) = sum h_{i+k} - sum h_{i+k}h_i.
       The term sum h_{i+k}h_i is a quadratic form. The Hessian has 1s on off-diagonals 
       corresponding to shifts. For a single shift k, the matrix has 1s at (i, i+k) and (i+k, i).
       This matrix is not positive semi-definite for all k. 
       However, the max of these functions is the objective. 
       Actually, let's look at the structure.
       The problem is to minimize t.
       Constraints: t - C_k(h) >= 0.
       This is a standard formulation for minimizing the maximum of a set of functions.
       If the functions C_k(h) are convex, the problem is convex.
       C_k(h) = sum_i h_{i+k} - sum_i h_{i+k} h_i.
       Let's check convexity of -sum h_i h_{i+k}.
       Hessian of -h_i h_j is - (E_{ij} + E_{ji}). Eigenvalues are +/- 1. Not convex.
       So the problem is NOT convex in general.
       
       However, SLSQP can handle non-convex problems. 
       The key advantage of the auxiliary variable formulation is that the gradient of the 
       objective with respect to t is 1, and the gradients of the constraints are well-defined.
       This allows SLSQP to navigate the "ridge" of the max function more effectively than 
       trying to smooth the max function itself, which can suffer from vanishing gradients 
       or poor conditioning when lambda is high.
       
       We will use multiple restarts with different initial points.
    """
    start_time = time.time()
    time_limit = 140.0  # Safety margin

    # Pre-compute indices for correlation terms to speed up constraint evaluation
    # np.correlate(h, 1-h, 'full') has length 2N-1.
    # Let's just use np.correlate for simplicity, it's fast for N=100.
    
    def get_corrs(h):
        return np.correlate(h, 1.0 - h, mode="full")

    def objective_func(x):
        """
        x = [h_0, ..., h_{N-1}, t]
        Minimize t
        """
        h = x[:N]
        t = x[N]
        return t

    def constraints_func(x):
        """
        Returns array of constraint violations.
        We need t - C_k(h) >= 0 for all k.
        In scipy minimize, 'ineq' constraints are g(x) >= 0.
        So we return t - C_k(h).
        """
        h = x[:N]
        t = x[N]
        corrs = get_corrs(h)
        return t - corrs

    # We also need the sum constraint: sum(h) = N/2
    # This is an equality constraint.
    def sum_constraint(x):
        h = x[:N]
        return np.sum(h) - N / 2.0

    constraints = [
        {'type': 'ineq', 'fun': constraints_func},
        {'type': 'eq', 'fun': sum_constraint}
    ]

    # Bounds: h_i in [0, 1], t is unbounded (but will be pushed down)
    bounds = [(0.0, 1.0) for _ in range(N)]
    bounds.append((0.0, None)) # t can be any non-negative number, though it should be small

    best_h = np.full(N, 0.5)
    best_val = objective(best_h)

    # Initial points for h
    def repair(h):
        h = np.clip(h, 0.0, 1.0)
        diff = N / 2.0 - np.sum(h)
        if abs(diff) > 1e-9:
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
    for _ in range(5):
        h_rand = np.random.uniform(0, 1, N)
        initial_points.append(repair(h_rand))

    # 4. Block structure: 0.5 for first half, 0.5 for second half? No, that's uniform.
    # Try blocks of 0 and 1.
    h_block = np.zeros(N)
    h_block[0:50] = 1.0
    initial_points.append(repair(h_block))

    for h0 in initial_points:
        if time.time() - start_time > time_limit - 10.0:
            break
        
        # Initialize t to be slightly larger than the max correlation of h0
        corrs0 = get_corrs(h0)
        t0 = np.max(corrs0) + 0.1
        
        x0 = np.concatenate([h0, [t0]])
        
        try:
            res = minimize(
                objective_func,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 300, 'ftol': 1e-9, 'disp': False}
            )
            
            h_opt = res.x[:N]
            t_opt = res.x[N]
            
            # Repair h to ensure strict validity
            h_opt = repair(h_opt)
            
            # Evaluate with the RAW objective
            val = objective(h_opt)
            
            if val < best_val:
                best_val = val
                best_h = h_opt.copy()
                # Update the first initial point to be the best found so far
                initial_points[0] = best_h.copy()
                
        except Exception as e:
            pass

    # Final check
    if not np.all(np.isfinite(best_h)) or np.abs(np.sum(best_h) - N/2.0) > 1e-9 * N:
        return np.full(N, 0.5)
        
    return best_h

# EVOLVE-BLOCK-END
