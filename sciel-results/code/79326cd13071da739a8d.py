import numpy as np
from scipy.optimize import minimize, basinhopping
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 630s kill

    m = 12
    n = 1

    # Precompute Gauss-Laguerre nodes and weights for alpha=0
    # We need to integrate P(x)^2 * x * exp(-x).
    # Let f(x) = P(x)^2 * x.
    # Integral = sum w_i * f(x_i)
    # We need enough points to integrate a polynomial of degree 24 exactly.
    # Gauss-Laguerre with N points integrates polynomials of degree 2N-1 exactly.
    # So N >= 13. Let's use N=20 for safety and speed.
    x_nodes, w_weights = np.polynomial.laguerre.laggauss(20)

    def compute_cost(log_roots):
        """
        Compute the cost function (upper bound on C4) given log roots.
        Cost = 1 / ||P||^2_Laguerre
        We want to minimize this, which means maximizing ||P||^2.
        """
        elapsed = time.time() - start_time
        if elapsed > deadline:
            return 1e20

        # Check for early termination more frequently if needed, but time.time() is cheap.
        
        roots = np.exp(log_roots)
        
        # Check validity
        if np.any(roots <= 0) or np.any(np.isnan(roots)) or np.any(np.isinf(roots)):
            return 1e20

        # Evaluate P(x) at nodes
        # P(x) = prod(x - r_i)
        # Vectorized: x_nodes[:, None] - roots[None, :]
        # This creates a (20, 12) matrix.
        # prod along axis 1 gives P(x_nodes)
        
        # Optimization: Use log-sum-exp? No, roots are positive, x_nodes are positive.
        # Just direct product is fine.
        
        # Handle potential overflow/underflow? 
        # Roots are likely O(1) to O(100). x_nodes are O(1) to O(100).
        # Products might be large. 
        # log(P(x)) = sum log(x - r_i). 
        # But we need P(x)^2. 
        # log(P(x)^2) = 2 * sum log(x - r_i).
        # If x < r_i, log is undefined. 
        # Roots are positive. Nodes are positive.
        # If x_i < r_j, the term (x_i - r_j) is negative. 
        # P(x) is real. P(x)^2 is positive.
        # We can compute P(x) directly.
        
        diffs = x_nodes[:, None] - roots[None, :]
        P_vals = np.prod(diffs, axis=1)
        
        # Integrand: P(x)^2 * x * exp(-x) is handled by weights w_i which include exp(-x)?
        # No, np.polynomial.laguerre.laggauss returns weights for int f(x) exp(-x) dx.
        # So we need to compute sum w_i * P(x_i)^2 * x_i.
        
        integrand = P_vals**2 * x_nodes
        norm_sq = np.sum(w_weights * integrand)
        
        if norm_sq <= 1e-10 or not np.isfinite(norm_sq):
            return 1e20
        
        return 1.0 / norm_sq

    # 2. Optimization Strategy
    # Multi-start L-BFGS-B
    # This is structurally different from DE: deterministic local search from multiple seeds.
    
    best_val = 1e20
    best_log_roots = None

    # Bounds for log roots
    # Roots must be positive.
    # Seed [4, 8, 16, ...] -> logs [1.4, 2.0, 2.7, ...]
    # Let's use a wide range for log roots: [log(0.1), log(1000)]
    lower_bound = np.log(0.1)
    upper_bound = np.log(1000.0)
    bounds_log = [(lower_bound, upper_bound)] * 12

    def update_best(val, x):
        nonlocal best_val, best_log_roots
        if val < best_val:
            best_val = val
            best_log_roots = x.copy()

    # Generate diverse starting points
    initial_points = []
    
    # 1. Geometric seed (original)
    r_seed = 4.0 * (2.0 ** np.arange(12))
    initial_points.append(np.log(r_seed))
    
    # 2. Random restarts
    rng = np.random.RandomState(42)
    for _ in range(20):
        log_r = rng.uniform(lower_bound, upper_bound, 12)
        initial_points.append(log_r)
        
    # 3. Perturbed seed
    for _ in range(5):
        noise = rng.normal(0, 0.5, 12)
        initial_points.append(np.log(r_seed) + noise)

    # Sort initial points by initial cost to try the best ones first?
    # Or just iterate.
    
    for i, init_x in enumerate(initial_points):
        if time.time() - start_time > deadline - 10:
            break
            
        try:
            res_lf = minimize(
                compute_cost, 
                init_x, 
                method='L-BFGS-B', 
                bounds=bounds_log, 
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            update_best(res_lf.fun, res_lf.x)
        except Exception:
            pass

    # If we have time, try Basin Hopping from the best found so far
    if time.time() - start_time < deadline - 30 and best_log_roots is not None:
        try:
            # Basin Hopping requires a step function
            def step(x):
                # Random step in log space
                noise = np.random.normal(0, 0.5, 12)
                return x + noise
            
            res_bh = basinhopping(
                compute_cost, 
                best_log_roots, 
                stepfunc=step, 
                niter=50, 
                T=1.0, 
                stepsize=0.5, 
                minimizer_kwargs={'method': 'L-BFGS-B', 'bounds': bounds_log},
                seed=42
            )
            update_best(res_bh.fun, res_bh.x)
        except Exception:
            pass

    # Final Fallback
    if best_log_roots is None:
        best_log_roots = np.log(4.0 * (2.0 ** np.arange(12)))
        best_val = compute_cost(best_log_roots)

    # Convert back to roots
    best_roots = np.exp(best_log_roots)
    
    # Ensure positivity and sorting
    best_roots = np.sort(best_roots)
    best_roots = np.maximum(best_roots, 1e-10)
    
    return list(best_roots)

# Main entry point
if __name__ == "__main__":
    roots = construct()
    print(roots)
