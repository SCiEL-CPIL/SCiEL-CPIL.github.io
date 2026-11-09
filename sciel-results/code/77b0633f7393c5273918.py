import numpy as np
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    deadline = start_time + 580  # Safety margin

    # The polynomial P(x) has 12 double roots, so P(x) = Q(x)^2
    # where Q(x) is a monic polynomial of degree 12 with roots r_1, ..., r_12.
    # We want to minimize the upper bound on C4, which is proportional to 1 / ||P||^2.
    # Thus we maximize ||P||^2 = ||Q^2||^2 = \int_0^\infty Q(x)^4 e^{-x} dx.
    
    # We use Gaussian quadrature to compute the integral exactly.
    # Q(x) is degree 12, so Q(x)^4 is degree 48.
    # A Gaussian quadrature rule with n points is exact for polynomials of degree 2n-1.
    # We need 2n - 1 >= 48 => 2n >= 49 => n >= 24.5 => n = 25.
    
    nodes, weights = np.polynomial.laguerre.laggauss(25)
    
    def objective(log_roots):
        # log_roots are the natural logs of the 12 roots
        roots = np.exp(log_roots)
        
        # Q(x) = prod_{j=1}^{12} (x - r_j)
        # Evaluate Q at the 25 nodes
        # nodes is shape (25,), roots is shape (12,)
        # We need Q(nodes) for each node.
        # Q(node_i) = prod_j (node_i - r_j)
        
        # Vectorized computation:
        # diff is shape (25, 12)
        diff = nodes[:, np.newaxis] - roots[np.newaxis, :]
        
        # Log of Q to avoid overflow/underflow? 
        # Roots are positive. Nodes are positive.
        # If roots are large, (node - root) can be negative. 
        # Q(x) can be negative. Q(x)^4 is always positive.
        # It is safer to compute the product and then power 4.
        # However, if roots are very large, products can overflow.
        # Let's check the scale. Laguerre roots of L_12 are around 1 to 30.
        # Q(x) values might be large.
        # log(abs(Q)) = sum log(abs(node - root)).
        # sign(Q) = prod sign(node - root).
        # Q^4 = exp(4 * sum log(abs(node - root))).
        
        abs_diff = np.abs(diff)
        # Handle zeros (if a root is exactly a node, which is unlikely but possible)
        # If diff is 0, log is -inf, Q is 0, Q^4 is 0.
        # We can add a small epsilon or handle separately.
        # Given the continuous nature, hitting exactly 0 is measure zero.
        
        log_abs_Q = np.sum(np.log(abs_diff + 1e-300), axis=1)
        
        # Q^4 = exp(4 * log_abs_Q)
        # Integral = sum weights * Q(nodes)^4
        # log_integral = log(sum weights * exp(4 * log_abs_Q))
        
        # To compute log(sum exp(...)) stably:
        # Let y_i = 4 * log_abs_Q_i + log(weights_i)
        # log_sum = max(y) + log(sum(exp(y - max(y))))
        
        y = 4.0 * log_abs_Q + np.log(weights)
        max_y = np.max(y)
        if np.isinf(max_y):
            # This would happen if Q is 0 everywhere, which is impossible
            return 0.0 
        
        log_integral = max_y + np.log(np.sum(np.exp(y - max_y)))
        
        # We want to minimize -log_integral
        return -log_integral

    # Initial guess: Roots of the Laguerre polynomial L_12(x)
    # These are the nodes for Laguerre-Gauss quadrature of order 12
    init_roots, _ = np.polynomial.laguerre.laggauss(12)
    init_log_roots = np.log(init_roots)
    
    best_log_roots = init_log_roots
    best_score = objective(init_log_roots)
    
    # Optimization
    # L-BFGS-B is good for this size and smoothness
    try:
        res = minimize(
            objective, 
            init_log_roots, 
            method='L-BFGS-B', 
            options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        if res.success and res.fun < best_score:
            best_score = res.fun
            best_log_roots = res.x
    except Exception:
        pass

    # Try a few random restarts if time permits, just in case L_12 roots aren't the global optimum
    # But L_12 roots are extremely likely to be the optimum for this symmetric functional.
    # The functional is invariant under permutations of roots.
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed < deadline - 10:
        # Do a couple of random restarts to be safe
        for i in range(5):
            if time.time() - start_time > deadline - 10:
                break
            # Random log-roots, scaled similarly to L_12 roots
            # L_12 roots range approx [0.5, 30]
            # Log range approx [-0.7, 3.4]
            rand_log_roots = np.random.uniform(-2.0, 4.0, size=12)
            try:
                res_r = minimize(
                    objective, 
                    rand_log_roots, 
                    method='L-BFGS-B', 
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                if res_r.success and res_r.fun < best_score:
                    best_score = res_r.fun
                    best_log_roots = res_r.x
            except Exception:
                pass

    final_roots = np.exp(best_log_roots)
    
    # Ensure roots are positive
    final_roots = np.maximum(final_roots, 1e-6)
    
    return list(final_roots)
