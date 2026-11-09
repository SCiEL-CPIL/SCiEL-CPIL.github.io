"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    # Deadline guard: the evaluator kills at 630s wall-clock.
    t_start = time.time()
    DEADLINE = 620.0  # 10s safety margin

    # Seed fallback
    seed_result = [4.0 * (i + 1) for i in range(12)]

    try:
        import numpy as np
        from scipy.optimize import minimize
        from scipy.special import eval_laguerre, gammaln

        # The Cohn-Gonçalves construction for Fourier sign uncertainty with
        # m double roots and n=1 dimension.
        #
        # The refined bound C4 depends on the root locations a_1,...,a_12.
        # A natural functional for the L4-based uncertainty bound is:
        #   C4 ~ (∫_0^∞ |P(x)|^4 dx) / (∫_0^∞ |P(x)|^2 dx)^2
        # where P(x) = ∏_{i=1}^{12} (x - a_i)^2
        #
        # However, for the Fourier sign uncertainty problem, the relevant
        # functional involves the Laguerre polynomial weight and the
        # specific structure of the sign-changing functions.
        #
        # A more precise formulation: for the Cohn-Gonçalves construction,
        # the bound involves the ratio of certain integrals with the
        # Laguerre weight. Let's use a computable surrogate.
        #
        # For practical optimization, we'll minimize a functional that
        # captures the essential structure: the spread and clustering of
        # roots affects the L4 norm ratio.

        def compute_C4_bound(roots):
            """Compute a surrogate for the C4 upper bound given 12 root locations."""
            if len(roots) != 12:
                return 1e10
            if any(r <= 0 for r in roots):
                return 1e10
            
            # Sort roots for consistent processing
            a = np.sort(roots)
            
            # P(x) = prod (x - a_i)^2
            # |P(x)|^2 = prod (x - a_i)^4
            # |P(x)|^4 = prod (x - a_i)^8
            
            # Use logarithmic computation for stability
            # log(|P(x)|^2) = 4 * sum log|x - a_i|
            # log(|P(x)|^4) = 8 * sum log|x - a_i|
            
            # The bound C4 involves integrals with Laguerre-type weights.
            # For the n=1 case, the weight is typically e^{-x} or x^{n-1}e^{-x} = e^{-x}.
            
            # A practical surrogate: the C4 bound is related to the ratio
            # of the L4 norm to the L2 norm squared, weighted appropriately.
            
            # Use Gauss-Laguerre quadrature for the integrals
            # ∫_0^∞ f(x) e^{-x} dx ≈ sum w_i f(x_i)
            
            # Get Gauss-Laguerre nodes and weights
            x_nodes, w_nodes = np.polynomial.laguerre.laggauss(64)
            
            # Compute log|P(x)| at quadrature nodes
            # log|P(x)| = 2 * sum_i log|x - a_i|
            
            log_abs_P = np.zeros(len(x_nodes))
            for i, node in enumerate(x_nodes):
                # Sum of log|node - a_j| for all j
                diffs = np.abs(node - a)
                # Avoid log(0) by clipping
                diffs = np.maximum(diffs, 1e-30)
                log_abs_P[i] = 2.0 * np.sum(np.log(diffs))
            
            # |P(x)|^2 * e^{-x} and |P(x)|^4 * e^{-x}
            # Use log-sum-exp for numerical stability
            log_P2_weighted = 2.0 * log_abs_P - x_nodes  # log(|P|^2 * e^{-x})
            log_P4_weighted = 4.0 * log_abs_P - x_nodes  # log(|P|^4 * e^{-x})
            
            # Integrate using quadrature weights
            # ∫ |P|^2 e^{-x} dx ≈ sum w_i * |P(x_i)|^2 * e^{-x_i}
            # = sum w_i * exp(log_P2_weighted[i])
            
            # Use log-sum-exp for stability
            def log_sum_exp(log_vals, weights):
                """Compute log(sum w_i * exp(log_vals_i))"""
                max_log = np.max(log_vals)
                if max_log == -np.inf:
                    return -np.inf
                # log(sum w_i * exp(log_vals_i)) = max_log + log(sum w_i * exp(log_vals_i - max_log))
                exp_shifted = np.exp(log_vals - max_log)
                weighted_sum = np.sum(weights * exp_shifted)
                if weighted_sum <= 0:
                    return -np.inf
                return max_log + np.log(weighted_sum)
            
            log_int_P2 = log_sum_exp(log_P2_weighted, w_nodes)
            log_int_P4 = log_sum_exp(log_P4_weighted, w_nodes)
            
            # C4 ~ int_P4 / (int_P2)^2
            # log(C4) = log_int_P4 - 2*log_int_P2
            log_C4 = log_int_P4 - 2.0 * log_int_P2
            
            if np.isinf(log_C4) or np.isnan(log_C4):
                return 1e10
            
            return math.exp(log_C4)

        # Optimize the 12 root locations
        # Initial guess: use Laguerre polynomial zeros scaled appropriately
        n_lag = 12
        c = []
        for k in range(n_lag + 1):
            val = ((-1) ** k) * math.comb(n_lag + k, k) / math.factorial(k)
            c.append(val)
        c_rev = list(reversed(c))
        lag_roots = np.roots(c_rev)
        real_lag_roots = sorted([r.real for r in lag_roots if abs(r.imag) < 1e-10 and r.real > 0])
        
        if len(real_lag_roots) == 12:
            # Scale to reasonable range
            lag_min, lag_max = real_lag_roots[0], real_lag_roots[-1]
            scale = 44.0 / (lag_max - lag_min) if lag_max > lag_min else 1.0
            x0 = np.array([(r - lag_min) * scale + 4.0 for r in real_lag_roots])
        else:
            x0 = np.array(seed_result)
        
        # Define objective for scipy (minimize)
        def objective(x):
            if time.time() - t_start > DEADLINE:
                return 1e10
            return compute_C4_bound(x)
        
        # Constraints: all roots positive, sorted
        # Use bounds for positivity
        bounds = [(0.01, 200.0)] * 12
        
        # Try SLSQP optimization
        best_result = x0.copy()
        best_obj = objective(x0)
        
        # Multiple restarts with different initial guesses
        restarts = [
            x0,
            np.array(seed_result),
            np.linspace(2, 60, 12),
            np.array([2.0, 4.0, 6.0, 8.0, 10.0, 14.0, 18.0, 24.0, 30.0, 36.0, 42.0, 48.0]),
        ]
        
        for start in restarts:
            if time.time() - t_start > DEADLINE - 30:
                break
            
            try:
                result = minimize(
                    objective,
                    start,
                    method='SLSQP',
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-10, 'disp': False}
                )
                
                if result.success or (result.fun < best_obj and np.all(result.x > 0)):
                    if np.all(result.x > 0):
                        best_obj = result.fun
                        best_result = result.x.copy()
            except Exception:
                pass
        
        # Final polish with tighter tolerance if time permits
        if time.time() - t_start < DEADLINE - 10:
            try:
                result = minimize(
                    objective,
                    best_result,
                    method='SLSQP',
                    bounds=bounds,
                    options={'maxiter': 500, 'ftol': 1e-12, 'disp': False}
                )
                if result.success or (result.fun < best_obj and np.all(result.x > 0)):
                    if np.all(result.x > 0):
                        best_obj = result.fun
                        best_result = result.x.copy()
            except Exception:
                pass
        
        # Sort the final result
        best_result = sorted(best_result)
        
        # Verify all positive
        if all(r > 0 for r in best_result):
            return list(best_result)
        else:
            return seed_result
            
    except Exception:
        pass

    # Fallback to seed
    return seed_result
# EVOLVE-BLOCK-END
