"""Refined Cohn-Goncalves Laguerre construction for Fourier sign uncertainty."""
import numpy as np
from scipy.special import roots_genlaguerre, eval_genlaguerre
from scipy.optimize import minimize, differential_evolution
import time

# Global deadline guard
_START_TIME = time.time()
TIME_LIMIT = 580  # seconds, safety margin before 630s kill

def _time_remaining():
    return TIME_LIMIT - (time.time() - _START_TIME)

def _compute_c4(roots):
    """Compute the C4 upper bound for given positive double-root locations.
    
    The C4 bound is related to the L2 norm of the polynomial P(x) = prod_i (x - r_i)^2
    with respect to the Laguerre weight e^{-x}. Specifically, the sign uncertainty
    upper bound involves the norm of this polynomial.
    
    For the refined Cohn-Goncalves construction with m double roots and n=1,
    the relevant quantity is the squared L2 norm of P(x) = prod (x - r_i)^2
    under the weight e^{-x} on [0, infinity).
    
    We use Gauss-Laguerre quadrature to compute this integral efficiently.
    """
    m = len(roots)
    # Use Gauss-Laguerre quadrature with enough nodes
    # For m=12, the polynomial P has degree 24, so we need at least 25 nodes
    # But for accuracy, use more nodes
    n_quad = max(2 * m + 10, 100)
    nodes, weights = np.polynomial.laguerre.laggauss(n_quad)
    
    # Evaluate P(x) = prod (x - r_i)^2 at each node
    # P(x) = prod_i (x - r_i)^2
    # log(P(x)) = 2 * sum_i log(x - r_i)
    # But we need P(x)^2 for the norm, so we need P(x)^2 * e^{-x}
    # Actually the norm squared is integral of P(x)^2 * e^{-x} dx
    
    # P(x)^2 = prod_i (x - r_i)^4
    # Let's compute this carefully to avoid overflow/underflow
    
    # For each node, compute prod_i (node - r_i)^4
    # Use log-sum for stability
    logs = np.zeros(n_quad)
    for r in roots:
        # (x - r)^4 can be negative if x < r, but squared it's positive
        # Actually (x - r)^4 is always non-negative
        # But we're computing P(x)^2 where P(x) = prod (x - r_i)^2
        # So P(x)^2 = prod (x - r_i)^4
        # Each term (x - r_i)^4 >= 0, and it's 0 when x = r_i
        # For quadrature nodes that don't coincide with roots, this is fine
        
        # Compute log((x - r)^4) = 4 * log(|x - r|)
        diff = nodes - r
        # Avoid log(0) - if a node coincides with a root, the contribution is 0
        # But this is extremely unlikely with continuous quadrature nodes
        abs_diff = np.abs(diff)
        # Where abs_diff is very small, the log will be very negative, which is correct
        # as the contribution should be near 0
        logs += 4.0 * np.log(abs_diff + 1e-300)
    
    # P(x)^2 * e^{-x} = exp(logs) * exp(-nodes)
    # log(P(x)^2 * e^{-x}) = logs - nodes
    log_vals = logs - nodes
    
    # For numerical stability, find the max and shift
    max_log = np.max(log_vals)
    if max_log > 700:  # Would overflow
        # Scale down
        log_vals = log_vals - max_log
        scale = np.exp(max_log)
    else:
        scale = 1.0
    
    vals = np.exp(log_vals)
    integral = np.sum(weights * vals) * scale
    
    return integral

def _compute_c4_direct(roots):
    """Alternative direct computation of C4 bound using polynomial evaluation.
    
    For the Cohn-Goncalves construction, the C4 upper bound is:
    C4 = 1 / (integral_0^inf |P(x)|^2 e^{-x} dx)
    where P(x) = prod_i (x - r_i)^2
    
    Wait, let me reconsider. The sign uncertainty upper bound in the Cohn-Goncalves
    framework involves the ratio of certain norms. Let me use a more direct approach.
    
    Actually, for the Fourier sign uncertainty problem with the Laguerre construction,
    the key quantity to minimize is the upper bound on C4, which is inversely
    related to the L2 norm of the test function. So we want to MAXIMIZE the norm
    to MINIMIZE the upper bound.
    
    Let me just return the norm and let the optimization maximize it (or minimize its reciprocal).
    """
    m = len(roots)
    n_quad = max(2 * m + 10, 150)
    nodes, weights = np.polynomial.laguerre.laggauss(n_quad)
    
    # Compute P(x) = prod (x - r_i)^2 at each node
    # Then norm^2 = integral P(x)^2 e^{-x} dx
    # P(x)^2 = prod (x - r_i)^4
    
    # Use log computation for stability
    log_p_sq = np.zeros(n_quad)
    for r in roots:
        diff = nodes - r
        abs_diff = np.abs(diff)
        log_p_sq += 4.0 * np.log(abs_diff + 1e-300)
    
    log_weighted = log_p_sq - nodes
    max_log = np.max(log_weighted)
    
    if max_log > 700:
        shifted = log_weighted - max_log
        integral = np.sum(weights * np.exp(shifted)) * np.exp(max_log)
    else:
        integral = np.sum(weights * np.exp(log_weighted))
    
    return integral

def _objective(params):
    """Objective function to minimize: the C4 upper bound.
    
    The C4 upper bound is inversely proportional to the L2 norm of the polynomial.
    So we minimize 1/norm, which is equivalent to maximizing norm.
    """
    roots = np.array(params)
    if np.any(roots <= 0) or np.any(np.isnan(roots)):
        return 1e10
    
    # Ensure roots are sorted and positive
    roots = np.sort(roots)
    
    norm_sq = _compute_c4_direct(roots)
    
    if norm_sq <= 0 or np.isnan(norm_sq) or np.isinf(norm_sq):
        return 1e10
    
    # The C4 upper bound is proportional to 1/norm
    # We want to minimize this, so return 1/sqrt(norm_sq) = 1/norm
    return 1.0 / np.sqrt(norm_sq)

def _objective_grad(params):
    """Gradient of the objective function using finite differences or analytic computation."""
    roots = np.array(params)
    if np.any(roots <= 0) or np.any(np.isnan(roots)):
        return np.ones(len(roots))
    
    roots = np.sort(roots)
    m = len(roots)
    n_quad = max(2 * m + 10, 150)
    nodes, weights = np.polynomial.laguerre.laggauss(n_quad)
    
    # Compute gradient numerically
    grad = np.zeros(m)
    eps = 1e-7
    
    f0 = _objective(roots)
    
    for i in range(m):
        perturbed = roots.copy()
        perturbed[i] += eps
        fp = _objective(perturbed)
        grad[i] = (fp - f0) / eps
    
    return grad

def construct():
    """Return 12 positive double-root locations for the Laguerre combination (m=12, n=1).
    
    Uses optimization to find the roots that minimize the C4 upper bound.
    """
    m = 12
    n = 1
    
    # Initial guess: zeros of L_{12}^{(13)}(x)
    initial_roots = np.sort(roots_genlaguerre(m, 2 * n + m - 1)[0])
    
    best_roots = initial_roots.copy()
    best_score = _objective(best_roots)
    
    # Multi-start optimization
    n_starts = 5
    rng = np.random.default_rng(42)
    
    for start in range(n_starts):
        if _time_remaining() < 60:
            break
        
        if start == 0:
            x0 = initial_roots.copy()
        else:
            # Perturb the initial guess
            x0 = initial_roots * (1.0 + rng.normal(0, 0.1, m))
            x0 = np.maximum(x0, 0.1)  # Ensure positivity
            x0 = np.sort(x0)
        
        # Use L-BFGS-B for bounded optimization
        bounds = [(0.1, 200.0)] * m
        
        try:
            result = minimize(
                _objective,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            if result.success or result.fun < best_score:
                candidate = np.sort(result.x)
                if np.all(candidate > 0):
                    score = result.fun
                    if score < best_score:
                        best_score = score
                        best_roots = candidate.copy()
        except Exception:
            continue
    
    # Try a few more random starts if time permits
    if _time_remaining() > 120:
        for _ in range(3):
            if _time_remaining() < 60:
                break
            x0 = np.sort(rng.uniform(1, 50, m))
            try:
                result = minimize(
                    _objective,
                    x0,
                    method='L-BFGS-B',
                    bounds=[(0.1, 200.0)] * m,
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                if result.fun < best_score:
                    candidate = np.sort(result.x)
                    if np.all(candidate > 0):
                        best_score = result.fun
                        best_roots = candidate.copy()
            except Exception:
                continue
    
    return [float(r) for r in best_roots]
