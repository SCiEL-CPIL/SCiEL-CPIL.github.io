"""
Refined uncertainty inequality: Global optimization for Laguerre double-root construction.
Uses Differential Evolution to escape local minima, followed by L-BFGS-B refinement.
"""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.special import gammaln

_START = time.time()
_DEADLINE = 630.0
_MARGIN = 30.0


def _time_left():
    return _DEADLINE - _MARGIN - time.time()


def _compute_c4_bound(roots):
    """
    Compute the upper bound on C4 from the Laguerre polynomial construction.
    P(x) = prod_{i=0}^{11} (x - r_i)^2, degree 24.
    Norm: ||P||^2 = integral_0^inf P(x)^2 * x^2 * exp(-x) dx
    C4_bound = 1 / sqrt(||P||^2)
    """
    # Build P(x) = [prod (x - r_i)]^2
    # q(x) = prod (x - r_i)
    q_coeff = np.array([1.0])
    for r in roots:
        q_coeff = np.convolve(q_coeff, np.array([1.0, -r]))
    p_coeff = np.convolve(q_coeff, q_coeff)
    
    # ||P||^2 = sum_{i,j} p_i * p_j * Gamma(i+j+3)  (alpha=2, so i+j+alpha+1 = i+j+3)
    n = len(p_coeff)
    indices = np.arange(n)
    gamma_args = indices[:, None] + indices[None, :] + 3
    log_gv = gammaln(gamma_args)
    log_gv = np.clip(log_gv, -700, 700)
    gv = np.exp(log_gv)
    
    norm_sq = p_coeff @ (gv @ p_coeff)
    
    if norm_sq <= 0 or not np.isfinite(norm_sq):
        return np.inf
    
    return 1.0 / np.sqrt(norm_sq)


def _objective_log(params):
    """
    Objective function for optimization in log-space.
    Returns the C4 bound. Lower is better.
    """
    roots = np.exp(params)
    # Sanity checks
    if np.any(roots > 1e6) or np.any(roots < 1e-6):
        return 1e10
    return _compute_c4_bound(roots)


def _objective_log_neg(params):
    """
    Objective for maximization (used internally if needed, but we minimize bound).
    """
    return _objective_log(params)


def construct():
    """
    Construct 12 positive double-root locations for the Laguerre combination
    (m=12, n=1) that minimize the upper bound on C4.
    Returns a list of 12 positive root locations.
    """
    deadline = _START + _DEADLINE - _MARGIN
    
    best_roots = None
    best_score = np.inf
    
    # Define bounds for log(roots)
    # Roots likely range from 0.1 to 500.
    # log(0.1) ~ -2.3, log(500) ~ 6.2
    bounds = [(-2.0, 6.5)] * 12
    
    # Phase 1: Differential Evolution (Global Search)
    # DE is slow but robust. We limit population size and generations to fit budget.
    # We use a smaller population to allow more generations within the time limit.
    if _time_left() > 180:
        try:
            # Strategy: 'best1bin' is fast, 'best1bin' with high mutation can help.
            # We use 'best1bin' with default parameters but tuned for speed.
            # maxiter is not strictly bounded by time in DE, so we monitor time.
            # We will run DE for a fixed number of iterations that should take ~100-150s.
            # popsize=10 (120 individuals) * 200 generations = 24000 evals.
            # Each eval is fast (~1ms), so ~24s. We can afford more.
            # Let's try popsize=15, maxiter=300.
            
            # To strictly respect the deadline, we wrap the call or rely on DE's internal
            # stopping if it's too slow, but DE doesn't have a time limit.
            # We must ensure it finishes. 
            # Let's run DE with a conservative budget.
            
            start_de_time = time.time()
            de_result = differential_evolution(
                _objective_log,
                bounds=bounds,
                maxiter=400,
                popsize=12,
                tol=1e-9,
                atol=1e-12,
                seed=42,
                init='latinhypercube',
                polish=False  # We will polish manually
            )
            
            if de_result.success or de_result.fun < best_score:
                best_score = de_result.fun
                best_roots = np.exp(de_result.x)
                
            # If DE is taking too long, we might have hit the limit, but 
            # differential_evolution returns when maxiter is reached.
            # We check time after.
            
        except Exception as e:
            print(f"DE failed: {e}")
            pass

    # Phase 2: Local Refinement with L-BFGS-B
    # We start from the best DE solution and a few other promising initial guesses.
    
    initial_guesses_log = []
    if best_roots is not None:
        initial_guesses_log.append(np.log(best_roots))
    
    # Add some structured initial guesses in log-space
    # 1. Geometric progression
    for r0_log in [-1.0, 0.0, 1.0]:
        for q_log in [0.1, 0.2, 0.3]:
            log_roots = r0_log + q_log * np.arange(12)
            # Ensure within bounds
            if np.all(log_roots > -2.0) and np.all(log_roots < 6.5):
                initial_guesses_log.append(log_roots)
                
    # 2. Random starts (if time permits)
    if _time_left() > 120:
        rng = np.random.RandomState(123)
        for _ in range(5):
            log_roots = rng.uniform(-1.0, 5.0, 12)
            initial_guesses_log.append(log_roots)

    for log_init in initial_guesses_log:
        if _time_left() < 30:
            break
        try:
            result = minimize(
                _objective_log,
                x0=log_init,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-14}
            )
            if result.fun < best_score:
                best_score = result.fun
                best_roots = np.exp(result.x)
        except Exception:
            pass

    # Phase 3: Multi-start DE on a smaller scale if time remains? 
    # Or just more L-BFGS starts from perturbed best solution.
    if best_roots is not None and _time_left() > 60:
        log_best = np.log(best_roots)
        rng = np.random.RandomState(42)
        for _ in range(10):
            if _time_left() < 20:
                break
            # Perturb in log space
            noise = rng.normal(0, 0.1, 12)
            log_init = log_best + noise
            # Clip to bounds
            log_init = np.clip(log_init, -2.0, 6.5)
            try:
                result = minimize(
                    _objective_log,
                    x0=log_init,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-14}
                )
                if result.fun < best_score:
                    best_score = result.fun
                    best_roots = np.exp(result.x)
            except Exception:
                pass

    if best_roots is None:
        # Fallback to a reasonable guess
        best_roots = np.array([4.0 * (i + 1) for i in range(12)])
    
    # Ensure sorted, positive
    best_roots = np.sort(best_roots)
    best_roots = np.maximum(best_roots, 1e-6)
    
    return [float(r) for r in best_roots]
