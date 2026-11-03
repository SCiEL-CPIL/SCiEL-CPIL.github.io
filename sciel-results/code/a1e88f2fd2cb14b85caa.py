import numpy as np
from scipy.special import roots_laguerre, eval_genlaguerre, gammaln
from scipy.optimize import minimize
import time

def _compute_c4_bound(roots):
    """
    Compute the C4 upper bound for given double-root positions.
    Uses Gauss-Laguerre quadrature for numerical stability.
    """
    roots = np.asarray(roots, dtype=np.float64)
    if np.any(roots <= 0) or np.any(np.isnan(roots)):
        return np.inf
    
    # P(x) = prod (x - x_i)^2
    # C4 = [∫ P(x)^2 e^{-x} dx] / [∫ P(x) e^{-x} dx]^2
    
    # Use Gauss-Laguerre quadrature
    # The integrals are over [0, infinity) with weight e^{-x}
    n_quad = 150
    x_quad, w_quad = np.polynomial.laguerre.laggauss(n_quad)
    
    # Compute P(x) at quadrature points
    # P(x) = prod (x - x_i)^2
    # Use log-space for stability
    log_P = np.zeros(n_quad)
    for r in roots:
        log_P += 2.0 * np.log(np.abs(x_quad - r) + 1e-300)
    
    P = np.exp(log_P)
    P2 = P * P
    
    int_P = np.sum(w_quad * P)
    int_P2 = np.sum(w_quad * P2)
    
    if int_P <= 0 or np.isinf(int_P) or np.isinf(int_P2):
        return np.inf
    
    c4 = int_P2 / (int_P * int_P)
    return c4

def _geometric_objective(params):
    """
    Objective function for geometric sequence parameterization.
    params = [log_r0, log_q]
    roots_i = r0 * q^i for i=0..11
    """
    log_r0, log_q = params
    r0 = np.exp(log_r0)
    q = np.exp(log_q)
    
    if r0 <= 0 or q <= 0 or not np.isfinite(r0) or not np.isfinite(q):
        return 1e100
    
    # Generate roots
    i = np.arange(12)
    roots = r0 * (q ** i)
    
    # Check if roots are reasonable (positive, not too large)
    if np.any(roots <= 0) or np.any(roots > 1e6):
        return 1e100
    
    return _compute_c4_bound(roots)

def _dyson_gas_initial_guess():
    """
    Generate initial guess using Dyson gas equilibrium concept.
    For repulsion exponent 2 and linear confining potential,
    the equilibrium positions satisfy a mean-field equation.
    
    For Laguerre polynomial zeros with parameter alpha=1,
    the asymptotic density is related to the Marchenko-Pastur law.
    The zeros are approximately distributed according to the
    equilibrium of a Dyson gas with beta=2 (Gaussian Unitary Ensemble-like).
    
    For our case with 12 double roots, we use the known asymptotic
    distribution of Laguerre zeros as a starting point.
    """
    # Standard Laguerre roots as a good starting point
    x0, _ = roots_laguerre(12)
    x0 = np.sort(x0)
    
    # Apply Dyson gas-inspired scaling
    # The mean-field equilibrium for repulsion exponent 2 suggests
    # that roots should spread according to the square root of the
    # confining potential balance.
    
    # For linear potential V(x) = x, the equilibrium density satisfies:
    # 2 * integral (rho(t)/(x-t)) dt = dV/dx = 1
    # This leads to a semi-circular-like distribution scaled appropriately.
    
    # Use the known asymptotic behavior: for large n, Laguerre zeros
    # scale as n * (some function). For n=12, we use the actual roots
    # but perturb them slightly to explore the neighborhood.
    
    return x0

def construct():
    """Return 12 positive double-root locations for the Laguerre combination (m=12, n=1)."""
    deadline = time.time() + 590  # Safety margin
    
    # Phase 1: Dyson gas-inspired initial guesses
    x0 = _dyson_gas_initial_guess()
    
    best_roots = x0.copy()
    best_score = _compute_c4_bound(best_roots)
    
    # Phase 2: Geometric sequence optimization
    # Parameterize as r0 * q^i and optimize (log_r0, log_q)
    if time.time() < deadline - 100:
        # Multiple starts for geometric optimization
        rng = np.random.default_rng(42)
        
        # Initial guesses for (log_r0, log_q)
        # Based on Laguerre roots: first root ~0.09, last root ~47
        # So r0 ~ 0.1, q ~ (47/0.1)^(1/11) ~ 2.3
        initial_params = [
            [np.log(0.1), np.log(2.3)],  # Based on Laguerre root spread
            [np.log(0.05), np.log(2.5)],
            [np.log(0.2), np.log(2.0)],
            [np.log(0.1), np.log(2.0)],
            [np.log(0.1), np.log(2.5)],
        ]
        
        # Add some random perturbations
        for _ in range(10):
            log_r0 = np.log(rng.uniform(0.01, 1.0))
            log_q = np.log(rng.uniform(1.5, 3.0))
            initial_params.append([log_r0, log_q])
        
        for param_init in initial_params:
            if time.time() > deadline - 50:
                break
            
            try:
                result = minimize(
                    _geometric_objective,
                    param_init,
                    method='L-BFGS-B',
                    bounds=[(np.log(1e-4), np.log(10.0)), (np.log(1.01), np.log(5.0))],
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                
                if result.fun < best_score:
                    log_r0, log_q = result.x
                    r0 = np.exp(log_r0)
                    q = np.exp(log_q)
                    i = np.arange(12)
                    candidate = r0 * (q ** i)
                    
                    if np.all(candidate > 0) and not np.any(np.isnan(candidate)):
                        candidate = np.sort(candidate)
                        score = _compute_c4_bound(candidate)
                        if score < best_score:
                            best_score = score
                            best_roots = candidate.copy()
            except Exception:
                pass
    
    # Phase 3: Local refinement of best found roots
    if time.time() < deadline - 30:
        candidates = [best_roots.copy(), x0.copy()]
        
        # Add perturbed versions
        rng = np.random.default_rng(123)
        for _ in range(5):
            perturbed = best_roots * (1 + 0.1 * rng.standard_normal(12))
            perturbed = np.maximum(perturbed, 1e-3)
            perturbed = np.sort(perturbed)
            candidates.append(perturbed)
        
        bounds_local = [(1e-6, 100.0)] * 12
        
        for x_init in candidates:
            if time.time() > deadline - 10:
                break
            
            try:
                result = minimize(
                    _compute_c4_bound,
                    x_init,
                    method='L-BFGS-B',
                    bounds=bounds_local,
                    options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                
                if result.fun < best_score:
                    candidate = result.x
                    if np.all(candidate > 0) and not np.any(np.isnan(candidate)):
                        candidate = np.sort(candidate)
                        score = _compute_c4_bound(candidate)
                        if score < best_score:
                            best_score = score
                            best_roots = candidate.copy()
            except Exception:
                pass
    
    return [float(xi) for xi in best_roots]
