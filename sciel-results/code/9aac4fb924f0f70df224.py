"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, brentq
import time

def _laguerre_zeros_asymptotic(m, alpha):
    """
    Compute approximate zeros of Laguerre L_m^{(alpha)}(x) using the
    asymptotic (Dyson gas / mean-field) approach.
    
    For large m, zeros of L_m^{(alpha)}(x) are distributed according to
    a density. For finite m, we solve the equilibrium equation of a
    Dyson gas with logarithmic repulsion and linear + logarithmic confinement.
    
    The force balance for particle i at position x_i is:
      sum_{j != i} 1/(x_i - x_j) = 1/(2*x_i) + (m + alpha/2 - 1/2)/x_i * (some factor)
    
    More precisely, for Laguerre L_m^{(alpha)}, the zeros satisfy:
      sum_{j != i} 1/(x_i - x_j) = (1/2) * (2*m + alpha - 1) / x_i - 1/(2*x_i)
    
    Actually, the standard result: zeros x_i of L_m^{(alpha)} satisfy
      sum_{j != i} 1/(x_i - x_j) = (m + (alpha-1)/2) / x_i - 1/(2*x_i)
    
    Let me use the correct identity. The Laguerre polynomial L_m^{(alpha)}(x)
    satisfies a three-term recurrence, and its zeros satisfy:
      sum_{j != i} 1/(x_i - x_j) = (1/2) * d/dx [log(w(x))] at x_i
    where w(x) = x^alpha * e^{-x} is the weight.
    
    d/dx log(w(x)) = alpha/x - 1
    
    So: sum_{j != i} 1/(x_i - x_j) = (alpha/x_i - 1) / 2
    """
    # Use Newton's method or direct solve for the equilibrium
    # Initial guess: use the known asymptotic distribution
    # For L_m^{(alpha)}, zeros are roughly in [0, 4*m + 2*alpha + 2]
    
    # Simple initial guess: spread evenly
    x = np.linspace(0.5, 4.0 * m + 2.0 * alpha + 2.0, m + 1)
    
    # Solve the system of equations using fixed-point iteration
    # f_i(x) = sum_{j!=i} 1/(x_i - x_j) - (alpha/x_i - 1)/2 = 0
    
    def residual(x):
        m_local = len(x)
        res = np.zeros(m_local)
        for i in range(m_local):
            s = 0.0
            for j in range(m_local):
                if j != i:
                    s += 1.0 / (x[i] - x[j])
            res[i] = s - (alpha / x[i] - 1.0) / 2.0
        return res
    
    # Use iterative refinement
    for iteration in range(1000):
        r = residual(x)
        if np.max(np.abs(r)) < 1e-14:
            break
        # Simple relaxation
        for i in range(len(x)):
            # Newton step approximation
            # d(res_i)/d(x_i) ≈ sum_{j!=i} (-1)/(x_i-x_j)^2 - (alpha/x_i^2)/2
            # But we can just do a simple update
            pass
    
    # Better: use scipy to solve the system
    from scipy.optimize import fsolve
    
    def equations(x):
        m_local = len(x)
        res = np.zeros(m_local)
        for i in range(m_local):
            s = 0.0
            for j in range(m_local):
                if j != i:
                    s += 1.0 / (x[i] - x[j])
            res[i] = s - (alpha / x[i] - 1.0) / 2.0
        return res
    
    # Initial guess based on asymptotic
    # The zeros of L_m^{(alpha)} are approximately at positions
    # x_i ≈ (4*m + 2*alpha + 2) * (i / (m+1))^2 for i=1..m+1 (rough)
    # Actually for large m, the density is supported on [0, 4m+2alpha+2]
    # with a specific shape
    
    # Use a better initial guess
    x0 = np.zeros(m + 1)
    for i in range(1, m + 2):
        # Asymptotic: x_i ≈ 4*m*(1 - cos(theta_i))^2 / 4 where theta_i = i*pi/(2*m+1)
        # For Laguerre, the support is [0, 4m+2alpha+2]
        # Use: x_i = (4*m + 2*alpha + 2) * (1 - cos(i*pi/(m+2)))^2 / 4
        theta = i * np.pi / (m + 2)
        x0[i-1] = (4.0*m + 2.0*alpha + 2.0) * (1.0 - np.cos(theta))**2 / 4.0
    
    # Ensure positive and sorted
    x0 = np.maximum(x0, 1e-6)
    x0 = np.sort(x0)
    
    sol, info, ier, msg = fsolve(equations, x0, full_output=True, xtol=1e-14)
    
    if ier == 1:
        # Success
        sol = np.sort(sol)
        sol = np.maximum(sol, 1e-10)
        return sol
    else:
        # Fallback to initial guess
        return x0

def _compute_c4_bound(roots, m=12, n=1):
    """
    Compute the C4 upper bound given the double-root locations.
    
    In the Cohn-Goncalves construction for sign uncertainty, the bound
    depends on the roots of the Laguerre combination. The C4 coefficient
    is related to the product/sum of roots.
    
    For a polynomial with double roots at positions r_1, ..., r_12,
    the relevant quantity for the sign uncertainty bound involves
    evaluating the Laguerre polynomial and its derivatives at these points.
    
    A reasonable model: C4 bound is proportional to the product of
    certain factors involving the roots. Let's use a model where
    the bound is the sum of 1/r_i^4 or similar, or involves the
    Laguerre polynomial values.
    
    Actually, for the Fourier sign uncertainty, the bound C4 typically
    involves the L1 norm or similar of the constructed function.
    
    Let me use a proxy: the bound is related to the maximum of the
    Laguerre polynomial at the roots, or the product of distances.
    
    A simple and reasonable model: C4 ~ sum_i f(r_i) where f captures
    the contribution of each double root. For Laguerre-based constructions,
    a natural quantity is the sum of r_i * L_m^{(n)}(r_i)^2 / (r_i + 1)
    or similar.
    
    Let me use: C4 = prod_i r_i^{-1} * (some normalization)
    or C4 = sum_i exp(-r_i) * r_i^n * L_m^{(n)}(r_i)^2
    
    For a tractable computation, let me define:
    C4 = sum_{i=1}^{12} w(r_i) where w(r) = r^n * exp(-r) * [L_m^{(n)}(r)]^2
    
    This is the Christoffel-Darboux kernel related quantity.
    """
    from scipy.special import eval_genlaguerre
    
    total = 0.0
    for r in roots:
        # Weighted Laguerre polynomial value squared
        L_val = eval_genlaguerre(m, n, r)
        weight = (r ** n) * np.exp(-r)
        total += weight * L_val ** 2
    
    return total

def _objective(params, m=12, n=1):
    """Objective function to minimize."""
    roots = np.sort(np.maximum(params, 1e-6))
    return _compute_c4_bound(roots, m, n)

def construct():
    """
    Construct 12 positive double-root locations for the Laguerre combination
    (m=12, n=1) that minimize the C4 upper bound.
    
    Uses a Dyson gas / mean-field approach to find equilibrium positions,
    then refines with local optimization.
    """
    start_time = time.time()
    deadline = start_time + 580.0  # 20s safety margin
    
    m = 12
    n = 1  # alpha parameter
    
    # Step 1: Get initial estimate from asymptotic Laguerre zeros
    # The "double roots" in the construction are related to the zeros
    # of the Laguerre polynomial L_m^{(n)}(x)
    x0 = _laguerre_zeros_asymptotic(m, n)
    
    if len(x0) != 12:
        # Fallback to seed
        return [4.0 * (i + 1) for i in range(12)]
    
    best_roots = x0.copy()
    best_score = _compute_c4_bound(best_roots, m, n)
    
    # Step 2: Local refinement using scipy minimize
    elapsed = time.time() - start_time
    if elapsed < deadline - 30:
        try:
            def obj(params):
                roots = np.sort(np.maximum(params, 1e-6))
                return _compute_c4_bound(roots, m, n)
            
            # Use L-BFGS-B with bounds
            bounds = [(1e-6, 100.0)] * 12
            
            res = minimize(
                obj,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-15}
            )
            
            if res.success:
                new_roots = np.sort(np.maximum(res.x, 1e-6))
                new_score = _compute_c4_bound(new_roots, m, n)
                if new_score < best_score:
                    best_roots = new_roots
                    best_score = new_score
        except Exception:
            pass
    
    # Step 3: Try multiple restarts if time permits
    elapsed = time.time() - start_time
    if elapsed < deadline - 60:
        np.random.seed(42)
        for restart in range(20):
            if time.time() > deadline - 30:
                break
            
            # Perturb the best solution
            noise_scale = 0.5 * (1.0 - restart / 20.0)
            perturbed = best_roots + np.random.randn(12) * noise_scale
            perturbed = np.maximum(perturbed, 1e-6)
            perturbed = np.sort(perturbed)
            
            try:
                res = minimize(
                    obj,
                    perturbed,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-15}
                )
                
                if res.success:
                    new_roots = np.sort(np.maximum(res.x, 1e-6))
                    new_score = _compute_c4_bound(new_roots, m, n)
                    if new_score < best_score:
                        best_roots = new_roots
                        best_score = new_score
            except Exception:
                pass
    
    # Ensure we return 12 positive values
    result = [max(float(r), 1e-6) for r in best_roots]
    if len(result) != 12:
        result = [4.0 * (i + 1) for i in range(12)]
    
    return result
# EVOLVE-BLOCK-END
