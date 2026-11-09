import numpy as np
import numpy.polynomial.laguerre as npl
import time
from scipy.special import factorial

def construct():
    start_time = time.time()
    deadline = 600.0
    
    # m=12, n=1
    m = 12
    n = 1
    
    # The Cohn-Goncalves construction involves a polynomial P(x) = prod (x - r_i)^2 
    # where r_i are the roots.
    # The uncertainty bound C4 is related to the moments of |P(x)|^2 e^{-x} dx.
    # Specifically, for the Fourier sign uncertainty, the bound involves:
    # C = ( \int_0^infty x^2 |P(x)|^2 e^{-x} dx \cdot \int_0^infty |P(x)|^2 e^{-x} dx ) / 
    #     ( \int_0^infty x |P(x)|^2 e^{-x} dx )^2 - 1 ? 
    # Actually, the standard uncertainty principle for Fourier transform on L2(R) with 
    # weight e^{-x^2} or similar. Here we are in Laguerre space (half-line).
    
    # Let's look at the "refined uncertainty inequality" context.
    # The bound on C4 is typically:
    # C4 <= (M2 * M0) / M1^2 - 1
    # where Mk = integral x^k |P(x)|^2 e^{-x} dx.
    # Wait, the variance is M2/M0 - (M1/M0)^2 = (M2*M0 - M1^2)/M0^2.
    # The "upper bound on C4" might be exactly this variance or a scaled version.
    
    # Let's assume the score is the variance of the distribution |P(x)|^2 e^{-x} / M0.
    # Minimizing this variance is the goal.
    
    # Initial guess: Asymptotic roots of Laguerre L_m.
    # Roots of L_m(x) are approx m + 2*sqrt(m*x) + ... ?
    # Actually, for large m, roots of L_m(x) are distributed in [0, 4m].
    # The density is ~ 1/(pi sqrt(x(4m-x))).
    # For m=12, 4m=48.
    
    # Let's use the exact roots of L_12 as a strong starting point, 
    # but perturb them to escape local minima if the objective is non-convex.
    # However, the current program (score 0.4869) uses L-BFGS on log-roots 
    # starting from Laguerre roots. It likely got stuck or the proxy was slightly off.
    
    # Let's compute the exact objective function using Gauss-Laguerre quadrature.
    # P(x) = prod_{i=1}^{12} (x - r_i)^2
    # |P(x)|^2 = prod (x - r_i)^4
    
    # Quadrature nodes: Need enough to integrate degree 48 polynomial exactly.
    # Gauss-Laguerre with N nodes integrates exactly up to degree 2N-1.
    # 2N-1 >= 48 => N >= 24.5 => N=25 is enough.
    # Let's use N=30 for safety.
    
    try:
        nodes, weights = npl.laggauss(30)
    except Exception:
        # Fallback if npl fails
        nodes = np.linspace(0.1, 50, 100)
        weights = np.ones(100) * 0.5 # Rough approximation, not exact
    
    def compute_bound(roots):
        # roots: array of 12 positive numbers
        # Compute P(x)^2 = prod (x - r_i)^4 at nodes
        # diff shape (30, 12)
        diff = nodes[:, np.newaxis] - roots[np.newaxis, :]
        p_sq = np.prod(diff**4, axis=1)
        
        # Moments
        m0 = np.dot(weights, p_sq)
        m1 = np.dot(weights, nodes * p_sq)
        m2 = np.dot(weights, nodes**2 * p_sq)
        
        if m0 <= 1e-100:
            return 1e100
            
        mean = m1 / m0
        var = m2 / m0 - mean**2
        
        # The "upper bound on C4" is likely this variance.
        # Or is it related to the Heisenberg uncertainty principle?
        # For Fourier transform, (Delta x)(Delta k) >= 1/2.
        # In Laguerre basis, the operators are different.
        # The prompt says "score is the upper bound on C4".
        # And "lower is better".
        # Variance is a measure of spread. Smaller spread = lower score.
        # Let's return the variance.
        
        return var

    # Get initial roots from Laguerre L_12
    try:
        roots_init, _ = npl.laggauss(12)
        roots_init = np.sort(roots_init).astype(float)
    except Exception:
        roots_init = np.linspace(1, 40, 12)
        
    # Simulated Annealing
    # Objective: minimize variance
    # Parameters: roots (12 values)
    # Constraint: roots > 0, distinct
    
    best_roots = roots_init.copy()
    best_score = compute_bound(best_roots)
    current_roots = roots_init.copy()
    current_score = best_score
    
    T = 1.0 # Initial temperature
    cooling_rate = 0.995
    steps = 20000
    
    # Use log-space for positivity
    log_current = np.log(current_roots)
    log_best = log_current.copy()
    
    for i in range(steps):
        if time.time() > deadline - 30:
            break
            
        # Propose new roots
        # Gaussian noise in log-space
        noise = np.random.normal(0, 0.1, size=12)
        log_proposed = log_current + noise
        proposed = np.exp(log_proposed)
        
        # Check validity (positive, distinct)
        # Distinctness: if any two roots are too close, the polynomial has a high peak, 
        # variance might spike or behave weirdly. 
        # Let's just compute the score.
        
        prop_score = compute_bound(proposed)
        
        # Acceptance probability
        delta = prop_score - current_score
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            current_roots = proposed
            log_current = log_proposed
            current_score = prop_score
            
            if current_score < best_score:
                best_score = current_score
                best_roots = current_roots.copy()
                log_best = log_current.copy()
                
        T *= cooling_rate
        
        # Optional: Local refinement every 1000 steps?
        # The directive says "MATERIALLY DIFFERENT". SA is different from L-BFGS.
        # But SA can be slow.
        # Let's add a simple local search step using Nelder-Mead on the best found so far
        # every 5000 steps if time permits.
        
        if i % 5000 == 4999 and time.time() < deadline - 60:
            from scipy.optimize import minimize
            try:
                # Optimize from current best
                res = minimize(
                    lambda x: compute_bound(np.exp(x)),
                    log_best,
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8}
                )
                if res.fun < best_score:
                    best_score = res.fun
                    log_best = res.x
                    best_roots = np.exp(res.x)
            except Exception:
                pass
                
    # Final local polish
    if time.time() < deadline - 10:
        from scipy.optimize import minimize
        try:
            res = minimize(
                lambda x: compute_bound(np.exp(x)),
                log_best,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-12}
            )
            if res.fun < best_score:
                best_score = res.fun
                best_roots = np.exp(res.x)
        except Exception:
            pass

    # Ensure sorted and positive
    best_roots = np.sort(best_roots)
    if np.any(best_roots <= 0):
        best_roots = roots_init
        
    return best_roots.tolist()
