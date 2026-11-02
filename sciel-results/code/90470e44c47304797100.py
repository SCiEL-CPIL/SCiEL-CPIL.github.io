"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution, basinhopping
import time

def construct():
    """
    Find optimal coefficients (c0, c4, c8) for the Hermite-Gaussian test function
    g(x) = (c0*H0 + c4*H4 + c8*H8 + c12*H12) * exp(-x^2/2)
    where c12 is determined by g(0)=0.
    
    The objective is to minimize the upper bound on A(f)*A(Fourier(f)), which for
    this Fourier-invariant combination is (sup |x*g(x)|)^2.
    
    We use a smooth surrogate (log-sum-exp) for the max to enable gradient-based
    optimization, combined with global search and local refinement.
    """
    start_time = time.time()
    deadline = 195.0  # Safety margin before 210s limit

    # Pre-compute constants
    H0_0 = 1.0
    H4_0 = 12.0
    H8_0 = 1680.0
    H12_0 = 665280.0

    # Grid for fast evaluation
    N = 32768
    x = np.linspace(-8, 8, N)
    
    # Pre-compute Hermite polynomials on the grid
    x2 = x * x
    H0 = np.ones_like(x)
    H4 = 16.0*x2*x2 - 48.0*x2 + 12.0
    x4 = x2 * x2
    x6 = x4 * x2
    x8 = x4 * x4
    H8 = 256.0*x8 - 4480.0*x6 + 26880.0*x4 - 6720.0*x2 + 1680.0
    x10 = x8 * x2
    x12 = x6 * x6
    H12 = (4096.0*x12 - 129024.0*x10 + 1761280.0*x8 - 11942400.0*x6 
           + 37158400.0*x4 - 54432000.0*x2 + 47900160.0)
    
    # Pre-compute the basis functions for g(x)
    basis0 = (H0 - (H0_0/H12_0)*H12) * np.exp(-x**2 / 2.0)
    basis4 = (H4 - (H4_0/H12_0)*H12) * np.exp(-x**2 / 2.0)
    basis8 = (H8 - (H8_0/H12_0)*H12) * np.exp(-x**2 / 2.0)
    
    xb0 = x * basis0
    xb4 = x * basis4
    xb8 = x * basis8

    # Pre-compute absolute values for log-sum-exp
    abs_xb0 = np.abs(xb0)
    abs_xb4 = np.abs(xb4)
    abs_xb8 = np.abs(xb8)

    def objective(coeffs):
        """
        Compute the squared supremum of |x * g(x)| using a smooth approximation.
        coeffs: [c0, c4, c8]
        """
        c0, c4, c8 = coeffs
        gx = c0*xb0 + c4*xb4 + c8*xb8
        max_val = np.max(np.abs(gx))
        return max_val * max_val

    def smooth_objective(coeffs, beta=50.0):
        """
        Smooth approximation of (sup|gx|)^2 using log-sum-exp.
        This is differentiable and helps gradient-based methods.
        """
        c0, c4, c8 = coeffs
        gx = c0*xb0 + c4*xb4 + c8*xb8
        abs_gx = np.abs(gx)
        
        # Log-sum-exp for stability
        m = np.max(abs_gx)
        if m < 1e-15:
            return 0.0
        log_sum = m + np.log(np.sum(np.exp((abs_gx - m) * beta)) / len(abs_gx))
        # The soft-max approximates the max; scale by beta to get closer to true max
        soft_max = log_sum  # This is actually log(mean(exp(beta*|gx|)))
        # Better: use the actual soft-max formula
        # soft_max = (1/beta) * log(sum(exp(beta * |gx|)))
        # For large beta, this approaches max(|gx|)
        
        # Recompute properly
        beta_safe = min(beta, 100.0)
        shifted = abs_gx - m
        log_terms = shifted * beta_safe
        # Clip to avoid overflow
        log_terms = np.clip(log_terms, -700, 700)
        log_sum_exp = m + (1.0/beta_safe) * np.log(np.sum(np.exp(log_terms)))
        
        return log_sum_exp * log_sum_exp

    # Initial guess - the known good starting point
    x0 = np.array([-1.13, 0.04, 0.000308642])
    best_val = objective(x0)
    best_x = x0.copy()

    # Try multiple random starts with smooth objective
    rng = np.random.RandomState(42)
    
    # Generate diverse initial points
    init_points = [x0.copy()]
    for _ in range(20):
        pt = np.array([
            rng.uniform(-3.0, 3.0),
            rng.uniform(-0.5, 0.5),
            rng.uniform(-0.005, 0.005)
        ])
        init_points.append(pt)
    
    # Evaluate all initial points
    for pt in init_points:
        val = objective(pt)
        if val < best_val:
            best_val = val
            best_x = pt.copy()
    
    # Use smooth objective for L-BFGS-B optimization from best points
    smooth_obj = lambda c: smooth_objective(c, beta=50.0)
    
    # Optimize from top few initial points
    scored_points = [(objective(pt), pt) for pt in init_points]
    scored_points.sort(key=lambda t: t[0])
    
    for i, (val, pt) in enumerate(scored_points[:5]):
        if time.time() > deadline - 60:
            break
        try:
            res = minimize(smooth_obj, pt, method='L-BFGS-B', 
                          options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10})
            if res.fun < best_val:
                # Verify with exact objective
                exact_val = objective(res.x)
                if exact_val < best_val:
                    best_val = exact_val
                    best_x = res.x.copy()
        except Exception:
            pass
    
    # Differential Evolution with exact objective
    bounds = [(-5.0, 5.0), (-1.0, 1.0), (-0.01, 0.01)]
    try:
        if time.time() < deadline - 40:
            de_result = differential_evolution(
                objective, bounds, 
                maxiter=80, 
                popsize=15, 
                seed=42,
                tol=1e-10,
                mutation=(0.3, 1.0),
                recombination=0.7
            )
            if de_result.fun < best_val:
                best_val = de_result.fun
                best_x = de_result.x.copy()
    except Exception:
        pass

    # Basin Hopping for local refinement
    try:
        if time.time() < deadline - 20:
            def local_minimizer(x):
                res = minimize(objective, x, method='Nelder-Mead', 
                              options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12})
                return res.x, res.fun, {}
            
            bh_result = basinhopping(
                objective, best_x, 
                minimizer_kwargs={'method': 'Nelder-Mead', 'options': {'maxiter': 200}},
                niter=40, 
                T=0.5, 
                stepsize=0.05,
                seed=42
            )
            if bh_result.fun < best_val:
                best_val = bh_result.fun
                best_x = bh_result.x.copy()
    except Exception:
        pass

    # Final high-precision polish
    try:
        if time.time() < deadline - 5:
            res = minimize(objective, best_x, method='Powell', 
                          options={'maxiter': 1000, 'ftol': 1e-14, 'xtol': 1e-14})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
                
            res_nm = minimize(objective, best_x, method='Nelder-Mead', 
                             options={'maxiter': 2000, 'xatol': 1e-14, 'fatol': 1e-14})
            if res_nm.fun < best_val:
                best_val = res_nm.fun
                best_x = res_nm.x.copy()
    except Exception:
        pass

    return best_x

# EVOLVE-BLOCK-END
