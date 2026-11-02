import numpy as np
import time
import scipy.integrate as integrate
import scipy.optimize as opt

# Precompute constants
H0_0 = 1.0
H4_0 = 3.0
H8_0 = 105.0
H12_0 = 10395.0

def he0(x):
    return np.ones_like(x)

def he4(x):
    return x**4 - 6*x**2 + 3

def he8(x):
    return x**8 - 28*x**6 + 420*x**4 - 840*x**2 + 105

def he12(x):
    return x**12 - 66*x**10 + 2310*x**8 - 39900*x**6 + 332640*x**4 - 798336*x**2 + 10395

def c3_from(c0, c1, c2):
    return -(c0 + H4_0*c1 + H8_0*c2) / H12_0

def compute_l1(c0, c1, c2):
    """
    Compute L1 norm of g(x) = P(x) exp(-x^2/2)
    g(x) is even, so integrate from 0 to inf and multiply by 2.
    """
    c3 = c3_from(c0, c1, c2)
    
    def integrand(x):
        # P(x) * exp(-x^2/2)
        p_val = (c0 * he0(np.array([x]))[0] + 
                 c1 * he4(np.array([x]))[0] + 
                 c2 * he8(np.array([x]))[0] + 
                 c3 * he12(np.array([x]))[0])
        return abs(p_val * np.exp(-x**2 / 2))
    
    # Integrate from 0 to a sufficiently large limit (e.g., 10)
    # exp(-x^2/2) decays very fast
    val, err = integrate.quad(integrand, 0, 15, limit=100, epsabs=1e-10, epsrel=1e-10)
    return 2 * val

def compute_min_abs_hat(c0, c1, c2):
    """
    Compute min_{xi} |hat{g}(xi)|.
    hat{g}(xi) = sqrt(2*pi) * P(xi) * exp(-xi^2/2)
    Since P(xi) is a sum of even Hermite polynomials, P(xi) is even.
    exp(-xi^2/2) is even.
    So |hat{g}(xi)| is even. We search xi in [0, inf).
    """
    c3 = c3_from(c0, c1, c2)
    sqrt_2pi = np.sqrt(2 * np.pi)
    
    def hat_g_abs(xi):
        # P(xi)
        p_val = (c0 * he0(np.array([xi]))[0] + 
                 c1 * he4(np.array([xi]))[0] + 
                 c2 * he8(np.array([xi]))[0] + 
                 c3 * he12(np.array([xi]))[0])
        return abs(sqrt_2pi * p_val * np.exp(-xi**2 / 2))
    
    # Strategy:
    # 1. Find local minima using a coarse grid.
    # 2. Refine each local minimum using minimize_scalar.
    
    # The function decays to 0 as xi -> inf.
    # We are looking for the global minimum of the magnitude.
    # Note: If the function touches 0, the score is 0 (or infinity depending on definition, 
    # but here we want to MINIMIZE the ratio L1/min_abs. If min_abs -> 0, ratio -> inf. 
    # Wait, the task says "reduce the upper bound". 
    # Upper bound on C4 is ||g||_1 / min|hat{g}|.
    # To minimize this bound, we want to MAXIMIZE min|hat{g}| relative to ||g||_1.
    # So we want to AVOID zeros in hat{g} if possible, or make the minimum as large as possible 
    # relative to the L1 norm.
    
    # Let's find the global minimum of hat_g_abs on [0, 15].
    # If it's 0, the bound is infinite (bad).
    # We want to find coefficients such that this minimum is as large as possible 
    # compared to the L1 norm.
    
    # Grid search for local minima candidates
    xi_grid = np.linspace(0, 15, 1501)
    hat_vals = np.array([hat_g_abs(xi) for xi in xi_grid])
    
    # Find indices where hat_vals is a local minimum
    local_min_indices = []
    for i in range(1, len(hat_vals) - 1):
        if hat_vals[i] < hat_vals[i-1] and hat_vals[i] <= hat_vals[i+1]:
            local_min_indices.append(i)
            
    # Also check boundaries (0 and 15)
    local_min_indices.append(0)
    local_min_indices.append(len(hat_vals)-1)
    
    min_val = np.inf
    for idx in local_min_indices:
        xi_start = xi_grid[idx]
        # Refine
        try:
            res = opt.minimize_scalar(hat_g_abs, bounds=(max(0, xi_start-0.5), min(15, xi_start+0.5)), 
                                      method='bounded', options={'xatol': 1e-10})
            if res.fun < min_val:
                min_val = res.fun
        except Exception:
            pass
            
    # If the minimum is extremely small, it might be a zero.
    # We return the minimum value.
    return min_val

def objective(coeffs):
    c0, c1, c2 = coeffs
    l1 = compute_l1(c0, c1, c2)
    min_hat = compute_min_abs_hat(c0, c1, c2)
    
    if min_hat < 1e-12:
        return 1e10 # Penalize zeros heavily
    return l1 / min_hat

def construct():
    t0 = time.time()
    deadline = t0 + 190  # Safety margin for 210s limit
    
    # Seed coefficients from current program
    seed = np.array([-113/100, 1/25, 1/3240])
    best = seed.copy()
    best_val = objective(best)
    
    # Local refinement from seed
    try:
        res = opt.minimize(objective, seed, method='Nelder-Mead', 
                           options={'maxiter': 1000, 'xatol':1e-6, 'fatol':1e-8, 'adaptive': True})
        if time.time() < deadline and res.fun < best_val:
            best = res.x.copy()
            best_val = res.fun
    except Exception:
        pass
    
    # Basin Hopping to escape local minima
    if time.time() < deadline - 20:
        try:
            # Define step size for basin hopping
            # Coefficients are roughly in range [-1, 1]
            step = 0.5
            
            def step_func(x):
                # Random perturbation
                return x + np.random.normal(0, step, size=3)
                
            # Run basin hopping
            bh_res = opt.basinhopping(objective, best, stepfunc=step_func, 
                                      minimizer_kwargs={'method': 'Nelder-Mead', 'options': {'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}},
                                      niter=50, T=1.0, stepsize=0.5, 
                                      accept_test=opt.fminbound, # Dummy, we use standard
                                      )
            # Actually, basinhopping returns a result object
            if bh_res.fun < best_val:
                best = bh_res.x.copy()
                best_val = bh_res.fun
        except Exception:
            pass

    # Multi-start with diverse initial points if time permits
    if time.time() < deadline - 30:
        rng = np.random.default_rng(42)
        for _ in range(5):
            if time.time() > deadline - 15:
                break
            # Try points in a wider range
            start = rng.uniform(-2, 2, size=3)
            try:
                res = opt.minimize(objective, start, method='Nelder-Mead',
                                   options={'maxiter': 500, 'xatol':1e-5, 'fatol':1e-7, 'adaptive': True})
                if time.time() > deadline - 15:
                    break
                if res.fun < best_val:
                    best = res.x.copy()
                    best_val = res.fun
            except Exception:
                pass

    # Final local polish
    if time.time() < deadline - 5:
        try:
            res = opt.minimize(objective, best, method='Nelder-Mead',
                               options={'maxiter': 1000, 'xatol':1e-8, 'fatol':1e-10, 'adaptive': True})
            if res.fun < best_val:
                best = res.x.copy()
                best_val = res.fun
        except Exception:
            pass
    
    return np.array(best)
