"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution

_start_time = time.time()
DEADLINE = 85.0  # Safety margin before 90s kill

def _deadline_ok():
    return (time.time() - _start_time) < DEADLINE

def _E_of_a(a, Om0, Ok0, Ol0, w0, w1):
    """H(a)/H0 for CPL model. Vectorized for speed."""
    a = np.asarray(a, dtype=np.float64)
    a3 = a ** 3
    a2 = a ** 2
    # w(z) = w0 + w1*z, z = 1/a - 1
    # Handle a=0 safely, though nodes are in (0,1)
    z = 1.0 / a - 1.0
    w = w0 + w1 * z
    de = Ol0 * np.power(a, -3.0 * (1.0 + w))
    return np.sqrt(Om0 / a3 + Ok0 / a2 + de)

def _integrand_vec(t, z, Om0, Ok0, Ol0, w0, w1):
    """Vectorized integrand 1/E(z*t) where a = 1/(1+z*t)"""
    a = 1.0 / (1.0 + z * t)
    return 1.0 / _E_of_a(a, Om0, Ok0, Ol0, w0, w1)

def _exact_integral(z, Om0, Ok0, Ol0, w0, w1):
    """
    High-precision reference integral.
    Since we cannot use scipy.integrate.quad inside the tight loop of the optimizer
    efficiently enough for a dense grid, we use a fixed high-order Gauss-Legendre
    rule (64 points) as a proxy for the 'exact' value. The error of this proxy
    is ~1e-15, which is negligible compared to the quadrature error we are minimizing.
    """
    # 64-point Gauss-Legendre nodes and weights on [0,1]
    # These are pre-computed constants for speed
    # Using standard GL nodes scaled to [0,1]
    x_gl, w_gl = np.polynomial.legendre.leggauss(64)
    t = 0.5 * (x_gl + 1.0)
    w = 0.5 * w_gl
    
    vals = _integrand_vec(t, z, Om0, Ok0, Ol0, w0, w1)
    return np.dot(w, vals)

def _quadrature_value(x, w, z, Om0, Ok0, Ol0, w0, w1):
    """Compute quadrature sum for given nodes x and weights w."""
    a = 1.0 / (1.0 + z * x)
    E_vals = _E_of_a(a, Om0, Ok0, Ol0, w0, w1)
    return np.dot(w, 1.0 / E_vals)

# Define a comprehensive grid of cosmologies to test against
# This ensures the rule is robust across the parameter space, not just a few points.
# z in [0.1, 5.0]
# Om0 in [0.1, 0.5]
# Ok0 in [-0.1, 0.1] (curvature)
# Ol0 = 1 - Om0 - Ok0
# w0 in [-1.5, 0.5]
# w1 in [-1.0, 1.0]

# To keep the optimization time manageable, we subsample this dense grid.
# We generate a set of representative points.
def _generate_test_set():
    zs = np.array([0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0])
    Oms = np.array([0.2, 0.3, 0.4])
    Oks = np.array([-0.05, 0.0, 0.05])
    w0s = np.array([-1.2, -1.0, -0.5, 0.0])
    w1s = np.array([-0.5, 0.0, 0.5])
    
    test_cases = []
    # We don't want to test ALL combinations (8*3*3*4*3 = 864), that might be too slow per eval.
    # Let's aim for ~100-200 diverse cases.
    # Random sampling from the hypercube with some stratification
    rng = np.random.default_rng(42)
    for _ in range(150):
        z = rng.uniform(0.1, 5.0)
        Om0 = rng.uniform(0.1, 0.5)
        Ok0 = rng.uniform(-0.1, 0.1)
        Ol0 = 1.0 - Om0 - Ok0
        if Ol0 < 0.0: continue
        w0 = rng.uniform(-1.5, 0.5)
        w1 = rng.uniform(-1.0, 1.0)
        test_cases.append((z, Om0, Ok0, Ol0, w0, w1))
        
    # Add specific extreme cases
    test_cases.extend([
        (0.1, 0.3, 0.0, 0.7, -1.0, 0.0),
        (5.0, 0.3, 0.0, 0.7, -1.0, 0.0),
        (1.0, 0.1, -0.05, 0.95, -1.5, -1.0),
        (2.0, 0.5, 0.1, 0.4, 0.5, 1.0),
        (0.5, 0.3, 0.0, 0.7, -1.0, 0.0),
        (3.0, 0.3, 0.0, 0.7, -0.5, 0.5),
    ])
    return np.array(test_cases)

_test_set = _generate_test_set()

def _objective(params):
    """
    Objective function for Differential Evolution.
    params: [x1..x8, logw1..logw8]
    Returns max relative error.
    """
    if not _deadline_ok():
        return 1e10

    x = params[:8]
    logw = params[8:]
    
    # Constraints: nodes in (0,1), strictly increasing
    if np.any(x <= 0) or np.any(x >= 1) or np.any(np.diff(x) <= 0):
        return 1e10

    w = np.exp(logw)
    w_sum = np.sum(w)
    if w_sum <= 0:
        return 1e10
    w = w / w_sum

    max_err = 0.0
    # Evaluate on the test set
    for z, Om0, Ok0, Ol0, w0, w1 in _test_set:
        exact = _exact_integral(z, Om0, Ok0, Ol0, w0, w1)
        approx = _quadrature_value(x, w, z, Om0, Ok0, Ol0, w0, w1)
        err = abs(approx - exact) / abs(exact)
        if err > max_err:
            max_err = err
            if max_err > 0.1: # Early exit if error is already huge
                return max_err
                
    return max_err

def construct():
    n = 8
    bounds = []
    # Nodes: strictly increasing in (0,1). 
    # DE handles bounds per parameter. We use loose bounds and rely on penalty, 
    # or we can transform. Loose bounds + penalty is simpler for DE.
    for i in range(n):
        bounds.append((1e-4, 1 - 1e-4))
    # Log weights: wide range to allow optimization
    for i in range(n):
        bounds.append((-10, 10))

    # Initial population size and max iterations
    # DE is stochastic, so we need to be careful with time.
    # popsize=20, maxiter=50 is a reasonable start.
    # 20 * 50 * 150 test cases * 8 evals per case = 1.2M integrand evals.
    # Each integrand eval is vectorized over 8 points.
    
    # To speed up, we can reduce the test set size if time is tight, 
    # but let's try a moderate one first.
    
    # We will run DE. If it times out, we return the best found so far.
    # scipy's differential_evolution doesn't support mid-run interruption easily 
    # without a custom callback that raises an exception, which is messy.
    # Instead, we run it with a conservative maxiter and check time before/after.
    # Or we can use a smaller population and fewer iterations.
    
    # Let's try popsize=15, maxiter=30.
    # 15 * 30 = 450 function evaluations.
    # Each function eval loops over ~150 test cases.
    # 450 * 150 = 67,500 integrals.
    # Each integral uses 64-point GL for exact + 8-point quad for approx.
    # This should be fast enough in numpy.
    
    try:
        result = differential_evolution(
            _objective, 
            bounds, 
            maxiter=30, 
            popsize=15, 
            tol=1e-6, 
            seed=42,
            polish=True, # Use L-BFGS-B to polish the result
            # maxfev=5000, # Limit total function evaluations
            init='sobol' # Sobol sequence for better coverage
        )
        
        best_params = result.x
        best_obj = result.fun
        
    except Exception as e:
        # If DE fails, fall back to a simple heuristic or random search
        print(f"DE failed: {e}")
        # Fallback: Gauss-Legendre nodes and weights
        x_gl, w_gl = np.polynomial.legendre.leggauss(n)
        x = 0.5 * (x_gl + 1.0)
        w = 0.5 * w_gl
        w = w / np.sum(w)
        return {"nodes": x, "weights": w}

    # Extract and normalize
    x = best_params[:8]
    logw = best_params[8:]
    w = np.exp(logw)
    w = w / np.sum(w)
    
    # Ensure nodes are strictly in (0,1) and increasing
    # DE might produce slightly out of bounds or non-increasing nodes due to floating point
    x = np.sort(x)
    x = np.clip(x, 1e-6, 1 - 1e-6)
    # Ensure strict increase
    for i in range(1, n):
        if x[i] <= x[i-1]:
            x[i] = x[i-1] + 1e-6
    
    # Final normalization
    w = w / np.sum(w)
    
    return {"nodes": x, "weights": w}
# EVOLVE-BLOCK-END
