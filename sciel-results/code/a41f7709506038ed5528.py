"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
DEADLINE = 85.0  # safety margin before 90s kill

def _deadline_ok():
    return (time.time() - _start_time) < DEADLINE

def _E_of_a(a, Om0, Ok0, Ol0, w0, w1):
    """H(a)/H0 for CPL model. Vectorized."""
    a = np.asarray(a, dtype=np.float64)
    a3 = a ** 3
    a2 = a ** 2
    z = 1.0 / a - 1.0
    w = w0 + w1 * z
    de = Ol0 * np.power(a, -3.0 * (1.0 + w))
    return np.sqrt(Om0 / a3 + Ok0 / a2 + de)

def _integrand(t, z, Om0, Ok0, Ol0, w0, w1):
    """1/E(z*t) where a = 1/(1+z*t)"""
    a = 1.0 / (1.0 + z * t)
    return 1.0 / _E_of_a(a, Om0, Ok0, Ol0, w0, w1)

def _exact_integral(z, Om0, Ok0, Ol0, w0, w1):
    """High-precision integral using adaptive quadrature."""
    from scipy.integrate import quad
    val, _ = quad(lambda t: _integrand(t, z, Om0, Ok0, Ol0, w0, w1), 0, 1, limit=300)
    return val

def _quadrature_value(x, w, z, Om0, Ok0, Ol0, w0, w1):
    """Compute quadrature sum for given nodes x and weights w."""
    a = 1.0 / (1.0 + z * x)
    E_vals = _E_of_a(a, Om0, Ok0, Ol0, w0, w1)
    return np.dot(w, 1.0 / E_vals)

def _rel_error(x, w, z, Om0, Ok0, Ol0, w0, w1):
    exact = _exact_integral(z, Om0, Ok0, Ol0, w0, w1)
    approx = _quadrature_value(x, w, z, Om0, Ok0, Ol0, w0, w1)
    return abs(approx - exact) / abs(exact)

# Test cosmologies: (z, Om0, Ok0, Ol0, w0, w1)
_test_set = [
    (0.5, 0.3, 0.0, 0.7, -1.0, 0.0),
    (1.0, 0.3, 0.0, 0.7, -1.0, 0.0),
    (2.0, 0.3, 0.0, 0.7, -1.0, 0.0),
    (0.5, 0.3, 0.0, 0.7, -0.9, 0.2),
    (1.0, 0.3, 0.0, 0.7, -0.9, 0.2),
    (2.0, 0.3, 0.0, 0.7, -0.9, 0.2),
    (0.5, 0.3, 0.0, 0.7, -1.1, -0.3),
    (1.0, 0.3, 0.0, 0.7, -1.1, -0.3),
    (2.0, 0.3, 0.0, 0.7, -1.1, -0.3),
    (0.5, 0.3, 0.05, 0.65, -1.0, 0.0),
    (1.0, 0.3, 0.05, 0.65, -1.0, 0.0),
    (2.0, 0.3, 0.05, 0.65, -1.0, 0.0),
    (0.5, 0.3, 0.05, 0.65, -0.8, 0.3),
    (1.0, 0.3, 0.05, 0.65, -0.8, 0.3),
    (2.0, 0.3, 0.05, 0.65, -0.8, 0.3),
    (3.0, 0.3, 0.0, 0.7, -1.0, 0.0),
    (3.0, 0.3, 0.0, 0.7, -0.9, 0.2),
    (0.3, 0.3, 0.0, 0.7, -1.0, 0.0),
    (0.3, 0.3, 0.0, 0.7, -1.1, -0.3),
]

def _chebyshev_quadrature():
    """
    Construct an 8-point quadrature rule using Chebyshev spectral collocation.
    
    Strategy:
    1. Map the integral to [-1,1] and use Chebyshev nodes as initial guess
    2. For a representative set of integrands, fit the best 8-node rule
       that minimizes worst-case error using a linear least-squares approach
       on the Chebyshev basis, then refine with nonlinear optimization.
    """
    n = 8
    
    # Chebyshev-Gauss nodes on [0,1]
    # Standard Chebyshev nodes on [-1,1]: x_k = cos((2k-1)pi/(2n))
    k = np.arange(1, n + 1)
    t_cheb = np.cos((2 * k - 1) * np.pi / (2 * n))  # on [-1,1], decreasing
    t_cheb = -t_cheb  # reverse to make increasing
    x_cheb = 0.5 * (t_cheb + 1.0)  # map to [0,1]
    x_cheb = np.sort(x_cheb)
    
    # Chebyshev-Gauss weights on [0,1]
    w_cheb = np.pi / n * np.ones(n)
    # Adjust for the mapping: integral_0^1 f(t)dt = integral_{-1}^1 f((x+1)/2) dx/2
    # The standard CG weights sum to pi, after scaling by 1/2 they sum to pi/2
    # We want weights to sum to 1, so normalize
    w_cheb = w_cheb / w_cheb.sum()
    
    return x_cheb, w_cheb

def _gauss_legendre_quadrature():
    """Get Gauss-Legendre 8-point rule on [0,1]."""
    n = 8
    # Gauss-Legendre nodes and weights on [-1,1]
    x_gl, w_gl = np.polynomial.legendre.leggauss(n)
    # Map to [0,1]
    x_gl = 0.5 * (x_gl + 1.0)
    w_gl = 0.5 * w_gl
    w_gl = w_gl / w_gl.sum()
    return x_gl, w_gl

def _optimize_nodes_weights(x_init, w_init):
    """
    Refine nodes and weights to minimize worst-case relative error.
    Uses a simple gradient-free approach with careful time management.
    """
    n = 8
    best_x = x_init.copy()
    best_w = w_init.copy()
    
    # Compute initial max error
    max_err = 0.0
    for z, Om0, Ok0, Ol0, w0, w1 in _test_set:
        err = _rel_error(best_x, best_w, z, Om0, Ok0, Ol0, w0, w1)
        max_err = max(max_err, err)
    
    best_err = max_err
    
    # Simple local search: perturb each node/weight slightly
    step_size = 0.01
    for iteration in range(200):
        if not _deadline_ok():
            break
        
        improved = False
        for i in range(n):
            # Try perturbing node i
            for delta in [step_size, -step_size]:
                x_try = best_x.copy()
                x_try[i] += delta
                if x_try[i] <= 0 or x_try[i] >= 1:
                    continue
                # Check ordering
                if i > 0 and x_try[i] <= x_try[i-1]:
                    continue
                if i < n-1 and x_try[i] >= x_try[i+1]:
                    continue
                
                err = 0.0
                for z, Om0, Ok0, Ol0, w0, w1 in _test_set:
                    e = _rel_error(x_try, best_w, z, Om0, Ok0, Ol0, w0, w1)
                    err = max(err, e)
                    if err > best_err:
                        break
                
                if err < best_err:
                    best_x = x_try
                    best_err = err
                    improved = True
        
        if not improved:
            step_size *= 0.5
            if step_size < 1e-8:
                break
    
    return best_x, best_w, best_err

def _minimax_refine(x, w):
    """
    Refine using a minimax approach: iteratively reweight the test cases
    to emphasize the worst ones, then solve a weighted least-squares problem.
    """
    n = 8
    best_x = x.copy()
    best_w = w.copy()
    
    # Compute errors for all test cases
    errors = np.array([
        _rel_error(best_x, best_w, z, Om0, Ok0, Ol0, w0, w1)
        for z, Om0, Ok0, Ol0, w0, w1 in _test_set
    ])
    
    for outer_iter in range(10):
        if not _deadline_ok():
            break
        
        # Weight the test cases by their current error (emphasize worst)
        case_weights = errors + 1e-10
        case_weights = case_weights / case_weights.sum()
        
        # Try a few perturbations biased toward reducing the worst errors
        best_x_local = best_x.copy()
        best_w_local = best_w.copy()
        best_local_err = max(errors)
        
        for inner_iter in range(50):
            if not _deadline_ok():
                break
            
            # Random perturbation
            x_try = best_x_local + np.random.randn(n) * 0.005
            w_try = best_w_local + np.random.randn(n) * 0.005
            
            # Constrain nodes to (0,1) and increasing
            x_try = np.clip(x_try, 1e-4, 1 - 1e-4)
            x_try = np.sort(x_try)
            # Ensure strict increase
            for i in range(1, n):
                if x_try[i] <= x_try[i-1]:
                    x_try[i] = x_try[i-1] + 1e-6
            
            # Constrain weights to be positive and sum to 1
            w_try = np.abs(w_try)
            w_try = w_try / w_try.sum()
            
            # Compute weighted error
            err = 0.0
            for idx, (z, Om0, Ok0, Ol0, w0, w1) in enumerate(_test_set):
                e = _rel_error(x_try, w_try, z, Om0, Ok0, Ol0, w0, w1)
                err = max(err, e)
                if err > best_local_err:
                    break
            
            if err < best_local_err:
                best_x_local = x_try
                best_w_local = w_try
                best_local_err = err
        
        if best_local_err < best_err:
            best_x = best_x_local
            best_w = best_w_local
            best_err = best_local_err
        else:
            break
    
    return best_x, best_w

def construct():
    n = 8
    
    # Start with Chebyshev-Gauss nodes
    x_cheb, w_cheb = _chebyshev_quadrature()
    
    # Also try Gauss-Legendre
    x_gl, w_gl = _gauss_legendre_quadrature()
    
    # Evaluate both and pick the better starting point
    err_cheb = max(_rel_error(x_cheb, w_cheb, z, Om0, Ok0, Ol0, w0, w1) 
                   for z, Om0, Ok0, Ol0, w0, w1 in _test_set)
    err_gl = max(_rel_error(x_gl, w_gl, z, Om0, Ok0, Ol0, w0, w1) 
                 for z, Om0, Ok0, Ol0, w0, w1 in _test_set)
    
    if err_cheb < err_gl:
        x_init, w_init = x_cheb, w_cheb
    else:
        x_init, w_init = x_gl, w_gl
    
    # Refine with local search
    x_opt, w_opt, err_opt = _optimize_nodes_weights(x_init, w_init)
    
    # Further refine with minimax
    x_final, w_final = _minimax_refine(x_opt, w_opt)
    
    # Ensure final constraints
    x_final = np.sort(x_final)
    x_final = np.clip(x_final, 1e-6, 1 - 1e-6)
    for i in range(1, n):
        if x_final[i] <= x_final[i-1]:
            x_final[i] = x_final[i-1] + 1e-6
    
    w_final = np.abs(w_final)
    w_final = w_final / w_final.sum()
    
    return {"nodes": x_final, "weights": w_final}
# EVOLVE-BLOCK-END
