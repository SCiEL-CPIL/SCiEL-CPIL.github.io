import numpy as np
import time
from scipy.optimize import differential_evolution

N = 400
_DEADLINE = 145.0  # wall-clock limit with safety margin

def _exact_value(h):
    """Compute the exact task value."""
    s = np.sum(h)
    if s == 0 or not np.isfinite(s):
        return np.inf
    conv = np.convolve(h, h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def _param_to_h(params):
    """
    Convert parameters to step heights.
    params: [A, f0, beta, phi]
    h[n] = A * cos(2*pi*f0*n + beta*n^3 + phi)
    """
    A, f0, beta, phi = params
    n = np.arange(N)
    # Use cubic phase for better decorrelation than quadratic (linear chirp)
    # The term beta*n^3 creates a linear frequency sweep with quadratic rate of change
    phase = 2.0 * np.pi * f0 * n + beta * n**3 + phi
    return A * np.cos(phase)

def _objective(params):
    """Objective function for differential evolution."""
    h = _param_to_h(params)
    val = _exact_value(h)
    if not np.isfinite(val):
        return 1e10
    return val

def construct():
    t0 = time.time()
    deadline = t0 + _DEADLINE
    
    # Initial seed
    best_h = np.ones(N, dtype=float)
    best_val = _exact_value(best_h)  # 2.0

    # Define bounds for Differential Evolution
    # A: Amplitude (1.0 is seed scale)
    # f0: Base frequency (0 to 0.5 cycles/sample)
    # beta: Cubic phase coefficient (small values)
    # phi: Phase offset (0 to 2*pi)
    bounds = [
        (0.5, 2.0),       # A
        (0.0, 0.5),       # f0
        (-1e-4, 1e-4),    # beta (cubic term needs to be small to stay within range)
        (0.0, 2.0 * np.pi) # phi
    ]
    
    # Try a few different parametric forms to diversify search
    # Form 1: Pure cubic chirp
    # Form 2: Quadratic chirp (standard linear chirp) for comparison/fallback
    # Form 3: Mixed
    
    strategies = [
        {
            "name": "cubic",
            "param_to_h": _param_to_h,
            "bounds": bounds,
            "bounds_scale": 1.0
        },
        {
            "name": "quad",
            "param_to_h": lambda p: p[0] * np.cos(2.0 * np.pi * p[1] * np.arange(N) + p[2] * np.arange(N)**2 + p[3]),
            "bounds": [
                (0.5, 2.0),       # A
                (0.0, 0.5),       # f0
                (-0.05, 0.05),    # beta (quadratic term)
                (0.0, 2.0 * np.pi) # phi
            ],
            "bounds_scale": 1.0
        }
    ]
    
    rng = np.random.default_rng(42)
    
    for strat in strategies:
        if time.time() > deadline - 30:
            break
            
        p2h = strat["param_to_h"]
        bnds = strat["bounds"]
        
        # Run Differential Evolution
        # Max iterations and popsize tuned for speed within budget
        # popsize=20, maxiter=100 is usually fast enough for 4 params
        try:
            result = differential_evolution(
                _objective_wrapper(p2h), 
                bnds, 
                maxiter=100, 
                popsize=20, 
                tol=1e-4,
                seed=42,
                polish=True,
                init='latinhypercube'
            )
        except Exception as e:
            # If DE fails, fallback to random search
            result = None
            
        if result is not None and result.success and result.fun < best_val:
            best_h = p2h(result.x)
            best_val = result.fun
            
        # Also do a local random search around the best found or initial guess
        # This helps if DE gets stuck or if the landscape is noisy
        
        # If DE didn't find something better, try a few random chirps
        for _ in range(50):
            if time.time() > deadline - 10:
                break
            # Random params within bounds
            A = rng.uniform(bnds[0][0], bnds[0][1])
            f0 = rng.uniform(bnds[1][0], bnds[1][1])
            beta = rng.uniform(bnds[2][0], bnds[2][1])
            phi = rng.uniform(bnds[3][0], bnds[3][1])
            h_try = p2h([A, f0, beta, phi])
            v_try = _exact_value(h_try)
            if np.isfinite(v_try) and v_try < best_val:
                best_val = v_try
                best_h = h_try

    # Final refinement: Coordinate descent on the raw h vector
    # This allows the chirp structure to be broken slightly if beneficial
    if time.time() < deadline - 5:
        # Use a small step size to avoid jumping over good local minima
        step_size = 0.05
        for _ in range(5000):
            if time.time() > deadline - 3:
                break
            idx = rng.integers(0, N)
            h_new = best_h.copy()
            h_new[idx] += step_size
            v_new = _exact_value(h_new)
            if np.isfinite(v_new) and v_new < best_val:
                best_val = v_new
                best_h = h_new
            else:
                h_new[idx] -= 2 * step_size
                v_new2 = _exact_value(h_new)
                if np.isfinite(v_new2) and v_new2 < best_val:
                    best_val = v_new2
                    best_h = h_new

    # Final validation
    final_val = _exact_value(best_h)
    if not np.isfinite(final_val) or final_val >= 2.0:
        best_h = np.ones(N)
        
    return best_h

def _objective_wrapper(param_func):
    """
    Create a callable objective for DE that uses the specific param-to-h function.
    """
    def obj(params):
        h = param_func(params)
        val = _exact_value(h)
        if not np.isfinite(val):
            return 1e10
        return val
    return obj
