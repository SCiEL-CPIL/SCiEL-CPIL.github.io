import numpy as np
import time
from scipy.optimize import minimize

N = 400
_EPS = 1e-6  # scale-anchoring penalty strength (negligible effect on value)
_DEADLINE = 130.0  # safety margin before 150s kill


def _value_and_grad(h):
    """Return (value + penalty, gradient). value is scale-invariant."""
    S = float(np.sum(h))
    if abs(S) < 1e-12:
        return np.inf, np.zeros_like(h)
    c = np.convolve(h, h)
    m = float(np.max(c))
    jstar = int(np.argmax(c))
    val = 2.0 * N * m / (S * S)

    # dm/dh[i] = 2*h[jstar-i] when 0 <= jstar-i < N else 0
    idx = jstar - np.arange(N)
    valid = (idx >= 0) & (idx < N)
    g = np.zeros(N)
    g[valid] = 2.0 * h[idx[valid]]

    grad_val = 2.0 * N * (g / (S * S) - 2.0 * m / (S ** 3))

    # Tiny penalty to fix the (flat) scale direction -> well-conditioned BFGS
    penalty = 0.5 * _EPS * (S - 1.0) ** 2
    grad_pen = _EPS * (S - 1.0) * np.ones(N)
    return val + penalty, grad_val + grad_pen


def _sinusoidal_value_and_grad(params):
    """
    2-parameter sinusoidal parameterization:
        h[i] = a + b * cos(2*pi*k*i/N + phase)
    
    params = [a, b, phase, k] where k is treated as continuous for gradient.
    We optimize over (a, b, phase) with k fixed per call, or use a grid over k.
    """
    a, b, phase, k = params
    i = np.arange(N)
    h = a + b * np.cos(2.0 * np.pi * k * i / N + phase)
    
    S = float(np.sum(h))
    if abs(S) < 1e-12:
        return 1e10, np.zeros(4)
    
    c = np.convolve(h, h)
    m = float(np.max(c))
    val = 2.0 * N * m / (S * S)
    
    # Compute gradient via finite differences for simplicity (2-param is cheap)
    eps = 1e-7
    grad = np.zeros(4)
    for j in range(3):  # a, b, phase
        p_plus = params.copy()
        p_plus[j] += eps
        v_plus = _sinusoidal_value_no_grad(p_plus)
        p_minus = params.copy()
        p_minus[j] -= eps
        v_minus = _sinusoidal_value_no_grad(p_minus)
        grad[j] = (v_plus - v_minus) / (2 * eps)
    
    return val, grad


def _sinusoidal_value_no_grad(params):
    """Evaluate the value for sinusoidal params without gradient."""
    a, b, phase, k = params
    i = np.arange(N)
    h = a + b * np.cos(2.0 * np.pi * k * i / N + phase)
    S = float(np.sum(h))
    if abs(S) < 1e-12:
        return 1e10
    c = np.convolve(h, h)
    m = float(np.max(c))
    return 2.0 * N * m / (S * S)


def construct():
    t0 = time.time()
    deadline = _DEADLINE - 5.0  # safety margin

    best_h = np.ones(N, dtype=float)
    best_val, _ = _value_and_grad(best_h)  # seed value = 2.0

    rng = np.random.default_rng(20240601)

    # Phase 1: Grid search over frequency k, then optimize (a, b, phase)
    # For each k, the problem reduces to 3 parameters
    for k in range(1, N // 2 + 1):
        if (time.time() - t0) > (deadline - 20.0):
            break
        
        # Try multiple phase initializations
        for phase_init in [0.0, np.pi / 4, np.pi / 2, 3.0 * np.pi / 4, np.pi]:
            if (time.time() - t0) > (deadline - 20.0):
                break
            
            # Try multiple (a, b) initializations
            for a_init in [0.0, 1.0, -1.0, 0.5, -0.5]:
                for b_init in [0.5, 1.0, 2.0, 0.1, -0.5, -1.0]:
                    if (time.time() - t0) > (deadline - 20.0):
                        break
                    
                    params0 = np.array([a_init, b_init, phase_init, float(k)])
                    
                    def obj_2d(x):
                        v, _ = _sinusoidal_value_and_grad(x)
                        return v
                    
                    def grad_2d(x):
                        _, g = _sinusoidal_value_and_grad(x)
                        return g
                    
                    try:
                        res = minimize(obj_2d, params0, jac=grad_2d,
                                       method="L-BFGS-B",
                                       bounds=[(-10, 10), (-10, 10), (-np.pi, np.pi), (float(k), float(k))],
                                       options={"maxiter": 100, "maxfun": 300,
                                                "ftol": 1e-12, "gtol": 1e-10})
                        
                        if np.isfinite(res.fun) and res.fun < best_val:
                            a, b, phase, _ = res.x
                            i = np.arange(N)
                            h_candidate = a + b * np.cos(2.0 * np.pi * k * i / N + phase)
                            S = float(np.sum(h_candidate))
                            if abs(S) > 1e-9:
                                best_val = res.fun
                                best_h = h_candidate.copy()
                    except Exception:
                        pass
    
    # Phase 2: Full L-BFGS-B optimization from sinusoidal solutions and other starts
    def objective(h):
        v, _ = _value_and_grad(h)
        return v

    def grad(h):
        _, gg = _value_and_grad(h)
        return gg

    # Start from best sinusoidal solution
    start_points = [best_h.copy()]
    
    # Add some diverse starting points
    start_points.append(np.ones(N))
    start_points.append(rng.standard_normal(N))
    start_points.append(np.cumsum(rng.standard_normal(N)) - np.mean(rng.standard_normal(N)))
    
    for h0 in start_points:
        if (time.time() - t0) > (deadline - 5.0):
            break
        
        if abs(float(np.sum(h0))) < 1e-6:
            h0 = h0 + 1.0
        
        try:
            res = minimize(objective, h0, jac=grad, method="L-BFGS-B",
                           options={"maxiter": 200, "maxfun": 500,
                                    "ftol": 1e-14, "gtol": 1e-12})
            if np.isfinite(res.fun) and res.fun < best_val:
                best_val = res.fun
                best_h = res.x.copy()
        except Exception:
            pass
    
    # Phase 3: Local refinement with tighter tolerances
    if (time.time() - t0) < (deadline - 3.0):
        try:
            res = minimize(objective, best_h, jac=grad, method="L-BFGS-B",
                           options={"maxiter": 300, "maxfun": 1000,
                                    "ftol": 1e-15, "gtol": 1e-14})
            if np.isfinite(res.fun) and res.fun < best_val:
                best_val = res.fun
                best_h = res.x.copy()
        except Exception:
            pass

    # Rescale to sum=1 (value is scale-invariant, so this is safe) for validity
    S = float(np.sum(best_h))
    if abs(S) > 1e-9:
        best_h = best_h / S
    else:
        best_h = best_h + 1.0
    return best_h
