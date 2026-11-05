"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.linalg import expm

# Global state for caching matrix exponential if detected
_matrix_exp_cache = None
_is_linear = False
_deadline = None

def _probe_linearity(f, y0):
    """
    Probe the function f to determine if it is linear (y' = A*y) and time-invariant.
    Returns A matrix if linear, None otherwise.
    """
    global _is_linear
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # Check time invariance and linearity
    # f(t, y) should be independent of t and linear in y.
    
    # 1. Check time invariance: f(t1, y) == f(t2, y)
    # We don't know t_span yet, but we can assume t=0 and t=1 for probes if we pass them.
    # However, the interface is integrate(f, t_span, y0). 
    # The probes need to be done inside integrate where t_span is known.
    
    # Let's do this logic inside integrate to have access to t_span.
    pass

def _build_matrix_exp(f, t_span, y0, A):
    """
    Compute exp(A * dt) for a given dt.
    """
    dt = t_span[1] - t_span[0]
    # We want to step in N steps.
    # If we use 1 step, h = dt.
    # If we use many steps, h = dt / N.
    # The score depends on relative error and log(calls).
    # Using expm allows us to take large steps accurately if the system is stable.
    # For heat equation, it's stable for explicit methods only if h is small, 
    # but expm is exact for the linear step.
    # So we can use very few steps.
    # However, numerical errors in expm accumulation might be an issue if N is too small?
    # No, expm is highly accurate.
    # The main error source will be the truncation error if we don't take enough steps?
    # Wait, if we use exp(A*h), that is the EXACT solution for the linear ODE y'=Ay over time h.
    # So if the system is exactly linear and time-invariant, using exp(A*h) with h=total_time 
    # gives the EXACT solution (up to floating point precision of expm).
    # This would result in 0 relative error (or machine epsilon) and very few calls.
    # Score: -log10(eps) - 0.5*log10(calls). 
    # eps ~ 1e-15 -> 15. calls ~ 10 -> 1.5. Score ~ 13.5.
    # Current score 1.0 implies much worse performance.
    
    pass

def integrate(f, t_span, y0):
    global _matrix_exp_cache, _is_linear, _deadline
    
    # Set deadline
    if _deadline is None:
        _deadline = time.time() + 320.0  # 320s to be safe
    
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # 1. Probe for linearity and time-invariance
    # We need to check if f(t, y) = A @ y.
    # Pick two distinct t values in t_span.
    t_probe1 = t0
    t_probe2 = t1
    # Pick two linearly independent y vectors.
    # y0 is one. Create another random one.
    rng = np.random.default_rng(42)
    y_probe2 = rng.normal(size=n)
    
    # Ensure y_probe2 is not proportional to y0
    if np.allclose(y0, 0):
        y_probe2 = np.ones(n)
    else:
        # Check orthogonality or just distinctness
        if np.linalg.norm(y0 - y_probe2) < 1e-6:
            y_probe2 = y_probe2 + rng.normal(size=n)

    f1 = np.array(f(t_probe1, y0))
    f2 = np.array(f(t_probe2, y0))
    
    # Check time invariance
    if not np.allclose(f1, f2, rtol=1e-9, atol=1e-12):
        # Not time-invariant, fallback to generic solver
        return _generic_solve(f, t_span, y0)

    # Check linearity: f(t, a*y1 + b*y2) == a*f(t, y1) + b*f(t, y2)
    # Let's just assume it's linear if f(0) == 0 (homogeneous) and it passes a linearity check.
    # Heat equation is homogeneous (no source term) typically.
    f_zero = np.array(f(t_probe1, np.zeros(n)))
    if not np.allclose(f_zero, 0, rtol=1e-9, atol=1e-12):
        # Non-homogeneous, fallback
        return _generic_solve(f, t_span, y0)
        
    # Check linearity with y_probe2
    f_y0 = f1
    f_yp2 = np.array(f(t_probe1, y_probe2))
    
    # Test superposition: f(y0 + y_probe2) should be f(y0) + f(y_probe2)
    y_sum = y0 + y_probe2
    f_sum = np.array(f(t_probe1, y_sum))
    if not np.allclose(f_sum, f_y0 + f_yp2, rtol=1e-9, atol=1e-12):
        # Not linear, fallback
        return _generic_solve(f, t_span, y0)
        
    # It seems linear and time-invariant.
    # Construct A matrix.
    # We have A @ y0 = f_y0 and A @ y_probe2 = f_yp2.
    # This gives 2N equations for N^2 unknowns. We need more probes to get full A.
    # Actually, for 10x10 grid, N=100. We need 100 independent vectors to determine A.
    # This would cost 100 * 2 = 200 calls.
    # Is it worth it?
    # If we just use RK4 with small steps, we might use 1000 steps * 4 calls = 4000 calls.
    # 200 calls to build A + 1 call (matrix mult) for expm step is much cheaper.
    
    # Let's build A using an identity matrix approach or random vectors.
    # To save calls, we can use the fact that for a 2D grid, A is banded/sparse.
    # But we don't know the exact stencil.
    # Let's use a set of random vectors to reconstruct A.
    # We need N linearly independent vectors.
    # Let's generate N random vectors.
    # Calls: N vectors * 1 call each = N calls.
    # Total calls so far: ~2 (time invar) + 1 (zero) + 2 (linearity) + N (A construction) = N+5.
    
    if time.time() > _deadline - 10.0:
        return _generic_solve(f, t_span, y0)
        
    # Construct A
    Y = np.empty((n, n))
    F = np.empty((n, n))
    # Use first few random vectors
    # We already have y0 and y_probe2.
    # Let's just generate N random vectors to be safe and simple.
    # This is expensive if N is large, but N=100 is small.
    # 100 calls is nothing compared to 2,000,000 budget.
    
    # Optimization: We can use the identity matrix vectors?
    # e_j -> A[:, j]. This requires N calls.
    # Let's do that.
    A = np.empty((n, n))
    for j in range(n):
        e_j = np.zeros(n)
        e_j[j] = 1.0
        A[:, j] = f(t_probe1, e_j)
        
    # Verify A against previous probes to ensure consistency
    if not np.allclose(A @ y0, f_y0, rtol=1e-9, atol=1e-12):
        return _generic_solve(f, t_span, y0)
    if not np.allclose(A @ y_probe2, f_yp2, rtol=1e-9, atol=1e-12):
        return _generic_solve(f, t_span, y0)
        
    # We have A.
    # Now solve y(t) = exp(A * t) y0.
    # We can compute exp(A * (t1-t0)) directly.
    # This is 1 matrix exponential calculation.
    # Then y_end = exp(A * dt) @ y0.
    # This is extremely accurate.
    
    dt = t1 - t0
    # Compute expm
    try:
        E = expm(A * dt)
        y_end = E @ y0
    except Exception:
        return _generic_solve(f, t_span, y0)
        
    # Check validity
    if not np.all(np.isfinite(y_end)):
        return _generic_solve(f, t_span, y0)
        
    return y_end


def _generic_solve(f, t_span, y0):
    """Fallback generic solver if structure detection fails."""
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # Use RK4 with adaptive or fixed small step.
    # Given the stiffness of heat eq, RK4 might need small h.
    # Let's use a fixed number of steps that balances error and calls.
    # 10x10 grid, heat eq. Stability limit for explicit is h < 0.25 * dx^2 / D.
    # Assuming standard domain [0,1]x[0,1], dx=0.1. h < 0.25 * 0.01 = 0.0025.
    # t_span is [0, 0.05].
    # Min steps = 0.05 / 0.0025 = 20 steps.
    # 20 steps * 4 calls = 80 calls.
    # This is very cheap.
    # Let's use 100 steps to be safe on accuracy.
    steps = 100
    h = (t1 - t0) / steps
    t, y = t0, y0.copy()
    
    for _ in range(steps):
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
        
    return y
# EVOLVE-BLOCK-END
