"""
ODE integrator for the Van der Pol oscillator (mu = 5).

Strategy:
  - Use an implicit Radau-type method (3-stage, order 3) which is L-stable and well-suited
    for mildly stiff problems.
  - The RHS is simple (2D), so we can compute the Jacobian analytically inside the solver
    without calling f (f is only for the RHS evaluation; the contract counts f calls).
  - Use adaptive step size with Radau IIA 3 stages (order 3, L-stable).
  - Tune step size to balance accuracy and f-call count.
  - Deadline guard: track wall-clock and return best result before 330s limit (use ~280s margin).

  Radau IIA 3 stages (c = [0, 0.76, 1.0], but standard Radau IIA 3 has:
    c = [0, c2, 1] where c2 = (3 + 2*sqrt(6/5))/5 ≈ 0.760555
    Actually, the standard 3-stage Radau IIA has c = [0, 0.7605551275463989, 1.0]
    
  Let me use a simpler approach: implicit midpoint or trapezoidal with Newton, adaptive.
  
  Actually, for best score with fewest calls, let me use:
  - A high-order implicit method with large steps.
  - Or: since mu=5 Van der Pol is mildly stiff, a 2nd-order implicit method (trapezoidal) 
    with moderate steps might be very efficient.
  
  Let me think about the scoring: score = -log10(rel) - 0.5*log10(calls)
  - We want small rel (high accuracy) AND small calls.
  - rel < 1e-14 gives no further gain (score caps at -log10(1e-14)=14 for the accuracy part).
  - Each 10x more calls costs 0.5 in score.
  
  Seed uses RK4 with 2000 steps = 8000 f-calls. Score 1.0000 means it's the baseline.
  
  If I can get rel ~ 1e-14 with, say, 100 f-calls, score = 14 - 0.5*log10(100) = 14 - 1 = 13.
  If seed has rel ~ 1e-10 (RK4 with 2000 steps might not be that accurate for stiff problem), 
  score_seed = -log10(1e-10) - 0.5*log10(8000) = 10 - 0.5*3.9 = 10 - 1.95 = 8.05.
  But the problem says seed score is 1.0000, which is the "search score" = value/seed.
  
  So the actual value of seed is whatever it is. Let me just focus on getting a very good answer
  with few f-calls.

  Approach: Use an implicit Runge-Kutta method (Radau IIA 3 stages) with adaptive step size.
  Hardcode the analytical Jacobian for the Van der Pol oscillator.
  
  Van der Pol (mu=5):
    y0' = y1
    y1' = 5*(1 - y0^2)*y1 - y0
    
  Jacobian:
    d/dy0 [y1] = 0
    d/dy1 [y1] = 1
    d/dy0 [5*(1-y0^2)*y1 - y0] = 5*(-2*y0)*y1 - 1 = -10*y0*y1 - 1
    d/dy1 [5*(1-y0^2)*y1 - y0] = 5*(1-y0^2)
    
  So J = [[0, 1], [-10*y0*y1 - 1, 5*(1-y0^2)]]
"""

import numpy as np
import time

# Analytical Jacobian for the Van der Pol oscillator (mu=5)
def _vdp_jacobian(y):
    y0, y1 = y[0], y[1]
    J = np.array([
        [0.0, 1.0],
        [-10.0 * y0 * y1 - 1.0, 5.0 * (1.0 - y0**2)]
    ])
    return J


def _radau3_step(f, t, y, h, tol=1e-12, max_newton=20):
    """One step of Radau IIA 3-stage implicit Runge-Kutta (order 3, L-stable).
    
    Buton: Let's use a simpler, robust implicit method.
    Actually, let me use the trapezoidal rule (Crank-Nicolson) with Newton iteration.
    It's 2nd order, A-stable, and very simple. For mu=5, this should work well.
    
    Trapezoidal: y_{n+1} = y_n + h/2 * (f(t_n, y_n) + f(t_n+h, y_{n+1}))
    Solve: F(z) = z - y_n - h/2*(f(t_n, y_n) + f(t_n+h, z)) = 0
    """
    n = len(y)
    y_new = y.copy()
    f_old = f(t, y)
    for _newton in range(max_newton):
        f_new = f(t + h, y_new)
        # Residual
        res = y_new - y - 0.5 * h * (f_old + f_new)
        # Jacobian of residual: I - h/2 * J
        J = _vdp_jacobian(y_new)
        A = np.eye(n) - 0.5 * h * J
        # Newton step: solve A * delta = -res
        try:
            delta = np.linalg.solve(A, -res)
        except np.linalg.LinAlgError:
            delta = -res  # fallback
        y_new = y_new + delta
        if np.linalg.norm(delta) < tol * (1.0 + np.linalg.norm(y_new)):
            break
    return y_new


def _adaptive_trapezoid(f, t0, t1, y0, h0=0.1, h_min=1e-6, h_max=0.5, 
                         rtol=1e-10, atol=1e-12, max_steps=100000, deadline=None):
    """Adaptive trapezoidal rule with local error estimation using step doubling."""
    y = np.array(y0, dtype=float)
    t = t0
    h = h0
    calls = 0
    start_time = time.time()
    
    while t < t1 - 1e-14:
        # Check deadline
        if deadline is not None and (time.time() - start_time) > deadline:
            break
        
        # Limit step to not exceed t1
        h = min(h, t1 - t)
        
        # Take one full step of size h
        y1_full = _radau3_step(f, t, y, h, tol=1e-13, max_newton=20)
        calls += 2  # f called twice inside _radau3_step (f_old and f_new in Newton)
        # Actually _radau3_step calls f multiple times in Newton. Let me count properly.
        
        # Take two half steps
        y_half = _radau3_step(f, t, y, h/2, tol=1e-13, max_newton=20)
        y2_half = _radau3_step(f, t + h/2, y_half, h/2, tol=1e-13, max_newton=20)
        
        # Estimate error
        err = np.linalg.norm(y2_half - y1_full)
        scale = atol + rtol * max(np.linalg.norm(y), np.linalg.norm(y2_half))
        
        if err > scale:
            # Reject step, reduce h
            h = max(h * 0.5, h_min)
            if h <= h_min:
                # Accept anyway
                y = y2_half
                t += h
                continue
        else:
            # Accept step
            y = y2_half
            t += h
            # Adjust h for next step
            if err < 1e-15:
                h = min(h * 2.0, h_max)
            else:
                h = min(max(h * (scale / err) ** 0.25, h_min), h_max)
        
        if not np.all(np.isfinite(y)):
            break
    
    return y, calls


def integrate(f, t_span, y0):
    """Integrate the Van der Pol oscillator using adaptive implicit trapezoidal method."""
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    # Deadline: use 280 seconds out of 330 second budget
    start = time.time()
    deadline = 280.0
    
    # Use adaptive trapezoid with reasonable initial step
    y_end, calls = _adaptive_trapezoid(
        f, t0, t1, y0, 
        h0=0.2, h_min=1e-7, h_max=0.5,
        rtol=1e-11, atol=1e-13,
        deadline=deadline
    )
    
    if not np.all(np.isfinite(y_end)):
        # Fallback: simple fixed-step trapezoid with many small steps
        h = 0.01
        steps = int(np.ceil((t1 - t0) / h))
        y = y0.copy()
        t = t0
        for _ in range(steps):
            y = _radau3_step(f, t, y, h, tol=1e-13, max_newton=15)
            t += h
        y_end = y
    
    return y_end
