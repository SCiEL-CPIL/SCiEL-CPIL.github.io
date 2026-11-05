"""
Extrapolated RK4 solver for the 100-point convection-diffusion problem.

Strategy:
  - Use a fixed-step classical RK4 with a modest number of steps (e.g. 4000).
  - Perform Richardson extrapolation: compute the solution with N steps and 2N steps,
    then extrapolate to the zero-step limit using the formula (2^4 * y_2N - y_N) / (2^4 - 1).
  - This yields a 5th-order accurate result with only about 3x the cost of a single RK4 pass.
  - Additionally, we can do a second level of extrapolation using N, 2N, 4N to get 6th order,
    but that triples the cost again. We'll stick to one level for a good balance.
  - We also try a higher base N (e.g. 8000) if time permits, and pick the result with the
    best estimated error (via comparing N and 2N) within the time budget.

  This is materially different from adaptive DOP853/Radau: it's a fixed-step explicit method
  with post-hoc Richardson extrapolation, exploiting the known 4th-order convergence of RK4.
"""

import numpy as np
import time


def _rk4_fixed(f, t0, t1, y0, nsteps):
    """Integrate y' = f(t, y) from t0 to t1 using fixed-step RK4 with exactly nsteps steps."""
    h = (t1 - t0) / nsteps
    t = t0
    y = np.array(y0, dtype=float, copy=True)
    for _ in range(nsteps):
        k1 = f(t, y)
        k2 = f(t + 0.5 * h, y + 0.5 * h * k1)
        k3 = f(t + 0.5 * h, y + 0.5 * h * k2)
        k4 = f(t + h, y + h * k3)
        y = y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        t += h
    return y


def integrate(f, t_span, y0):
    """
    Integrate y' = f(t, y) from t_span[0] to t_span[1].
    Uses fixed-step RK4 with Richardson extrapolation.
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    
    start_time = time.time()
    deadline = start_time + 280  # Safety margin before 330s kill
    
    # Try several base step counts, from coarse to fine.
    # For each, compute y_N and y_2N, extrapolate, and check time.
    # We want the most accurate result we can get within the time budget.
    
    # Strategy: start with a reasonable N, then increase if time allows.
    # RK4 with N steps: 4*N function calls.
    # Richardson: need y_N (4N calls) and y_2N (8N calls) = 12N total calls.
    # Extrapolated result: y_ext = (16 * y_2N - y_N) / 15
    
    # Let's try N values: 2000, 4000, 8000, 16000
    # But we need to be careful about time.
    # For N=2000: 12*2000 = 24000 calls
    # For N=4000: 12*4000 = 48000 calls
    # For N=8000: 12*8000 = 96000 calls
    # For N=16000: 12*16000 = 192000 calls
    
    # Each call is O(100) operations, so even 200000 calls should be fast.
    # Let's be aggressive but safe.
    
    n_candidates = [2000, 4000, 8000, 16000, 32000]
    
    best_y = None
    best_est_err = float('inf')
    best_calls = 0
    
    for n in n_candidates:
        if time.time() > deadline - 30:  # Leave 30s margin
            break
        
        # Compute y with n steps
        y_n = _rk4_fixed(f, t0, t1, y0, n)
        calls_n = 4 * n
        
        if time.time() > deadline - 20:
            break
        
        # Compute y with 2n steps
        y_2n = _rk4_fixed(f, t0, t1, y0, 2 * n)
        calls_2n = 8 * n
        
        total_calls = calls_n + calls_2n
        
        # Richardson extrapolation: y_ext = (16 * y_2n - y_n) / 15
        y_ext = (16.0 * y_2n - y_n) / 15.0
        
        if not np.all(np.isfinite(y_ext)):
            continue
        
        # Estimate error: ||y_2n - y_n|| as a proxy
        est_err = np.max(np.abs(y_2n - y_n))
        
        # We want to minimize est_err, but also consider the call penalty.
        # Score = -log10(err) - 0.5*log10(calls)
        # We don't know the true err, but we can track est_err and calls.
        # For now, just keep the one with the smallest est_err.
        # A better approach: compute a score-like metric.
        # But since we don't know the reference, we'll just take the most refined
        # result that fits in time.
        
        if est_err < best_est_err:
            best_est_err = est_err
            best_y = y_ext
            best_calls = total_calls
    
    if best_y is not None:
        return best_y
    
    # Fallback: just do a fixed RK4 with a reasonable number of steps
    n = 4000
    y = _rk4_fixed(f, t0, t1, y0, n)
    return y
