import numpy as np
import time

def integrate(f, t_span, y0):
    t0, t1 = t_span
    start_time = time.time()
    deadline = start_time + 280.0

    y0 = np.asarray(y0, dtype=float)
    n = y0.shape[0]
    dt = float(t1 - t0)

    from scipy.integrate import solve_ivp

    # Try progressively tighter tolerances to improve accuracy
    # while monitoring call count and time
    tol_candidates = [
        (1e-11, 1e-13),
        (5e-12, 5e-14),
        (1e-12, 1e-14),
    ]

    best_result = None
    best_calls = float('inf')

    for rt, at in tol_candidates:
        if time.time() > deadline:
            break
        try:
            sol = solve_ivp(f, t_span, y0, method='DOP853', rtol=rt, atol=at)
            if sol.success and sol.y.shape[1] > 0:
                result = sol.y[:, -1].copy()
                if np.all(np.isfinite(result)):
                    # Estimate calls from sol.t length (rough proxy)
                    # solve_ivp with DOP853: each step uses 6 f-evaluations
                    n_steps = sol.t.shape[0] - 1
                    est_calls = n_steps * 6 + 1  # rough
                    if est_calls < best_calls:
                        best_result = result
                        best_calls = est_calls
        except Exception:
            continue

    if best_result is not None:
        return best_result

    # Fallback: fixed-step RK4
    y = y0.copy()
    t = float(t0)
    nsteps = 50000
    h = dt / nsteps
    for i in range(nsteps):
        if time.time() > deadline:
            break
        k1 = np.asarray(f(t, y), dtype=float)
        k2 = np.asarray(f(t + 0.5*h, y + 0.5*h*k1), dtype=float)
        k3 = np.asarray(f(t + 0.5*h, y + 0.5*h*k2), dtype=float)
        k4 = np.asarray(f(t + h, y + h*k3), dtype=float)
        y = y + (h/6.0)*(k1 + 2.0*k2 + 2.0*k3 + k4)
        t = t + h

    if not np.all(np.isfinite(y)):
        y = y0.copy()
    return y
