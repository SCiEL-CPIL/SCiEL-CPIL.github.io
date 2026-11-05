import time
import numpy as np
from scipy.integrate import solve_ivp


def integrate(f, t_span, y0):
    """Integrate y' = f(t, y) over t_span and return y at t_end."""
    t0, t1 = t_span
    y0_arr = np.asarray(y0, dtype=float)
    n = y0_arr.size
    deadline = time.time() + 285.0
    T = t1 - t0

    best_y = None
    best_score = -np.inf

    # Strategy: Use DOP853 with very tight tolerances.
    # The score is -log10(rel_err) - 0.5*log10(n_calls).
    # DOP853 is 8th order, so error ~ C*h^8.
    # For a fixed number of calls N_calls, we have ~N_calls/12 steps.
    # h = T / (N_calls/12) = 12*T/N_calls.
    # rel_err ~ C * (12*T/N_calls)^8.
    # score ~ -log10(C) - 8*log10(12*T) + 8*log10(N_calls) - 0.5*log10(N_calls)
    # score ~ 7.5*log10(N_calls) + const.
    # So we want to maximize N_calls subject to time and call limits.
    # However, adaptive solvers may use fewer steps if tolerances are loose.
    # We force many steps by setting max_step small and tolerances tight.
    #
    # The key insight: DOP853 is very accurate per call.
    # We should use as many calls as possible within the time budget.
    #
    # Let's estimate the time per call. For n=100, one f call is cheap.
    # Let's do a quick benchmark if possible, or just use a conservative estimate.
    # Actually, we can just run solve_ivp with a large max_step constraint
    # and tight tolerances, and let it run until close to the deadline.
    # But solve_ivp doesn't have a "stop at time" feature.
    # We can use a custom step size controller or just set max_step to force
    # a certain number of steps.
    #
    # Let's try to use the full time budget.
    # We'll do a first pass with a moderate number of steps to get a baseline,
    # then if time remains, do a second pass with more steps.
    #
    # Actually, a better approach:
    # 1. Estimate time per step from a small trial.
    # 2. Compute max steps possible within deadline.
    # 3. Run solve_ivp with max_step = T / max_steps and tight tolerances.
    #
    # Let's do a quick timing trial.
    t_start = time.time()
    # Small trial: 10 steps
    trial_steps = 10
    h_trial = T / trial_steps
    try:
        sol_trial = solve_ivp(
            f, t_span, y0_arr,
            method='DOP853',
            rtol=1e-10, atol=1e-12,
            max_step=h_trial
        )
        t_trial = time.time() - t_start
        # Estimate calls per step: DOP853 uses 12 evaluations per step
        calls_per_step = 12
        time_per_call = t_trial / (trial_steps * calls_per_step)
    except Exception:
        time_per_call = 1e-5  # conservative default: 10 microseconds per call

    # Max calls allowed by time
    max_calls_by_time = int((deadline - time.time()) / time_per_call)
    # Max calls allowed by the problem limit
    max_calls_by_limit = 2000000
    max_calls = min(max_calls_by_time, max_calls_by_limit)
    max_calls = max(max_calls, 120)  # at least 10 steps

    # Number of steps
    max_steps = max_calls // calls_per_step
    max_steps = max(max_steps, 10)

    h_max = T / max_steps

    # Run with tight tolerances and max_step constraint
    try:
        sol = solve_ivp(
            f, t_span, y0_arr,
            method='DOP853',
            rtol=1e-13,
            atol=1e-15,
            max_step=h_max
        )
        if sol.success and np.all(np.isfinite(sol.y[:, -1])):
            # Estimate actual calls
            # solve_ivp doesn't directly expose nfev in a simple way for all methods
            # but we can estimate from the number of steps taken
            n_steps_actual = len(sol.t)
            n_calls_est = n_steps_actual * calls_per_step
            # The actual error is unknown, but with rtol=1e-13, we expect
            # relative error around 1e-13 or better.
            rel_err_est = 1e-13
            score_est = -np.log10(rel_err_est) - 0.5 * np.log10(max(n_calls_est, 1))
            if score_est > best_score:
                best_score = score_est
                best_y = sol.y[:, -1].copy()
    except Exception:
        pass

    # If we have significant time left, try even more steps
    remaining = deadline - time.time()
    if best_y is not None and remaining > 30.0:
        # Try with 2x the steps
        max_steps2 = min(max_steps * 2, 2000000 // calls_per_step)
        if max_steps2 > max_steps:
            h_max2 = T / max_steps2
            try:
                sol2 = solve_ivp(
                    f, t_span, y0_arr,
                    method='DOP853',
                    rtol=1e-14,
                    atol=1e-16,
                    max_step=h_max2
                )
                if sol2.success and np.all(np.isfinite(sol2.y[:, -1])):
                    n_steps_actual2 = len(sol2.t)
                    n_calls_est2 = n_steps_actual2 * calls_per_step
                    rel_err_est2 = 1e-14
                    score_est2 = -np.log10(rel_err_est2) - 0.5 * np.log10(max(n_calls_est2, 1))
                    if score_est2 > best_score:
                        best_score = score_est2
                        best_y = sol2.y[:, -1].copy()
            except Exception:
                pass

    if best_y is not None:
        return best_y

    # Fallback: fixed-step RK4
    remaining = deadline - time.time()
    max_steps_rk4 = int(remaining / (4 * time_per_call))
    max_steps_rk4 = min(max_steps_rk4, 2000000 // 4)
    max_steps_rk4 = max(max_steps_rk4, 100)

    h = T / max_steps_rk4
    t = t0
    y = y0_arr.copy()

    for i in range(max_steps_rk4):
        if time.time() > deadline - 5.0:
            break
        k1 = f(t, y)
        k2 = f(t + 0.5 * h, y + 0.5 * h * k1)
        k3 = f(t + 0.5 * h, y + 0.5 * h * k2)
        k4 = f(t + h, y + h * k3)
        y = y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        t += h

    return y
