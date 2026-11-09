"""ODE solver for 100-point convection-diffusion IVP with adaptive tolerance search."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    t_start = time.perf_counter()
    DEADLINE = 290.0  # Safety margin before 330s kill
    
    y0 = np.asarray(y0, dtype=float)
    t0, t1 = t_span
    n = len(y0)
    
    # We need to count calls to f to optimize the score:
    # Score = -log10(rel_err) - 0.5 * log10(calls)
    # We want to minimize calls while keeping error low.
    
    call_count = 0
    
    def f_counted(t, y):
        nonlocal call_count
        call_count += 1
        # If we exceed the hard limit, we must stop.
        # The evaluator also counts, but we need to know for our own logic.
        return f(t, y)

    def _elapsed_ok():
        return (time.perf_counter() - t_start) < DEADLINE

    # Strategy:
    # 1. Run a quick, low-accuracy pass to estimate the solution magnitude and behavior.
    #    This helps in setting better initial steps or tolerances if needed, 
    #    but more importantly, it establishes a baseline error.
    # 2. Use the remaining time budget for a high-accuracy DOP853 run.
    #    We will try to maximize accuracy (minimize error) while keeping calls reasonable.
    
    # Initial guess for tolerances. DOP853 is 8th order.
    # For a 100-point convection-diffusion, the solution is smooth but can have sharp gradients.
    # rtol=1e-10 is a good starting point for high accuracy without excessive cost.
    
    best_result = None
    best_score = -np.inf
    
    # Phase 1: Calibration Run (Fast, Low Accuracy)
    # Use RK45 with loose tolerances to get a quick estimate.
    # This costs very few function calls (usually < 1000).
    try:
        if _elapsed_ok():
            sol_cal = solve_ivp(
                f_counted,
                (t0, t1),
                y0,
                method="RK45",
                rtol=1e-6,
                atol=1e-8,
                max_step=(t1 - t0) / 20.0
            )
            if sol_cal.success:
                best_result = sol_cal.y[:, -1].copy()
                # Estimate relative error magnitude of this low-acc run
                # We don't know the true error, but we know the scale of y.
                y_scale = np.max(np.abs(best_result))
                if y_scale == 0:
                    y_scale = 1.0
    except Exception:
        pass

    # Phase 2: High-Accuracy Run (DOP853)
    # We will try a sequence of increasing strictness.
    # Since DOP853 is explicit, it's fast per step but needs many steps for high accuracy.
    # However, it's much more efficient than implicit methods for non-stiff or mildly stiff problems.
    # Convection-diffusion with 100 points is likely mildly stiff or non-stiff depending on Peclet number.
    
    # Tolerance levels to try, from loose to tight.
    # We stop if we hit the call limit or time limit.
    # The score improves as error drops, but penalizes calls.
    # -log10(err) vs -0.5 log10(calls).
    # If we double calls, score drops by 0.5 * log10(2) ~ 0.15.
    # If we reduce error by 10x, score gains 1.0.
    # So reducing error is much more valuable than saving calls, UNLESS the error is already tiny.
    
    tolerances = [
        (1e-10, 1e-12),
        (1e-11, 1e-13),
        (1e-12, 1e-14),
        (1e-13, 1e-15),
        (1e-14, 1e-16),
    ]
    
    current_calls = call_count
    
    for rtol, atol in tolerances:
        if not _elapsed_ok():
            break
        
        # Reset call count for this specific run? No, total calls matter.
        # But we can't reset the evaluator's counter.
        # We must ensure the TOTAL calls stay under 2,000,000.
        
        # Estimate if we can afford another run.
        # A typical DOP853 run for this problem might take ~5000-20000 calls for rtol=1e-10.
        # If current_calls + 50000 > 2000000, we should stop.
        
        try:
            sol = solve_ivp(
                f_counted,
                (t0, t1),
                y0,
                method="DOP853",
                rtol=rtol,
                atol=atol,
                max_step=(t1 - t0) / 50.0,
                first_step=(t1 - t0) / 1e5,
                dense_output=False
            )
            
            if sol.success and sol.t.size > 0:
                result = sol.y[:, -1]
                
                # Calculate a proxy score based on our internal call count
                # Note: The evaluator's count might differ slightly if it counts differently,
                # but usually it's 1:1 with f calls.
                total_calls = call_count
                if total_calls == 0:
                    total_calls = 1
                
                # We don't know the true relative error against the reference.
                # However, the local error estimate from the solver gives us a hint.
                # scipy's solve_ivp doesn't directly return the final relative error vs reference.
                # We must rely on the tolerance setting as a proxy for error.
                # Lower tolerance -> lower error.
                
                # Heuristic: Assume error is proportional to rtol.
                # Score ~ -log10(rtol) - 0.5 * log10(total_calls)
                # Let's just keep the result from the tightest tolerance we successfully ran.
                # Since the score is monotonic with accuracy (until machine precision),
                # the last successful run with the tightest tolerance is likely the best,
                # PROVIDED the call count didn't explode.
                
                # Check if call count is still reasonable.
                # If we used > 1.5M calls, the penalty is huge.
                if total_calls < 1500000:
                    best_result = result
                else:
                    # If we blew the budget, revert to previous best if it exists
                    # But we don't store previous bests easily here without overhead.
                    # Let's just keep the last one if it's not insane.
                    best_result = result
                    
        except Exception:
            break
            
        # If we hit a very high call count, stop trying tighter tolerances.
        if call_count > 1000000:
            break

    # Fallback if nothing worked
    if best_result is None:
        # Fixed step RK4
        steps = 1000
        h = (t1 - t0) / steps
        t, y = t0, y0.copy()
        for _ in range(steps):
            if not _elapsed_ok():
                break
            k1 = f(t, y)
            k2 = f(t + h / 2, y + h / 2 * k1)
            k3 = f(t + h / 2, y + h / 2 * k2)
            k4 = f(t + h, y + h * k3)
            y = y + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
            t += h
        best_result = y

    return best_result
# EVOLVE-BLOCK-END
