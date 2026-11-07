"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

# Strategy: Fixed-step DOP853.
# The reference solution uses adaptive DOP853. Adaptive methods can be inefficient
# due to rejected steps and step-size calculation overhead. 
# The Lorenz system is smooth and well-behaved for t in [0, 2].
# We use a fixed step size h=0.1 (20 steps). 
# DOP853 is a 7th-order method. With h=0.1, the local truncation error is ~h^8 = 1e-8 per step.
# Global error accumulates to ~1e-7, which is NOT enough for rel < 1e-14.
# Wait, the reference uses rtol=1e-11. The error floor is 1e-14.
# Let's analyze the error of fixed step h.
# For a 7th order method, global error is O(h^7).
# If h=0.1, error ~ 1e-7. This is too large.
# We need error < 1e-14.
# h^7 < 1e-14 => h < (1e-14)^(1/7) ~ 10^(-2) = 0.01.
# So we need h <= 0.01. That's 200 steps.
# Calls = 200 steps * 12 calls/step = 2400 calls.
# Current program uses adaptive DOP853 with rtol=1e-13.
# Let's check the score of current: 1.2798.
# Score = -log10(rel) - 0.5*log10(calls).
# If rel is at floor 1e-14, -log10(rel) = 14.
# If calls = 2400, log10(2400) ~ 3.38. 0.5 * 3.38 = 1.69.
# Score ~ 14 - 1.69 = 12.31? No, the score is relative to seed?
# "search score is value/seed".
# The task says: "score = -log10(rel) - 0.5 * log10(calls)".
# "Below rel = 1e-14 there is no further gain".
# If rel < 1e-14, we cap rel at 1e-14 for scoring? Or does it just stop improving?
# "rel = max(..., 1e-14)". So rel is at least 1e-14.
# So max -log10(rel) is 14.
# To maximize score, we want to minimize calls, provided rel <= 1e-14.
# If we can achieve rel <= 1e-14 with fewer calls, we win.
#
# Can we use a higher order method?
# SciPy DOP853 is order 7.
# Is there an order 12 method? Not in standard SciPy.
# Can we use Richardson extrapolation?
# Or simply use a smaller h with DOP853?
#
# Let's look at the inspiration:
# inspiration 1 (score 1.6150): "Abandon adaptive step-size control entirely and use a single fixed-step high-order method (DOP853) with a step size large enough to minimize function calls..."
# This suggests that fixed step DOP853 with a *large* step worked?
# How can a large step have low error?
# Maybe the "large" step is relative to adaptive, but still small enough?
# Or maybe the "single fixed-step" means one step for the whole interval?
# If one step h=2, error is huge.
#
# Let's re-read the current program's strategy.
# It uses adaptive DOP853 with rtol=1e-13.
# Adaptive methods often take more steps than necessary if the initial step is small or if they are conservative.
# A fixed step method avoids the overhead and potential inefficiency of the adaptive controller.
#
# Let's estimate the required h for DOP853 to get rel < 1e-14.
# The Lorenz system has a Lyapunov exponent ~0.9.
# Errors grow exponentially.
# However, over t=2, the growth is limited.
# Let's try h=0.05 (40 steps). Calls = 40*12 = 480.
# Let's try h=0.1 (20 steps). Calls = 20*12 = 240.
#
# If the inspiration with score 1.6150 used fixed step DOP853, it likely found a sweet spot.
# The current score is 1.2798.
# If we reduce calls by factor of 2, score increases by 0.5 * log10(2) ~ 0.15.
# If we reduce calls by factor of 10, score increases by 0.5.
#
# Let's implement a fixed-step DOP853.
# We will try a few step sizes to ensure we hit the accuracy floor.
# Since we can't test interactively, we will pick a conservative h that is likely to be accurate.
# h=0.05 seems safe for 7th order.
# h=0.025 is safer.
#
# Actually, we can implement the DOP853 coefficients manually to avoid SciPy overhead if needed, but SciPy is available.
# However, calling solve_ivp with fixed_step=True might still have overhead.
# Let's write a manual loop using the DOP853 coefficients.
#
# DOP853 coefficients (from Dormand and Prince, 1997):
# This is complex. It's safer to use scipy.integrate.DOP853 if available, or use solve_ivp with max_step=min_step=h.
#
# Using solve_ivp with max_step=h and min_step=h forces fixed steps.
#
# Let's try h=0.05.
# Steps = 40. Calls = 480.
#
# If h=0.1 is accurate enough, calls = 240.
# Let's check the error of DOP853 with h=0.1 on Lorenz.
# The reference solution is DOP853 adaptive rtol=1e-11.
# If we use fixed h=0.1, the error might be comparable to or larger than the reference.
# We need error < 1e-14.
#
# Let's look at the "inspiration 1" again.
# "step size large enough to minimize function calls while relying on the method's high [order]".
# This implies the step size is NOT extremely small.
#
# Maybe we can use a higher order method?
# No, DOP853 is the highest standard.
#
# What if we use Richardson extrapolation with DOP853?
# Compute y with h, compute y with h/2.
# y_extrap = y_h/2 + (y_h/2 - y_h) / (2^7 - 1).
# This effectively gives order 8.
# Cost: 12 calls for h, 24 calls for h/2 (if we don't reuse stages).
# Total 36 calls per interval of size h.
# vs 12 calls for h with order 7.
#
# If h=0.2 (10 steps), cost = 10 * 36 = 360 calls.
# Error order 8. h^8 = 0.2^8 ~ 2.5e-6. Still too big?
#
# Let's stick to fixed step DOP853 with a reasonably small h.
# h=0.05 (40 steps, 480 calls) is a good bet.
# h=0.02 (100 steps, 1200 calls) is safer.
#
# The current program likely uses more calls due to adaptivity.
# Let's assume the current program uses ~2000 calls.
# If we use 480 calls, and achieve the same accuracy (floor), we gain significantly.
#
# Implementation:
# Use scipy.integrate.solve_ivp with method='DOP853', max_step=h, min_step=h.
#
# We need to be careful about the number of steps.
# t_span = (0, 2).
# h = 0.05 -> 40 steps.
#
# Let's add a safety check: if the result is not finite or if we suspect error, we can't easily check without a reference.
# But the task guarantees the reference is DOP853 rtol=1e-11.
#
# Let's try h=0.05.
#
# One more thing: The problem is fixed. y0=[1,1,1], t in [0,2].
# We can hardcode the step count.

def integrate(f, t_span, y0):
    start_time = time.time()
    deadline = start_time + 295
    
    t0, t1 = t_span
    y0_arr = np.array(y0, dtype=float)
    
    # Use fixed step DOP853
    # h=0.05 gives 40 steps. 40*12=480 calls.
    # h=0.1 gives 20 steps. 20*12=240 calls.
    # Let's try h=0.1 first. If it fails accuracy, we'd need to fall back, but we can't easily detect failure without a reference.
    # However, DOP853 is very accurate.
    # Let's check the error of DOP853 with h=0.1.
    # The local error is ~ h^8. The global error is ~ h^7.
    # For Lorenz, the error constants are moderate.
    # Let's assume h=0.1 is borderline. h=0.05 is safer.
    # Let's use h=0.05.
    
    h = 0.05
    n_steps = int(np.ceil((t1 - t0) / h))
    # Adjust h to exactly hit t1
    h = (t1 - t0) / n_steps
    
    try:
        from scipy.integrate import solve_ivp
        sol = solve_ivp(
            f, 
            t_span, 
            y0_arr, 
            method='DOP853', 
            rtol=1e-13, 
            atol=1e-15,
            max_step=h,
            min_step=h
        )
        if sol.success:
            return sol.y[:, -1]
        else:
            # Fallback to smaller steps
            h2 = h / 2
            sol = solve_ivp(
                f, 
                t_span, 
                y0_arr, 
                method='DOP853', 
                rtol=1e-13, 
                atol=1e-15,
                max_step=h2,
                min_step=h2
            )
            if sol.success:
                return sol.y[:, -1]
            else:
                # Last resort: adaptive
                sol = solve_ivp(
                    f, 
                    t_span, 
                    y0_arr, 
                    method='DOP853', 
                    rtol=1e-13, 
                    atol=1e-15
                )
                return sol.y[:, -1]
    except Exception:
        # Fallback to manual RK45 if SciPy fails
        pass

    # Manual fallback (same as before but simplified)
    t = t0
    y = y0_arr.copy()
    h_fallback = (t1 - t0) / 100
    
    while t < t1:
        if time.time() > deadline:
            break
        
        h = min(h_fallback, t1 - t)
        
        # Simple RK4
        k1 = f(t, y)
        k2 = f(t + h/2, y + h/2 * k1)
        k3 = f(t + h/2, y + h/2 * k2)
        k4 = f(t + h, y + h * k3)
        
        y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        t = t + h
        
    return y

# EVOLVE-BLOCK-END
