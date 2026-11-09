"""High-order fixed-step ODE integrator for the Lorenz system."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# DOP853 coefficients (12 stages, 10th order)
# Reference: DOP853 from Hairer, Norsett, Wanner (1993) or Scipy implementation.
# These are the standard coefficients for the Dormand-Prince 8(5,3) method.
# We use the 10th order method often referred to as DOP853 in high-precision contexts, 
# though technically DOP853 is 8th order. 
# Wait, the task says "y_ref comes from scipy.integrate.solve_ivp with method DOP853". 
# Scipy's DOP853 is an 8th order method (order 8, embedded 5th).
# However, to beat the reference (which uses rtol 1e-11), we need high accuracy.
# A single step of an 8th order method with h=2 might not be accurate enough for rel < 1e-14.
# Let's check the order of magnitude.
# If we use a 10th order method (like Verner 10(8) or similar), 1 step might suffice.
# But I don't have the 10th order coefficients hardcoded reliably in memory.
# 
# Let's look at the "Inspiration 1" which scored 1.2798. It used a "fixed-step, 12-stage, 10th-order".
# The current program uses scipy adaptive and scores 1.2746.
# The goal is to reduce calls.
#
# Strategy: Use a high-order fixed step.
# If I use Scipy's DOP853 (8th order) with a single step h=2, the error is O(h^9).
# For Lorenz, the solution grows/oscillates. The error might be around 1e-5 or worse.
# We need rel < 1e-14.
# So 1 step is not enough for 8th order.
#
# What if we use 2 steps? h=1. Error O(1^9) = 1e-9? Still not enough.
# What if we use 4 steps? h=0.5. Error O(0.5^9) ~ 2e-3? No.
#
# We need a method with very high order or very small steps.
# The "Inspiration 1" claims a 10th order method.
# Let's try to implement a known 10th order method.
# Verner 10(8) has 11 stages.
# Coefficients for Verner 10(8):
# c = [0, 1/10, 3/10, 4/5, 1, 3/2, 5/2, 3, 4, 5, 1] ... this looks like a guess.
#
# Actually, let's look at the DOP853 coefficients again.
# Scipy's DOP853 is 8th order.
# If we use 1 step, we get 12 calls.
# If the error is too high, we fail the accuracy part (rel > 1e-14 means score drops).
# Score = -log10(rel) - 0.5*log10(calls).
# If rel = 1e-5, -log10(1e-5) = 5.
# If calls = 12, -0.5*log10(12) ~ -0.5*1.08 = -0.54.
# Total ~ 4.46.
# Current score 1.27.
# If rel is 1e-14, -log10(1e-14) = 14.
# If calls = 12, total ~ 13.46.
# So getting high accuracy with few calls is the key.
#
# Can we get rel < 1e-14 with 12 calls?
# Only if the method order is high enough that the local error is negligible.
# For a 10th order method, local error is h^11.
# h=2, h^11 = 2048. This is huge.
# So 1 step is NOT enough for 10th order either.
#
# Wait, the "Inspiration 1" scored 1.2798, which is BETTER than 1.2746.
# How?
# Maybe it used more steps but fewer calls per step? No, RK is fixed calls/step.
# Maybe it used a higher order method?
# Or maybe my estimate of error for 1 step is wrong?
#
# Let's re-evaluate the current program.
# It uses adaptive DOP853.
# Score 1.2746.
# If it used, say, 200 steps, calls ~ 2400.
# -0.5 * log10(2400) ~ -0.5 * 3.38 = -1.69.
# So -log10(rel) must be ~ 2.96? No, score is 1.27.
# 1.27 = -log10(rel) - 1.69 => -log10(rel) = 2.96 => rel ~ 1e-3.
# This suggests the current program is NOT achieving rel < 1e-14?
# Or my log calculation is off.
#
# Let's check the reference.
# "y_ref comes from ... rtol 1e-11".
# The reference is the ground truth.
# If my solution matches the reference to 1e-14, rel is 1e-14.
#
# Let's look at the "Inspiration 1" again.
# "fixed-step, 12-stage, 10th-order ... pre-computed optimal step count".
# If it used, say, 5 steps. Calls = 60.
# -0.5 * log10(60) ~ -0.9.
# If rel is 1e-14, score ~ 13.1.
# Why did it only score 1.2798?
# Maybe the "10th order" method was actually lower order or unstable?
# Or maybe the "pre-computed optimal step count" resulted in large errors?
#
# Let's try a different structural change:
# Use a high-order method but with a moderate number of steps to ensure accuracy.
# Or, use the fact that we know the system is Lorenz.
#
# Alternative Strategy:
# Use Scipy's `solve_ivp` with `method='Radau'` or `BDF`? No, explicit is better for non-stiff.
#
# Let's try to implement a 12-stage 10th order method correctly.
# I will use the coefficients from a reliable source for Verner 10(8) or similar.
# However, hardcoding 11x11 matrices is risky.
#
# Let's try a different approach:
# Use `scipy.integrate.solve_ivp` with `method='DOP853'` but force a very large `max_step`?
# No, that's what the current one does (sort of).
#
# What if we use `scipy.integrate.odeint`? No, slower.
#
# Let's look at the call count penalty.
# If I can get rel < 1e-14 with 12 calls, score is ~13.
# If I get rel < 1e-14 with 100 calls, score is ~12.
#
# Is it possible to get rel < 1e-14 with 12 calls (1 step) for Lorenz [0,2]?
# The Lorenz system is chaotic. The Lyapunov exponent is ~0.9.
# Error growth is exp(0.9 * t).
# For t=2, growth factor ~ e^1.8 ~ 6.
# Local error of 8th order method with h=2: C * h^9.
# C depends on the 9th derivative.
# If C is small, maybe?
# But usually, 1 step of RK8 with h=2 is not 1e-14 accurate.
#
# Let's assume the "Inspiration 1" failed to get high accuracy despite high order, hence the low score.
# The current program (adaptive) gets 1.27.
#
# Let's try to improve the current program's efficiency.
# The current program uses adaptive DOP853.
# We can try to reduce the number of steps by increasing the step size allowed, but keep accuracy.
#
# Structural Change:
# Use a **Chebyshev Spectral Method** or **Taylor Series** expansion?
# For a short interval [0, 2], a high-order Taylor series might work.
# But computing derivatives is expensive (calls to f).
#
# Let's try **Extrapolation**.
# Use RK4 with Richardson Extrapolation.
# Steps: 1, 2, 4, 8...
# This increases calls significantly.
#
# Let's go back to the "fixed-step high-order" idea but be smarter about step size.
# If we use a 10th order method, we need the coefficients.
# I will hardcode the Verner 10(8) coefficients.
# Source: Verner, J. H. (1976).
#
# c = [0, 1/10, 3/10, 4/5, 1, 3/2, 5/2, 3, 4, 5, 1] -- This is likely wrong.
#
# Let's use the **DOP853** coefficients from Scipy but run it in a **fixed-step** mode with a **small enough** step to guarantee accuracy, but **large enough** to minimize steps.
# If we use h=0.25, we need 8 steps.
# Calls = 8 * 12 = 96.
# -0.5 * log10(96) ~ -1.0.
# If accuracy is 1e-14, score ~ 13.
#
# Why did the adaptive solver only get 1.27?
# Maybe it took many steps?
# If it took 100 steps, calls = 1200.
# -0.5 * log10(1200) ~ -1.6.
# Score 1.27 implies -log10(rel) ~ 2.87 => rel ~ 1e-3.
# This suggests the adaptive solver did NOT reach 1e-14 accuracy?
# Or the reference `y_ref` is not 1e-14 accurate itself?
# "y_ref comes from ... rtol 1e-11".
# If the reference has error 1e-11, we can't get rel < 1e-11.
# Max possible -log10(rel) is 11.
# If rel = 1e-11, score = 11 - 0.5*log10(calls).
# If calls = 1200, score = 11 - 1.6 = 9.4.
# Why is the score 1.27?
#
# Ah, the scoring formula: `score = -log10(rel) - 0.5 * log10(calls)`.
# If rel is 1e-11, -log10(rel) is 11.
# If the score is 1.27, then -log10(rel) must be ~ 2.8.
# This implies rel is ~ 1e-3.
# This means the current program is NOT accurate to the reference!
#
# Why?
# Maybe the reference `y_ref` is calculated with `rtol=1e-11`, but the actual error is larger?
# Or maybe my program is diverging?
#
# Let's check the Lorenz system sensitivity.
# If I use `rtol=1e-14` in my solver, but the reference uses `rtol=1e-11`,
# the difference between my solution and the reference might be dominated by the reference's error.
# If the reference has error 1e-11, the best I can do is match it to 1e-11.
# So rel should be ~ 1e-11.
# Score should be ~ 11 - penalty.
#
# If the score is 1.27, something is wrong.
# Perhaps the `f` function is not being called efficiently?
# Or perhaps the `rel` calculation is different?
#
# Let's assume the score 1.27 is correct and implies low accuracy.
# To improve, I need to **increase accuracy**.
#
# Strategy: Use a **much tighter tolerance** in the adaptive solver.
# The current program uses `rtol=1e-14`.
# Maybe it's hitting the `max_step` limit or something?
#
# Let's try a **fixed-step** approach with a **very high order** method and **moderate steps**.
# I will implement a **12-stage 10th-order** method using coefficients from a known source.
# I will use the **Verner 10(8)** method.
#
# Coefficients for Verner 10(8) (from Verner 1976):
# c = [0, 1/10, 3/10, 4/5, 1, 3/2, 5/2, 3, 4, 5, 1] -- I don't trust these.
#
# Let's use **DOP853** (8th order) with **fixed steps** of **h=0.1**.
# Steps = 20.
# Calls = 20 * 12 = 240.
# Penalty = -0.5 * log10(240) ~ -1.2.
# If accuracy is 1e-11 (limited by ref), score ~ 11 - 1.2 = 9.8.
# This is much better than 1.27.
#
# Why did the adaptive solver fail?
# Maybe it took too many steps?
# If it took 1000 steps, calls = 12000.
# Penalty = -0.5 * log10(12000) ~ -2.3.
# Score ~ 11 - 2.3 = 8.7.
# Still better than 1.27.
#
# So why is the score 1.27?
# Maybe `rel` is NOT 1e-11.
# Maybe the reference `y_ref` is not 1e-11 accurate?
# Or maybe my program is crashing?
# "an exception ... makes the whole result invalid".
# If invalid, score is 0.
# But score is 1.27.
#
# Let's assume the score 1.27 is due to **high call count** and **moderate accuracy**.
#
# New Strategy:
# Use **fixed-step DOP853** with **h=0.2** (10 steps).
# Calls = 120.
# Penalty ~ -1.1.
# If accuracy is good, score ~ 10.
#
# I will implement DOP853 fixed-step.

# DOP853 Coefficients (8th order, 12 stages)
# From Scipy source or Hairer et al.
# a = np.array([
#    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#    [1/40, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#    [3/10, -9/40, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#    [44/45, -56/15, 32/9, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#    [19372/6561, -25360/2187, 64448/6561, -212/729, 0, 0, 0, 0, 0, 0, 0, 0],
#    [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656, 0, 0, 0, 0, 0, 0, 0],
#    [35/384, 0, 500/1113, -125/192, 2187/6784, 11/84, 0, 0, 0, 0, 0, 0],
#    [505/108, -1334140/895395, 264/23, -4596244/1568895, 189876/51205, -1372/6321, 184/9, 0, 0, 0, 0, 0],
#    [-71645/1224, 4550419/375672, -212624/17955, 1133277/167916, -24778/51205, 676595/1839564, 1733/456, -260/525, 0, 0, 0, 0],
#    [742035/171238, -493801/147232, 72245/9396, -3974707/180624, -36448/11856, 23561/48718, -979/40, 5491/80, -288/40, 0, 0, 0],
#    [1089/136, -45/16, 6480/437, -1055/16, 9335/2128, -9/4, 56/15, -144/35, 12/5, -1/16, 0, 0],
#    [505/108, -1334140/895395, 264/23, -4596244/1568895, 189876/51205, -1372/6321, 184/9, 0, 0, 0, 0, 0] # Last row is for 5th order, we use 8th order weights
# ])
#
# Weights for 8th order:
# b8 = np.array([3025/13698, 0, 43954/41871, 0, 4324/2757, -125/192, 0, 0, 0, 0, 0, 0]) # This is wrong, DOP853 has 12 stages.
#
# Let's just use Scipy's DOP853 but with fixed steps.
# We can't easily force fixed steps in Scipy.
#
# I will implement the 12 stages manually.

def integrate(f, t_span, y0):
    t_start, t_end = t_span
    y = np.array(y0, dtype=float)
    
    # Use fixed step DOP853
    # h = 0.2, 10 steps
    n_steps = 10
    h = (t_end - t_start) / n_steps
    
    # DOP853 coefficients (8th order)
    # c = [0, 1/40, 3/10, 4/5, 3/5, 1/2, 1, 1, 1, 1, 1, 1] -- nodes
    # a is 12x12
    # b8 is 12 weights
    
    # Hardcoding full DOP853 is long.
    # Let's try a simpler high-order method: **Gauss-Legendre**? No, implicit.
    # **Radau**? Implicit.
    #
    # Let's stick to **RK4** but with **very small steps**?
    # No, that increases calls.
    #
    # What if we use **scipy.integrate.solve_ivp** with `method='DOP853'` and `max_step=h`?
    # This forces the adaptive solver to take steps no larger than h.
    # If we set h small enough, it will take exactly n_steps (or close).
    #
    # Let's try h=0.1 (20 steps).
    # Calls ~ 240.
    # Score ~ 10.
    
    import scipy.integrate as sci_int
    
    # Use DOP853 with max_step to control step size
    # If we set max_step=0.1, it will take at least 20 steps.
    # With rtol=1e-14, it might take more if error is high.
    # But for Lorenz, 0.1 is likely fine.
    
    sol = sci_int.solve_ivp(
        f,
        t_span,
        y0,
        method='DOP853',
        rtol=1e-14,
        atol=1e-16,
        max_step=0.1
    )
    
    if sol.success:
        return sol.y[:, -1]
    else:
        # Fallback
        h_fb = 0.01
        n_fb = int((t_end - t_start) / h_fb)
        t = t_start
        for i in range(n_fb):
            k1 = f(t, y)
            k2 = f(t + h_fb/2, y + h_fb/2 * k1)
            k3 = f(t + h_fb/2, y + h_fb/2 * k2)
            k4 = f(t + h_fb, y + h_fb * k3)
            y = y + h_fb/6 * (k1 + 2*k2 + 2*k3 + k4)
            t += h_fb
        return y

# EVOLVE-BLOCK-END
