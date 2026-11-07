"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    dt = t1 - t0
    
    # Deadline guard: start with a generous margin
    deadline = time.time() + 280  # 280s out of 300s limit
    
    # Strategy: Use Richardson extrapolation with RK4.
    # Start with a coarse step, refine until error is small or time runs out.
    # RK4 has global error O(h^4).
    
    # We want to minimize calls while maximizing accuracy.
    # rel = max(||y_end - y_ref|| / max(||y_ref||, 1e-12), 1e-14)
    # score = -log10(rel) - 0.5 * log10(calls)
    
    # Let's estimate the reference solution magnitude.
    # For harmonic oscillator y0=[1,0], t in [0,10], y stays in unit circle.
    # So ||y_ref|| ~ 1.
    
    # We will perform a sequence of RK4 integrations with increasing steps.
    # Steps: N, 2N, 4N, ...
    # Extrapolate to h=0 using Richardson extrapolation.
    
    # Start with N=100 (h=0.1). Calls = 4*100 = 400.
    # Then N=200, 400, 800, 1600, 3200, 6400...
    
    # However, we can be smarter. We know the problem is smooth and non-stiff.
    # Let's use a fixed high-accuracy approach if we have time, or adaptive.
    
    # Let's try to hit rel ~ 1e-14 with minimal calls.
    # If we use N=1000, calls=4000. rel might be ~1e-10?
    # If we use N=10000, calls=40000. rel might be ~1e-14?
    
    # Let's implement a loop that doubles N until we have two consecutive results
    # that agree to within 1e-15, or until we run out of time.
    
    # Actually, simpler: Just run RK4 with a very high N if we have time.
    # But we want to minimize calls.
    
    # Let's try N=200, 400, 800, 1600, 3200.
    # Store results.
    
    results = []
    Ns = []
    
    N = 50
    # Ensure N is at least 1
    N = max(1, N)
    
    while True:
        h = dt / N
        t = t0
        y_curr = y.copy()
        
        # Check time before starting this block
        if time.time() > deadline - 5:
            break
            
        for _ in range(N):
            k1 = f(t, y_curr)
            k2 = f(t + h/2, y_curr + h/2 * k1)
            k3 = f(t + h/2, y_curr + h/2 * k2)
            k4 = f(t + h, y_curr + h * k3)
            y_curr = y_curr + h/6 * (k1 + 2*k2 + 2*k3 + k4)
            t += h
            
            # Early exit if time is tight?
            # No, let's finish the step.
            
        results.append(y_curr)
        Ns.append(N)
        
        # If we have at least 2 results, check convergence
        if len(results) >= 2:
            y_prev = results[-2]
            y_curr = results[-1]
            diff = np.linalg.norm(y_curr - y_prev)
            # If difference is very small, we are converged
            if diff < 1e-15:
                break
                
        # Increase N
        N *= 2
        
        # Safety: don't go too high
        if N > 100000:
            break
            
    # If we have results, use the last one (highest accuracy)
    # Or use Richardson extrapolation if we have >= 2
    if len(results) >= 2:
        # Extrapolate using last two
        y1 = results[-2]
        y2 = results[-1]
        # h1 = dt/Ns[-2], h2 = dt/Ns[-1]
        # y_true = y2 + (y2 - y1) / (4^4 - 1) ?
        # Error term: y(h) = y_true + C*h^4
        # y1 = y_true + C*h1^4
        # y2 = y_true + C*h2^4
        # h1 = 2*h2
        # y1 = y_true + 16*C*h2^4
        # y2 - y1 = C*h2^4 - 16*C*h2^4 = -15*C*h2^4
        # C*h2^4 = (y2 - y1) / -15
        # y_true = y2 - C*h2^4 = y2 - (y2 - y1)/-15 = y2 + (y2 - y1)/15
        y_extrap = y2 + (y2 - y1) / 15.0
        return y_extrap
    elif len(results) == 1:
        return results[0]
    else:
        # Fallback: single step or return y0?
        # Must call f at least once.
        # If loop didn't run, do 1 step
        h = dt
        k1 = f(t0, y)
        k2 = f(t0 + h/2, y + h/2 * k1)
        k3 = f(t0 + h/2, y + h/2 * k2)
        k4 = f(t0 + h, y + h * k3)
        y_end = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        return y_end

# EVOLVE-BLOCK-END
