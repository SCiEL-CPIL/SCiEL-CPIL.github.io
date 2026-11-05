"""Rewritten program targeting the third autocorrelation inequality."""
import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

N = 400
TIME_LIMIT = 150.0  # seconds, with safety margin
START_TIME = time.time()

def objective(params):
    """Compute the objective value from parameters."""
    # params: [phase, amplitude_scale, ...] - but we need a flexible representation
    # Let's use a parameterization that allows arbitrary real values
    # We'll use a direct representation: params are the 400 heights themselves
    h = params
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e10
    conv = np.convolve(h, h)
    max_conv = np.max(conv)
    val = abs(2 * N * max_conv / (s ** 2))
    return val

def construct():
    global START_TIME
    start_time = time.time()
    
    # Deadline guard
    deadline = start_time + TIME_LIMIT - 5.0  # 5 second safety margin
    
    best_h = np.ones(N)
    best_val = objective(best_h)  # Should be 2.0
    
    # Strategy: Use differential evolution to find h that minimizes the objective
    # The objective is: |2*N*max(convolve(h,h))/sum(h)^2|
    # We want to minimize this.
    
    # Key insight: The sum in the denominator is squared, so we can normalize.
    # If we scale h by c, numerator scales by c^2, denominator by c^2, so ratio is invariant.
    # So we can fix sum(h) = 1 (or any non-zero constant) to reduce dimensionality.
    
    # Let's parameterize: h[i] = x[i] / sum(x) where x is a vector of 400 values
    # But this creates a non-smooth function due to the normalization.
    
    # Alternative: Just optimize h directly with the constraint that sum(h) != 0.
    # Since the objective is scale-invariant, we can add a constraint sum(h) = 1.
    
    # Let's try: optimize x in R^400 with constraint sum(x) = 1, then h = x
    # Actually, simpler: just optimize h directly, and the scale invariance means
    # the optimizer will naturally find good directions.
    
    # Better approach: parameterize h = z / ||z||_1 * N (so sum(h) = N)
    # This ensures sum is non-zero and fixed.
    
    # Let's use a different parameterization:
    # h[i] = tanh(x[i]) * scale, where scale is chosen so sum(h) = 1
    # This keeps values bounded.
    
    # Actually, let's just use differential_evolution directly on h with bounds.
    # The objective is scale-invariant, so we need to break the symmetry.
    # We can fix one component, e.g., h[0] = 1, and optimize the rest.
    # Or we can add a small regularization to prefer sum(h) = 1.
    
    # Let's try: optimize h in [-1, 1]^400, and compute objective.
    # The scale invariance means we might get stuck, but DE should handle it.
    
    # Actually, a cleaner approach: 
    # Let h = v / sum(v) where v is optimized, with the constraint that sum(v) > 0.
    # But this is tricky.
    
    # Simplest: just optimize h directly with DE, bounds [-10, 10].
    # The objective is well-defined as long as sum(h) != 0.
    
    # Let's define a wrapped objective that handles the sum(h) = 0 case.
    def obj_wrapped(x):
        h = x
        s = np.sum(h)
        if abs(s) < 1e-10:
            return 1e10
        conv = np.convolve(h, h)
        max_conv = np.max(conv)
        val = abs(2.0 * N * max_conv / (s * s))
        return val
    
    # Bounds for DE
    bounds = [(-10, 10)] * N
    
    # Run differential evolution with time-limited evaluation
    # DE can be slow, so let's limit the number of iterations
    
    # First, try a few random starts with local optimization
    rng = np.random.default_rng(42)
    
    # Try some structured initial guesses
    candidates = []
    
    # 1. All ones (seed)
    candidates.append(np.ones(N))
    
    # 2. Alternating signs: 1, -1, 1, -1, ...
    alt = np.ones(N)
    alt[1::2] = -1.0
    candidates.append(alt)
    
    # 3. Sinusoidal: cos(2*pi*k*x/N) for various k
    for k in [1, 2, 3, 5, 10]:
        x = np.arange(N)
        sin_h = np.cos(2 * np.pi * k * x / N)
        candidates.append(sin_h)
    
    # 4. Random Gaussian
    for _ in range(5):
        candidates.append(rng.standard_normal(N))
    
    # Evaluate all candidates
    for h in candidates:
        if time.time() > deadline:
            break
        val = obj_wrapped(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()
    
    # Now try differential evolution with limited iterations
    if time.time() < deadline - 30:  # Leave 30 seconds for polish
        try:
            # Use DE with maxiter limited by time
            maxiter = 50
            result = differential_evolution(
                obj_wrapped,
                bounds=bounds,
                maxiter=maxiter,
                tol=1e-6,
                seed=42,
                polish=True,
                update_interval=1
            )
            if result.success or result.fun < best_val:
                if time.time() < deadline:
                    # Polish with Nelder-Mead
                    res2 = minimize(
                        obj_wrapped,
                        result.x,
                        method='Nelder-Mead',
                        options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 1000}
                    )
                    if res2.fun < best_val:
                        best_val = res2.fun
                        best_h = res2.x.copy()
        except Exception:
            pass
    
    # Final polish on best_h
    if time.time() < deadline - 10:
        try:
            res_final = minimize(
                obj_wrapped,
                best_h,
                method='Nelder-Mead',
                options={'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': 5000}
            )
            if res_final.fun < best_val:
                best_val = res_final.fun
                best_h = res_final.x.copy()
        except Exception:
            pass
    
    # Ensure sum is not zero
    if abs(np.sum(best_h)) < 1e-10:
        best_h = np.ones(N)
    
    return best_h

# Keep the global N for compatibility
N = 400
