import numpy as np
import time

N = 600

def _score_heights(h):
    """
    Compute the normalized maximum of the autoconvolution of the step function h.
    The step function is defined on [-1/4, 1/4] with N equal intervals.
    The autoconvolution is computed on the grid of interval centers or edges.
    The normalization is likely by the L2 norm or the maximum of the convolution relative to some reference.
    
    Based on the task: "minimize the normalized maximum of its autoconvolution; this gives an upper bound on C1."
    The constant C1 in the first autocorrelation inequality is typically related to the L2 norm.
    A common normalization is ||h * h||_inf / ||h||_2^2 or similar.
    
    Let's assume the score is max(autoconv) / (sum(h)^2 / N) or max(autoconv) / ||h||_1^2?
    Actually, for the autocorrelation inequality, the constant C1 is often defined such that
    ||f * f||_inf <= C1 ||f||_2^2.
    So the score should be max(autoconv) / ||h||_2^2.
    
    Let's verify with the seed program [1.0]*N.
    h = [1, 1, ..., 1]
    ||h||_2^2 = sum(1^2) = N = 600.
    The autoconvolution of a rectangular pulse of width N (in discrete steps) is a triangle.
    The maximum is at the center, value = N * 1 * 1 = N = 600? 
    Wait, discrete convolution of [1]*N with [1]*N has max value N.
    So score = 600 / 600 = 1.0.
    This matches the seed score of 1.0.
    
    So the score function is: max(np.convolve(h, h)) / np.sum(h**2).
    """
    h = np.asarray(h, dtype=np.float64)
    autoconv = np.convolve(h, h, mode='full')
    max_ac = np.max(autoconv)
    l2_sq = np.sum(h**2)
    if l2_sq < 1e-12:
        return np.inf
    return max_ac / l2_sq

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    best_h = np.ones(N)
    best_score = _score_heights(best_h)  # Should be 1.0
    
    # Strategy: Use a parameterized family of functions to reduce dimensionality.
    # Family: h[i] = max(0, A - B * |i - center|) ** p  (Tent-like with power)
    # Or simply: h[i] = (1 - (i-center)^2 / R^2)^p for |i-center| < R, else 0.
    # This ensures non-negativity.
    
    # Let's try optimizing a Gaussian-like bump: h[i] = exp(-alpha * (i - center)^2)
    # But step functions with 600 intervals.
    # We can optimize 'alpha' and 'center' (though center is fixed at 299.5 by symmetry).
    # Actually, the function doesn't have to be symmetric, but symmetry likely helps minimize max autocorrelation.
    
    # Let's try a few different shapes:
    # 1. Gaussian: exp(-a * x^2)
    # 2. Power-law: (1 - |x|^p)^q
    # 3. Cosine bump: cos(pi * x / 2)^2
    
    # We will perform a simple grid search + local refinement on parameters.
    
    x = np.arange(N) - (N - 1) / 2.0
    
    # Parameter search for Gaussian
    # h[i] = exp(-a * x[i]^2)
    # We want to minimize max(conv(h,h)) / sum(h^2)
    
    # Let's sample a range of 'a'
    # If a is small, h is wide and flat -> approximates constant -> score ~ 1.0
    # If a is large, h is narrow -> spike -> score ~ 1.0 (narrow spike autocorr is narrow spike, ratio ~ 1)
    # Wait, for a delta function (approx), conv is delta, max=1, l2=1, ratio=1.
    # For a constant, conv is triangle, max=N, l2=N, ratio=1.
    # Is it possible to get < 1.0?
    # Yes, if the shape is such that the autocorrelation is flatter relative to the L2 norm.
    # The constant function gives 1. The delta gives 1.
    # What about a "top-hat" with rounded corners? Or a specific window?
    
    # Let's test a few specific shapes to see if any beat 1.0.
    
    candidates = []
    
    # 1. Constant (seed)
    candidates.append(np.ones(N))
    
    # 2. Gaussian with various widths
    for a in np.linspace(0.01, 20.0, 50):
        h = np.exp(-a * x**2)
        # Normalize? No, the score is scale-invariant.
        candidates.append(h)
    
    # 3. Cosine bump
    for w in np.linspace(0.1, 1.0, 20): # width fraction
        R = w * N / 2
        h = np.zeros(N)
        mask = np.abs(x) < R
        h[mask] = np.cos(np.pi * x[mask] / (2 * R))**2
        candidates.append(h)
    
    # 4. Power law bump
    for p in [1.0, 2.0, 3.0, 5.0]:
        for w in np.linspace(0.1, 1.0, 10):
            R = w * N / 2
            h = np.zeros(N)
            mask = np.abs(x) < R
            h[mask] = (1 - (x[mask] / R)**2)**p
            candidates.append(h)
            
    # 5. Random perturbations of constant (to check if local minima exist)
    # But random is risky. Let's stick to structured families.
    
    # 6. "Sinc-like" or oscillating? Non-negative required.
    # h = (1 + cos(k*x)) * envelope
    
    # Let's evaluate all candidates
    best_local_h = candidates[0]
    best_local_score = _score_heights(best_local_h)
    
    for h in candidates:
        if time.time() > deadline:
            break
        s = _score_heights(h)
        if s < best_local_score:
            best_local_score = s
            best_local_h = h.copy()
    
    # If we found something better than 1.0, try to refine it using gradient descent or coordinate descent.
    # Since we don't have a general gradient, we can do a simple hill-climbing on the parameters of the best family.
    # However, the time budget is tight.
    
    # Let's do a more focused search on the Gaussian family, as it's smooth and likely optimal.
    # Minimize f(a) = score(exp(-a*x^2))
    
    # We can use scipy.optimize.minimize_scalar on the parameter 'a'.
    try:
        from scipy.optimize import minimize_scalar
        
        def obj(a):
            h = np.exp(-a * x**2)
            return _score_heights(h)
            
        # Search in a reasonable range
        res = minimize_scalar(obj, bounds=(0.01, 50.0), method='bounded', options={'xatol': 1e-6})
        
        if res.success and res.fun < best_local_score:
            best_local_score = res.fun
            best_local_h = np.exp(-res.x * x**2)
            
    except Exception:
        pass
    
    # Also try refining the Cosine bump width
    try:
        from scipy.optimize import minimize_scalar
        
        def obj_cos(w):
            R = w * N / 2
            h = np.zeros(N)
            mask = np.abs(x) < R
            h[mask] = np.cos(np.pi * x[mask] / (2 * R))**2
            return _score_heights(h)
            
        res = minimize_scalar(obj_cos, bounds=(0.01, 0.99), method='bounded', options={'xatol': 1e-6})
        
        if res.success and res.fun < best_local_score:
            best_local_score = res.fun
            R = res.x * N / 2
            best_local_h = np.zeros(N)
            mask = np.abs(x) < R
            best_local_h[mask] = np.cos(np.pi * x[mask] / (2 * R))**2
            
    except Exception:
        pass
        
    # Final check
    if best_local_score < best_score:
        best_h = best_local_h
        best_score = best_local_score
        
    # Ensure non-negative
    best_h = np.maximum(best_h, 0)
    
    # If time remains, try a few random local perturbations of the best found shape
    if time.time() < deadline - 10:
        current_h = best_h.copy()
        current_s = best_score
        for _ in range(50):
            if time.time() > deadline - 5:
                break
            # Perturb
            noise = np.random.normal(0, 0.1, N)
            h_new = current_h + noise
            h_new = np.maximum(h_new, 0)
            s_new = _score_heights(h_new)
            if s_new < current_s:
                current_s = s_new
                current_h = h_new
        if current_s < best_score:
            best_h = current_h
            best_score = current_s

    return list(best_h)

# EVOLVE-BLOCK-END
