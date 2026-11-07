import numpy as np
import scipy.optimize
import time

N = 50
DEADLINE = 140.0  # Safety margin before 150 s evaluator kill

_start_time = time.perf_counter()

def _time_left():
    return DEADLINE - (time.perf_counter() - _start_time)

def compute_value(h):
    """
    Computes the value exactly as defined in the task contract.
    Vectorized for speed.
    """
    h = np.asarray(h, dtype=np.float64)
    if h.size != N:
        return 0.0
    if not np.all(np.isfinite(h)):
        return 0.0
    if np.any(h < 0):
        # Clip to non-negative to be safe, though optimizer should handle bounds
        h = np.maximum(h, 0.0)
    
    # Convolution
    g = np.convolve(h, h, mode='full')
    m = len(g)
    
    # y = [0, g[0], ..., g[m-1], 0]
    # y has length m + 2
    y = np.zeros(m + 2)
    y[1:-1] = g
    
    # Spacing w
    # Points are np.linspace(-0.5, 0.5, m + 2)
    # Number of intervals is m + 1
    w = 1.0 / (m + 1)
    
    # l2 = sum over i in 0..m of (w / 3) * (y[i]^2 + y[i] * y[i+1] + y[i+1]^2)
    # y indices: 0 to m+1. i goes from 0 to m.
    # So we sum over pairs (y[i], y[i+1]) for i=0..m.
    y_i = y[:m+1]
    y_ip1 = y[1:m+2]
    
    # Term: y_i^2 + y_i*y_ip1 + y_ip1^2
    terms = y_i**2 + y_i * y_ip1 + y_ip1**2
    l2 = (w / 3.0) * np.sum(terms)
    
    # Norms
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    
    if norm_1 <= 0 or norm_inf <= 0:
        return 0.0
        
    return l2 / (norm_1 * norm_inf)

def _objective(h):
    """
    Objective function for minimization.
    Includes normalization to max=1 to handle scale invariance.
    """
    h = np.maximum(h, 0.0)
    max_h = np.max(h)
    if max_h == 0:
        return 0.0
    h_norm = h / max_h
    val = compute_value(h_norm)
    return -val

def _initial_candidates():
    """
    Generate diverse initial shapes.
    """
    x = np.linspace(0, 1, N)
    cands = []
    
    # 1. Uniform
    cands.append(np.ones(N))
    
    # 2. Triangular
    cands.append(1.0 - 2.0 * np.abs(x - 0.5))
    
    # 3. Gaussian (wide)
    cands.append(np.exp(-((x - 0.5) ** 2) / (2 * 0.25 ** 2)))
    
    # 4. Gaussian (narrow)
    cands.append(np.exp(-((x - 0.5) ** 2) / (2 * 0.1 ** 2)))
    
    # 5. Box (wide)
    box_w = np.zeros(N)
    box_w[10:40] = 1.0
    cands.append(box_w)
    
    # 6. Box (narrow)
    box_n = np.zeros(N)
    box_n[20:30] = 1.0
    cands.append(box_n)
    
    # 7. Bimodal
    cands.append(np.exp(-((x - 0.3) ** 2) / (2 * 0.1 ** 2)) + np.exp(-((x - 0.7) ** 2) / (2 * 0.1 ** 2)))
    
    # 8. Raised Cosine
    cands.append(0.5 * (1 + np.cos(2 * np.pi * x)))
    
    # 9. Power law
    cands.append((1.0 - 2.0 * np.abs(x - 0.5)) ** 2)
    
    # 10. Random smooth (seeded)
    rng = np.random.RandomState(123)
    noise = rng.rand(N)
    # Smooth it a bit
    kernel = np.ones(5) / 5.0
    smooth_noise = np.convolve(noise, kernel, mode='same')
    cands.append(smooth_noise)
    
    # 11. Edge peak
    cands.append(np.exp(-x / 0.2) + np.exp(-(1 - x) / 0.2))
    
    # 12. Sine wave
    cands.append(0.5 + 0.5 * np.sin(2 * np.pi * 2 * x))
    
    # 13. Asymmetric
    cands.append(np.exp(-((x - 0.4) ** 2) / (2 * 0.15 ** 2)))
    
    # Normalize all to max 1
    for i, c in enumerate(cands):
        c = np.maximum(c, 0.0)
        mx = np.max(c)
        if mx > 0:
            c = c / mx
        cands[i] = c
        
    return cands

def construct():
    global _start_time
    _start_time = time.perf_counter()
    
    best_h = None
    best_val = -np.inf
    
    candidates = _initial_candidates()
    
    # First pass: L-BFGS-B on all candidates
    for i, h0 in enumerate(candidates):
        if _time_left() < 10.0:
            break
            
        try:
            # L-BFGS-B is good for this scale.
            # Bounds: 0 to 1 (since we normalize inside objective, 
            # but providing bounds helps the optimizer stay in a reasonable range 
            # before normalization, although normalization makes it scale invariant.
            # Actually, if we normalize inside objective, the gradient wrt raw h 
            # is complicated. It's better to optimize h directly with bounds [0, 1] 
            # and NOT normalize inside objective, but the ratio is scale invariant.
            # Wait, if we don't normalize, the optimizer might drift to very small or large values.
            # Let's stick to optimizing h in [0, 1] and computing value directly.
            # The value is scale invariant, so the optimizer will find a shape.
            # However, L-BFGS-B needs a well-scaled objective.
            # Let's use the normalized objective but pass h in [0,1] to the optimizer?
            # No, if we normalize inside objective, the gradient is zero at the optimum 
            # if we just scale? No, the shape matters.
            
            # Let's define a wrapper that takes h, clips to [0,1], and computes -value.
            # Since value is scale invariant, we can fix max(h)=1.
            # To do this properly with L-BFGS-B, we can optimize 49 parameters 
            # (ratios to the max) or just optimize h and let it find the scale.
            # Optimizing h directly is simpler.
            
            def neg_val(h_raw):
                h_clip = np.clip(h_raw, 0, 1)
                # If all zero, return 0
                if np.max(h_clip) == 0:
                    return 0.0
                return -compute_value(h_clip)
                
            res = scipy.optimize.minimize(
                neg_val, h0, method='L-BFGS-B',
                bounds=[(0, 1)] * N,
                options={'maxiter': 500, 'maxfun': 10000, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            h_opt = np.clip(res.x, 0, 1)
            val = -res.fun
            
            if val > best_val:
                best_val = val
                best_h = h_opt.copy()
                
        except Exception:
            continue
            
    # Second pass: Polish the best few
    if best_h is not None and _time_left() > 15.0:
        # Run L-BFGS-B again on the best result with tighter tolerances
        try:
            def neg_val_tight(h_raw):
                h_clip = np.clip(h_raw, 0, 1)
                if np.max(h_clip) == 0:
                    return 0.0
                return -compute_value(h_clip)
                
            res2 = scipy.optimize.minimize(
                neg_val_tight, best_h, method='L-BFGS-B',
                bounds=[(0, 1)] * N,
                options={'maxiter': 1000, 'maxfun': 20000, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            
            h_opt2 = np.clip(res2.x, 0, 1)
            val2 = -res2.fun
            
            if val2 > best_val:
                best_val = val2
                best_h = h_opt2.copy()
                
        except Exception:
            pass
            
    # Ensure valid output
    if best_h is None:
        # Fallback to uniform
        best_h = np.ones(N)
        
    return list(best_h)
