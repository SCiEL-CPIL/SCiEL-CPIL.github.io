import time
import numpy as np

N = 600

def compute_score(h):
    """
    Compute the normalized maximum of the autoconvolution.
    h: array of nonnegative values.
    Returns max_k (h*h)[k] / (sum h)^2
    """
    s = np.sum(h)
    if s == 0:
        return 0.0
    conv = np.convolve(h, h, mode='full')
    max_conv = np.max(conv)
    return max_conv / (s * s)

# EVOLVE-BLOCK-START
import scipy.optimize as so

def _top_hat_tapered(params):
    """
    Generate a nonnegative step function with a top-hat core and tapered edges.
    params: (width_frac, taper_power, center_frac)
      width_frac: fraction of [0, N) that is the flat top (0 < w < 1)
      taper_power: exponent for the raised-cosine taper (1 = cosine, 2 = smoother)
      center_frac: center of the window as a fraction of N (0 to 1)
    Returns: array of length N with nonnegative values.
    """
    width_frac, taper_power, center_frac = params
    if width_frac <= 0 or width_frac >= 1:
        return np.ones(N) / N
    if taper_power <= 0:
        taper_power = 1.0
    
    # The window occupies [center_frac - width_frac/2, center_frac + width_frac/2]
    # in normalized coordinates [0, 1)
    lo = center_frac - width_frac / 2.0
    hi = center_frac + width_frac / 2.0
    
    # Indices in [0, N)
    idx = np.arange(N, dtype=float)
    x = idx / N  # normalized position in [0, 1)
    
    h = np.zeros(N)
    
    # Determine which indices fall within [lo, hi]
    # Wrap around if needed (torus-like)
    # For simplicity, handle the case where the window fits in [0, N)
    # If it wraps, we'll handle it by checking both ends
    
    # Mask for indices in the flat-top region
    # Flat top: |x - center_frac| <= width_frac / 2 - taper_width
    # Taper region: width_frac/2 - taper_width < |x - center_frac| <= width_frac/2
    
    # Let's define: the taper takes up some fraction of the half-width
    # Actually, let's use a simpler model:
    # h(x) = 1 for |x - c| <= w/2 - t
    # h(x) = 0.5 * (1 + cos(pi * (|x-c| - (w/2-t)) / t * taper_power)) for (w/2-t) < |x-c| <= w/2
    # h(x) = 0 for |x-c| > w/2
    
    # But this is getting complex. Let me use a simpler super-ellipse approach:
    # h(x) = (1 - ((|x-c|)/(w/2))^(2*p))^(1/(2*p)) for |x-c| < w/2, else 0
    # where p is the taper power (p=1 gives a diamond, p=inf gives a rectangle)
    
    # Actually, let me use a raised cosine window:
    # h(x) = 0.5 * (1 + cos(pi * (|x-c| - w/2) / (w/2))) for |x-c| < w/2, else 0
    # This is the standard Hann window shape.
    
    # Let me use a more flexible family:
    # h(x) = (1 - u^(2*p))^(1/(2*p)) where u = |x-c|/(w/2), for u < 1
    # p=1: h = sqrt(1-u^2) (semicircle)
    # p=2: h = (1-u^4)^(1/4) (more rectangular)
    # p->inf: h -> 1 (rectangle)
    
    u = np.abs(x - center_frac) / (width_frac / 2.0)
    mask = u < 1.0
    
    # Also handle wrap-around
    u_wrap = np.minimum(u, np.abs(x - center_frac + 1.0) / (width_frac / 2.0))
    u_wrap2 = np.minimum(u, np.abs(x - center_frac - 1.0) / (width_frac / 2.0))
    u_final = np.minimum(u, np.minimum(u_wrap, u_wrap2))
    mask = u_final < 1.0
    
    if not np.any(mask):
        return np.ones(N) / N
    
    # Use the super-ellipse formula
    # Clip u_final to avoid numerical issues
    u_clipped = np.clip(u_final[mask], 0, 0.999999)
    
    # h = (1 - u^(2*p))^(1/(2*p))
    # For stability, use log form when p is large
    if taper_power > 50:
        # Approximately rectangular with smooth edges
        h[mask] = 1.0
        # Apply a small taper at the edges
        edge_mask = u_final[mask] > 0.95
        if np.any(edge_mask):
            u_edge = u_final[mask][edge_mask]
            h[mask][edge_mask] = 0.5 * (1 + np.cos(np.pi * (u_edge - 0.95) / 0.05))
    else:
        h[mask] = (1.0 - u_clipped ** (2 * taper_power)) ** (1.0 / (2 * taper_power))
    
    # Ensure nonnegative
    h = np.maximum(h, 0)
    
    # Normalize
    s = np.sum(h)
    if s > 0:
        h = h / s
    
    return h

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_h = np.ones(N) / N
    best_score = compute_score(best_h)
    
    # Optimize the parametric family using Nelder-Mead
    def param_objective(params):
        h = _top_hat_tapered(params)
        return compute_score(h)
    
    # Initial guesses for (width_frac, taper_power, center_frac)
    # Try several starting points
    initial_params = [
        [0.8, 1.0, 0.5],
        [0.6, 1.5, 0.5],
        [0.7, 2.0, 0.5],
        [0.9, 3.0, 0.5],
        [0.5, 1.0, 0.5],
        [0.8, 1.0, 0.3],
        [0.8, 1.0, 0.7],
        [0.6, 2.0, 0.4],
        [0.7, 1.5, 0.6],
    ]
    
    for x0 in initial_params:
        if time.time() > deadline - 10:
            break
        
        try:
            res = so.minimize(
                param_objective,
                x0,
                method='Nelder-Mead',
                bounds=[(0.1, 0.99), (0.5, 100.0), (0.05, 0.95)],
                options={
                    'maxiter': 5000,
                    'xatol': 1e-10,
                    'fatol': 1e-12,
                    'adaptive': True
                }
            )
            score = param_objective(res.x)
            if score < best_score:
                best_score = score
                best_h = _top_hat_tapered(res.x).copy()
        except Exception:
            pass
    
    # Also try a finer grid search in the parameter space for robustness
    if time.time() < deadline - 20:
        # Grid search over width_frac, taper_power, center_frac
        widths = np.linspace(0.3, 0.95, 8)
        powers = np.array([0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0])
        centers = np.linspace(0.2, 0.8, 5)
        
        for w in widths:
            if time.time() > deadline - 10:
                break
            for p in powers:
                for c in centers:
                    if time.time() > deadline - 10:
                        break
                    params = [w, p, c]
                    score = param_objective(params)
                    if score < best_score:
                        best_score = score
                        best_h = _top_hat_tapered(params).copy()
    
    # Final verification
    if np.any(best_h < 0):
        best_h = np.maximum(best_h, 0)
    
    return best_h.tolist()
# EVOLVE-BLOCK-END
