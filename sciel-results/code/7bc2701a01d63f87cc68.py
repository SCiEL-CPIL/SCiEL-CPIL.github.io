"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50

# EVOLVE-BLOCK-START
def compute_value(h):
    """Compute the value metric for given heights."""
    h = np.asarray(h, dtype=float)
    if h.size != N:
        h = h[:N]
        if h.size < N:
            h = np.pad(h, (0, N - h.size), 'constant')
    
    if np.any(h < 0) or not np.all(np.isfinite(h)):
        return 0.0

    g = np.convolve(h, h)
    m = len(g)
    y = np.zeros(m + 2)
    y[1:-1] = g
    w = 1.0 / (m + 1)
    
    # l2 = sum over i in 0..m of (w / 3) * (y[i]^2 + y[i] * y[i+1] + y[i+1]^2)
    # y has length m+2. Indices 0..m+1.
    # The sum is i=0 to m.
    # y[i] and y[i+1]
    y_sq_terms = y[:m+1]**2 + y[:m+1]*y[1:m+2] + y[1:m+2]**2
    l2 = (w / 3.0) * np.sum(y_sq_terms)
    
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    
    if norm_1 == 0 or norm_inf == 0:
        return 0.0
    
    value = l2 / (norm_1 * norm_inf)
    return value

def neg_value(h):
    """Negative value for minimization."""
    return -compute_value(h)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5 second safety margin
    
    best_h = np.ones(N)
    best_val = compute_value(best_h)
    
    # Generate diverse starting points
    # We want shapes that maximize l2 / (norm_1 * norm_inf).
    # l2 is related to L2 norm of autocorrelation.
    # norm_1 is L1 norm of autocorrelation.
    # norm_inf is max of autocorrelation.
    # For a signal h, g = h*h (autocorr).
    # If h is concentrated (spiky), norm_inf is large relative to norm_1?
    # Actually, for a delta function, norm_1 = norm_inf = 1, l2 is small (support is 1 point effectively).
    # For a box function (uniform), norm_1 is large, norm_inf is moderate.
    # We want high l2 (smooth autocorr) but low norm_1*norm_inf.
    # This suggests shapes that are "spread out" enough to keep autocorr smooth (high l2)
    # but "concentrated" enough to keep norm_1 low?
    # Wait, norm_1 = sum |g| / (m+1). g = conv(h,h). sum |g| = sum h * sum h = (sum h)^2.
    # So norm_1 = (sum h)^2 / (m+1).
    # norm_inf = max g.
    # l2 is roughly integral of g^2.
    # Let's try various smooth bumps.
    
    x = np.linspace(-1, 1, N)
    starts = []
    
    # 1. Gaussian bumps
    for c in np.linspace(-0.8, 0.8, 7):
        for s in np.linspace(0.2, 1.0, 7):
            h = np.exp(-((x - c)**2) / (2 * s**2))
            starts.append(h)
            
    # 2. Cosine bumps
    for c in np.linspace(-0.8, 0.8, 7):
        for s in np.linspace(0.3, 1.0, 7):
            h = np.where(np.abs(x - c) < s, 0.5 * (1 + np.cos(np.pi * (x - c) / s)), 0.0)
            starts.append(np.maximum(0, h))
            
    # 3. Triangular bumps
    for c in np.linspace(-0.8, 0.8, 7):
        for s in np.linspace(0.3, 1.0, 7):
            h = np.maximum(0, 1 - np.abs(x - c) / s)
            starts.append(h)
            
    # 4. Box bumps
    for c in np.linspace(-0.8, 0.8, 7):
        for s in np.linspace(0.3, 1.0, 7):
            h = np.where(np.abs(x - c) < s, 1.0, 0.0)
            starts.append(h)

    # Evaluate starts
    for h in starts:
        if time.time() > deadline:
            break
        val = compute_value(h)
        if val > best_val:
            best_val = val
            best_h = h

    # Gradient-based optimization from the best start
    # L-BFGS-B handles bounds well.
    # We optimize h directly.
    # To ensure non-negativity, we can use bounds (0, None).
    # However, L-BFGS-B might struggle if the initial point is on the boundary.
    # A common trick is to optimize log(h) or exp(z), but that changes the metric.
    # Since h must be non-negative, and the function is smooth for h>0,
    # we can try L-BFGS-B with bounds.
    
    # Let's try a few L-BFGS-B runs from the top 5 distinct starts.
    # To save time, we'll pick the best start and maybe a couple of others if time permits.
    
    # Sort starts by value
    start_vals = [(compute_value(h), h) for h in starts]
    start_vals.sort(key=lambda x: x[0], reverse=True)
    
    # Use the top few starts for L-BFGS-B
    # Each L-BFGS-B call takes ~0.1-0.5s. We have ~140s.
    # We can afford ~100 runs.
    # But we should be careful.
    
    # Let's just run L-BFGS-B on the single best start first, then refine.
    # Actually, the landscape might have multiple local optima.
    # Let's run L-BFGS-B on the top 10 distinct-ish starts.
    
    # To ensure diversity, we can just take the top 10 from the sorted list.
    top_starts = [h for v, h in start_vals[:10]]
    
    for h0 in top_starts:
        if time.time() > deadline - 10: # Leave 10s for final checks
            break
            
        try:
            res = minimize(
                neg_value, 
                h0, 
                method='L-BFGS-B', 
                bounds=[(0, None)] * N,
                options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8, 'disp': False}
            )
            if res.success or res.fun < -best_val:
                h_opt = res.x
                # Ensure non-negative
                h_opt = np.maximum(0, h_opt)
                val_opt = compute_value(h_opt)
                if val_opt > best_val:
                    best_val = val_opt
                    best_h = h_opt
        except Exception:
            pass

    # Final polish: Coordinate descent or random perturbation if time remains
    if time.time() < deadline - 5:
        # Simple local random search to escape potential flat spots or numerical noise
        step = 0.01
        for it in range(1000):
            if time.time() > deadline:
                break
            d = np.random.randn(N)
            d /= np.linalg.norm(d)
            new_h = best_h + step * d
            new_h = np.maximum(0, new_h)
            val = compute_value(new_h)
            if val > best_val:
                best_val = val
                best_h = new_h
                step *= 1.1
            else:
                step *= 0.9

    return best_h.tolist()
# EVOLVE-BLOCK-END
