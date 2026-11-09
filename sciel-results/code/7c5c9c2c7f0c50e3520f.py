import numpy as np
import time
from scipy.optimize import minimize

def _autoconv_max(h, dx):
    """Compute max of autoconvolution and (integral)^2 for step function."""
    # Pad for linear convolution
    n = len(h)
    conv = np.convolve(h, h, mode='full') * (dx * dx)
    integral = np.sum(h) * dx
    return conv.max() / (integral * integral)

def _objective(params, dx, N):
    """Objective: max(f*f) / (int f)^2, with nonnegativity constraint."""
    h = np.maximum(params, 0)
    if np.sum(h) < 1e-15:
        return 1e10
    conv = np.convolve(h, h, mode='full') * (dx * dx)
    integral = np.sum(h) * dx
    return conv.max() / (integral * integral)

def _objective_grad(params, dx, N):
    """Objective and gradient (numerical or analytical)."""
    h = np.maximum(params, 0)
    if np.sum(h) < 1e-15:
        return 1e10, np.zeros_like(params)
    
    conv = np.convolve(h, h, mode='full') * (dx * dx)
    integral = np.sum(h) * dx
    val = conv.max() / (integral * integral)
    
    # Gradient via automatic differentiation is complex; use finite differences
    # for the max part. For 600 dims, this is expensive but we'll use a few steps.
    # Actually, let's use scipy's minimize with numerical gradient.
    return val

def construct():
    N = 600
    dx = (0.25 - (-0.25)) / N  # = 0.5/600
    
    deadline = time.time() + 100  # 100s budget, 20s safety margin
    best_h = np.ones(N)
    best_val = _autoconv_max(best_h, dx)
    
    # Try several parametric initializations
    t = np.linspace(-0.25, 0.25, N)
    candidates = []
    
    # 1. Uniform
    candidates.append(np.ones(N))
    
    # 2. Raised cosine
    for w in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        h = np.maximum(np.cos(np.pi * t / (w * 0.5))**2, 0)
        if h.max() > 0:
            h = h / h.max()
        candidates.append(h)
    
    # 3. Gaussian
    for sigma in [0.05, 0.08, 0.1, 0.12, 0.15, 0.2]:
        h = np.exp(-t**2 / (2 * sigma**2))
        if h.max() > 0:
            h = h / h.max()
        candidates.append(h)
    
    # 4. Super-ellipse / polynomial shapes
    for p in [1.5, 2.0, 3.0, 4.0, 6.0]:
        h = np.maximum(1 - np.abs(t) / 0.25, 0) ** p
        if h.max() > 0:
            h = h / h.max()
        candidates.append(h)
    
    # 5. Bump with flat top
    for flat_frac in [0.2, 0.3, 0.4, 0.5, 0.6]:
        h = np.ones(N)
        mask = np.abs(t) < flat_frac * 0.25
        # Taper the edges
        edge_width = (1 - flat_frac) * 0.25
        for i in range(N):
            if not mask[i]:
                dist = abs(abs(t[i]) - flat_frac * 0.25)
                if dist < edge_width:
                    h[i] = np.cos(np.pi * dist / (2 * edge_width))**2
                else:
                    h[i] = 0
        candidates.append(h)
    
    # Evaluate all candidates
    for h in candidates:
        h = np.maximum(h, 0)
        if np.sum(h) < 1e-15:
            continue
        val = _autoconv_max(h, dx)
        if val < best_val:
            best_val = val
            best_h = h.copy()
    
    # Now do local optimization from the best candidate
    # Use L-BFGS-B with nonnegativity constraint
    def obj_and_grad(x):
        h = np.maximum(x, 0)
        if np.sum(h) < 1e-15:
            return 1e10, np.zeros_like(x)
        conv = np.convolve(h, h, mode='full') * (dx * dx)
        integral = np.sum(h) * dx
        val = conv.max() / (integral * integral)
        # Numerical gradient (central differences for active set)
        grad = np.zeros_like(x)
        eps = 1e-7
        for i in range(N):
            if h[i] < 1e-10:
                continue
            x_plus = x.copy()
            x_plus[i] += eps
            h_plus = np.maximum(x_plus, 0)
            conv_plus = np.convolve(h_plus, h_plus, mode='full') * (dx * dx)
            val_plus = conv_plus.max() / (np.sum(h_plus) * dx * np.sum(h_plus) * dx)
            grad[i] = (val_plus - val) / eps
        return val, grad
    
    # L-BFGS-B with numerical gradient is very slow for 600 dims.
    # Instead, use a simpler projected gradient method.
    
    # Let's use scipy's minimize with 'L-BFGS-B' and let it compute gradients numerically
    # but limit iterations.
    
    try:
        result = minimize(
            _objective,
            best_h.copy(),
            args=(dx, N),
            method='L-BFGS-B',
            bounds=[(0, None)] * N,
            options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        if result.success or result.fun < best_val:
            h_new = np.maximum(result.x, 0)
            val_new = _autoconv_max(h_new, dx)
            if val_new < best_val:
                best_val = val_new
                best_h = h_new
    except Exception:
        pass
    
    # Additional refinement: local averaging / smoothing steps
    # Try to reduce peaks in the autoconvolution by adjusting h
    for iteration in range(50):
        if time.time() > deadline:
            break
        
        h = best_h.copy()
        conv = np.convolve(h, h, mode='full') * (dx * dx)
        integral = np.sum(h) * dx
        val = conv.max() / (integral * integral)
        
        # Find where the autoconvolution peaks are
        # The autoconvolution at lag k is sum_i h[i]*h[i+k]
        # To reduce the max, we can try to "spread out" mass from high-h regions
        # to low-h regions where they don't contribute much to the peak
        
        # Simple heuristic: if a position has high h and its neighbors also high,
        # reduce it slightly and increase a low neighbor
        h_new = h.copy()
        # Smooth: average with neighbors
        h_smooth = np.convolve(h, [0.2, 0.6, 0.2], mode='same')
        h_new = 0.7 * h + 0.3 * h_smooth
        h_new = np.maximum(h_new, 0)
        
        if np.sum(h_new) < 1e-15:
            break
        
        val_new = _autoconv_max(h_new, dx)
        if val_new < val:
            best_h = h_new
            best_val = val_new
        else:
            break
    
    return list(best_h)
