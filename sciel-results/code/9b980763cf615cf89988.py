"""First autocorrelation inequality: minimize normalized max of autoconvolution
using LP bisection approach."""
import numpy as np
import time
from scipy.optimize import linprog

N = 600
DX = 1.0 / N  # interval width

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140.0  # safety margin before 150s kill
    
    def compute_score(h):
        """Compute the normalized maximum of the autoconvolution."""
        conv = np.convolve(h, h, mode='full')
        integral = np.sum(h) * DX
        norm = integral * integral
        if norm == 0:
            return np.inf
        return np.max(conv) / norm
    
    # Initial guess: flat function
    h_flat = np.ones(N)
    best_h = h_flat.copy()
    best_score = compute_score(h_flat)
    
    # LP feasibility check: can we find h >= 0 with sum(h)*DX = 1
    # such that max autoconv <= T?
    # Autoconv at lag k: sum_i h[i] * h[i+k] * DX^2 (for appropriate k)
    # We need: for all lags k, sum_i h[i]*h[i+k] <= T / DX^2
    # This is quadratic, not linear. But we can linearize using the fact that
    # we're checking feasibility.
    
    # Alternative: use the dual. The minimum of max autoconv subject to
    # sum(h) = 1/DX and h >= 0 can be found via bisection on T.
    
    # For a given T, the constraint is:
    # For each lag k (from -(N-1) to N-1):
    #   sum_{i} h[i] * h[i+k] * DX^2 <= T
    # where h[i+k] = 0 if i+k out of range
    
    # This is still quadratic. Let's use a different approach:
    # Parametrize h and use scipy.optimize.minimize with a smooth objective.
    
    # Better approach: use the fact that the optimal solution has a specific structure.
    # Try various parametric families and pick the best.
    
    # Family 1: Cosine bump
    def cosine_bump(center=0.0, width=0.5, amplitude=1.0):
        h = np.zeros(N)
        x = np.linspace(-0.25, 0.25, N+1)[:-1]  # left endpoints
        x_centers = x + DX/2
        for i, xc in enumerate(x_centers):
            d = abs(xc - center)
            if d < width:
                h[i] = amplitude * 0.5 * (1 + np.cos(np.pi * d / width))
            else:
                h[i] = 0
        return h
    
    # Family 2: Rectangular
    def rectangular(center=0.0, half_width=0.125, amplitude=1.0):
        h = np.zeros(N)
        x_centers = np.linspace(-0.25, 0.25, N) + DX/2
        for i, xc in enumerate(x_centers):
            if abs(xc - center) < half_width:
                h[i] = amplitude
            else:
                h[i] = 0
        return h
    
    # Family 3: Gaussian
    def gaussian(center=0.0, sigma=0.05, amplitude=1.0):
        h = np.zeros(N)
        x_centers = np.linspace(-0.25, 0.25, N) + DX/2
        for i, xc in enumerate(x_centers):
            h[i] = amplitude * np.exp(-0.5 * ((xc - center) / sigma) ** 2)
        return h
    
    # Generate candidate functions from various families
    candidates = [h_flat]
    
    # Cosine bumps with different parameters
    for center in [0.0]:
        for width in [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]:
            for amp in [0.5, 1.0, 2.0]:
                candidates.append(cosine_bump(center, width, amp))
    
    # Rectangular
    for half_w in [0.05, 0.1, 0.125, 0.15, 0.2]:
        for amp in [0.5, 1.0, 2.0]:
            candidates.append(rectangular(0.0, half_w, amp))
    
    # Gaussian
    for sigma in [0.02, 0.03, 0.05, 0.07, 0.1]:
        for amp in [0.5, 1.0, 2.0]:
            candidates.append(gaussian(0.0, sigma, amp))
    
    # Evaluate all candidates
    for h_cand in candidates:
        if time.time() > deadline:
            break
        s = compute_score(h_cand)
        if s < best_score:
            best_score = s
            best_h = h_cand.copy()
    
    # Local refinement using scipy.optimize.minimize
    # Use L-BFGS-B with a smooth approximation of the max
    
    def smooth_max_conv(h, temp=0.01):
        """Compute smooth max of autoconvolution."""
        conv = np.convolve(h, h, mode='full')
        integral = np.sum(h) * DX
        norm = integral * integral
        if norm == 0:
            return np.inf
        conv_norm = conv / norm
        m = np.max(conv_norm)
        return m + temp * np.log(np.sum(np.exp((conv_norm - m) / temp)))
    
    def obj_and_grad(h):
        """Objective and gradient for smooth max autoconv."""
        conv = np.convolve(h, h, mode='full')
        integral = np.sum(h) * DX
        norm = integral * integral
        if norm == 0:
            return np.inf, np.zeros_like(h)
        
        conv_norm = conv / norm
        
        # Smooth max
        temp = 0.01
        m = np.max(conv_norm)
        obj = m + temp * np.log(np.sum(np.exp((conv_norm - m) / temp)))
        
        # Gradient: d(obj)/d(h[i])
        # conv[k] = sum_j h[j] * h[k-j] (with bounds)
        # d(conv[k])/d(h[i]) = h[k-i] + h[i-k] if valid
        # d(conv_norm[k])/d(h[i]) = (d(conv[k])/d(h[i]) * norm - conv[k] * d(norm)/d(h[i])) / norm^2
        # d(norm)/d(h[i]) = 2 * integral * DX
        
        grad_conv = np.zeros_like(conv)
        for i in range(N):
            for k in range(len(conv)):
                j1 = k - i
                j2 = i - k
                if 0 <= j1 < N:
                    grad_conv[k] += h[j1]
                if 0 <= j2 < N:
                    grad_conv[k] += h[j2]
        
        # This is O(N^2) which is too slow. Use FFT-based gradient.
        # d(conv)/dh[i] is the convolution of delta_i with h, plus h with delta_i
        # = 2 * (h convolved with delta_i) but with proper indexing
        
        # Actually, let's just use numerical gradient or a simpler approach.
        # For now, return just the objective and use finite differences.
        return obj, None
    
    # Use Nelder-Mead or Powell which don't need gradients
    from scipy.optimize import minimize
    
    def obj_func(h_vec):
        h = np.asarray(h_vec)
        h = np.maximum(h, 0)  # enforce nonnegativity
        return smooth_max_conv(h)
    
    # Start from best_h
    h0 = best_h.copy()
    
    # Try Powell optimization
    if time.time() < deadline - 30:
        try:
            res = minimize(obj_func, h0, method='Powell', 
                          options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-6})
            if res.fun < best_score:
                best_h = np.maximum(res.x, 0)
                best_score = res.fun
        except Exception:
            pass
    
    # Try Nelder-Mead
    if time.time() < deadline - 20:
        try:
            res = minimize(obj_func, h0, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-6})
            if res.fun < best_score:
                best_h = np.maximum(res.x, 0)
                best_score = res.fun
        except Exception:
            pass
    
    # Final coordinate descent refinement
    if time.time() < deadline - 5:
        for coord in range(N):
            if time.time() > deadline - 1:
                break
            # Try perturbing this coordinate
            for delta in [0.1, -0.1, 0.5, -0.5, 1.0, -1.0]:
                h_trial = best_h.copy()
                h_trial[coord] = max(0, best_h[coord] + delta)
                s = compute_score(h_trial)
                if s < best_score:
                    best_score = s
                    best_h = h_trial.copy()
    
    return list(best_h)
# EVOLVE-BLOCK-END
