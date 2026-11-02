import numpy as np
from scipy.optimize import minimize
import time

N = 600

def fft_autocorr(h):
    """Compute autocorrelation S_k = sum_i h_i * h_{i+k} for k=0..N-1 using FFT."""
    # Pad to avoid circular convolution issues
    n_fft = 2 * N
    H = np.fft.rfft(h, n=n_fft)
    # Autocorrelation: inverse FFT of |H|^2
    corr = np.fft.irfft(H * np.conj(H), n=n_fft)
    # corr[k] = sum_i h_i * h_{i+k} for k >= 0 (with proper alignment)
    # Actually, irfft of |H|^2 gives circular autocorrelation
    # For linear autocorrelation, we need to extract the right part
    # corr[k] for k=0..N-1 gives sum_{i=0}^{N-1-k} h[i]*h[i+k]
    return corr[:N]

def compute_score(h):
    """Compute max_k(S_k) / sum(h) where S_k = sum_i h_i h_{i+k}."""
    S = fft_autocorr(h)
    H = np.sum(h)
    if H < 1e-6:
        return np.inf
    return np.max(S) / H

def smooth_max_surrogate(S, beta):
    """Smooth max of S using log-sum-exp."""
    m = np.max(S)
    return m + np.log(np.sum(np.exp(beta * (S - m)))) / beta

def objective_and_grad(x, beta=20.0):
    """
    Objective and gradient for L-BFGS-B.
    x is the raw variable; h = exp(x) ensures positivity.
    We minimize smooth_max(S(h)) / sum(h) where S = autocorr(h).
    But since score is scale invariant, we can fix sum(h)=1 via reparameterization.
    Alternative: minimize smooth_max(S(h)) with h = exp(x) / sum(exp(x)) to enforce sum=1.
    """
    h = np.exp(x)
    h = h / np.sum(h)  # normalize to sum 1
    
    S = fft_autocorr(h)
    val = smooth_max_surrogate(S, beta)
    
    # Gradient via chain rule
    # d/dx_j [smooth_max(S(h))] where h = exp(x)/sum(exp(x))
    # Let's use finite differences for simplicity? No, let's derive it.
    # Actually, for L-BFGS-B, we can use a simpler approach:
    # Work directly with h >= 0 and use bounds, with a penalty for sum(h) != 1.
    # Or better: use the fact that the problem is convex and use a different formulation.
    
    # Simpler: just return the value and use finite-difference gradient
    # But that's slow. Let's compute the gradient analytically.
    
    # Let f(h) = smooth_max(S(h), beta)
    # grad_h f = sum_k w_k * grad_h S_k where w_k = exp(beta*(S_k - m)) / sum_j exp(beta*(S_j - m))
    # grad_h S_k: S_k = sum_i h_i h_{i+k}
    # d/dh_j S_k = h_{j+k} (if j+k < N) + h_{j-k} (if j-k >= 0)
    
    m = np.max(S)
    weights = np.exp(beta * (S - m))
    weights = weights / np.sum(weights)
    
    # grad_h S_k for each k
    # grad_h = sum_k weights[k] * (shifted_h_plus_k + shifted_h_minus_k)
    grad_h = np.zeros(N)
    for k in range(N):
        if weights[k] < 1e-12:
            continue
        # d/dh_j S_k = h_{j+k} + h_{j-k} (with bounds)
        # For term h_{j+k}: valid when j+k < N, i.e., j < N-k
        # For term h_{j-k}: valid when j-k >= 0, i.e., j >= k
        if k == 0:
            grad_h += 2.0 * weights[k] * h
        else:
            # h_{j+k} for j=0..N-1-k
            grad_h[:N-k] += weights[k] * h[k:]
            # h_{j-k} for j=k..N-1
            grad_h[k:] += weights[k] * h[:N-k]
    
    # Now chain rule: h_j = exp(x_j) / sum(exp(x))
    # d h_j / d x_i = h_j * (delta_{ij} - h_i)
    # So grad_x_i = sum_j grad_h_j * d h_j / d x_i
    #              = sum_j grad_h_j * h_j * (delta_{ij} - h_i)
    #              = grad_h_i * h_i - h_i * sum_j grad_h_j * h_j
    
    dot_grad_h_h = np.dot(grad_h, h)
    grad_x = grad_h * h - h * dot_grad_h_h
    
    return val, grad_x

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Start with uniform distribution
    h0 = np.ones(N) / N
    x0 = np.log(h0)  # log of uniform
    
    best_score = compute_score(h0)
    best_h = h0.copy()
    
    # Try several initial shapes
    idx = np.arange(N)
    center = (N - 1) / 2.0
    
    shapes = []
    # Uniform
    shapes.append(np.ones(N) / N)
    # Triangle
    tri = np.clip(1 - np.abs(idx - center) / center, 0, 1)
    if np.sum(tri) > 0:
        shapes.append(tri / np.sum(tri))
    # Gaussians
    for sigma in [N/4, N/8, N/16, N/32, N/64]:
        g = np.exp(-0.5 * ((idx - center) / sigma)**2)
        if np.sum(g) > 0:
            shapes.append(g / np.sum(g))
    # Cosine bump
    cos_hump = 0.5 * (1 + np.cos(2 * np.pi * (idx - center) / N))
    cos_hump = np.clip(cos_hump, 0, None)
    if np.sum(cos_hump) > 0:
        shapes.append(cos_hump / np.sum(cos_hump))
    
    for h_shape in shapes:
        s = compute_score(h_shape)
        if s < best_score:
            best_score = s
            best_h = h_shape.copy()
    
    # Run L-BFGS-B optimization from the best starting point
    if time.time() < deadline - 5.0:
        x_start = np.log(best_h + 1e-15)
        
        try:
            res = minimize(
                objective_and_grad,
                x_start,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if res.x is not None:
                h_opt = np.exp(res.x)
                h_opt = h_opt / np.sum(h_opt)
                h_opt = np.clip(h_opt, 0, None)
                h_opt = h_opt / np.sum(h_opt)
                s_opt = compute_score(h_opt)
                if s_opt < best_score:
                    best_score = s_opt
                    best_h = h_opt
        except Exception:
            pass
    
    # If time remains, try a second round from a different starting point
    if time.time() < deadline - 10.0:
        # Try starting from a "two-hump" shape
        h_two = np.zeros(N)
        half = N // 2
        for i in range(N):
            dist1 = min(abs(i - half/2), N - abs(i - half/2))
            dist2 = min(abs(i - 3*half/2), N - abs(i - 3*half/2))
            h_two[i] = np.exp(-0.5 * (dist1 / (N/16))**2) + np.exp(-0.5 * (dist2 / (N/16))**2)
        if np.sum(h_two) > 0:
            h_two = h_two / np.sum(h_two)
            s_two = compute_score(h_two)
            if s_two < best_score:
                best_score = s_two
                best_h = h_two.copy()
            
            # Optimize from this starting point
            if time.time() < deadline - 5.0:
                x_start2 = np.log(best_h + 1e-15)
                try:
                    res2 = minimize(
                        objective_and_grad,
                        x_start2,
                        method='L-BFGS-B',
                        options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8}
                    )
                    if res2.x is not None:
                        h_opt2 = np.exp(res2.x)
                        h_opt2 = h_opt2 / np.sum(h_opt2)
                        h_opt2 = np.clip(h_opt2, 0, None)
                        h_opt2 = h_opt2 / np.sum(h_opt2)
                        s_opt2 = compute_score(h_opt2)
                        if s_opt2 < best_score:
                            best_score = s_opt2
                            best_h = h_opt2
                except Exception:
                    pass
    
    # Return the best heights found
    return list(best_h)

construct()
