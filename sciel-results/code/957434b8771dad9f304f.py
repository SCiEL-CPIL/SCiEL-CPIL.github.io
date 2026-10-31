import numpy as np
from scipy.signal import fftconvolve
import time

N = 600
_dx = (1.0 / 2.0) / N
_centers = -1.0 / 4.0 + (np.arange(N) + 0.5) * _dx

def _compute_score(heights):
    """Compute max(autoconvolution) / ||f||_2^2 for step function."""
    h = np.asarray(heights, dtype=np.float64)
    l2_sq = np.sum(h ** 2) * _dx
    if l2_sq <= 1e-15:
        return 1e10
    
    # Compute autoconvolution using FFT on fine grid
    n_fine = 4096
    t_fine = np.linspace(-0.5, 0.5, n_fine)
    dt = t_fine[1] - t_fine[0]
    
    # Map fine grid to interval indices
    idx = np.floor((t_fine + 0.25) / _dx).astype(int)
    idx = np.clip(idx, 0, N - 1)
    in_support = (t_fine >= -0.25) & (t_fine < 0.25)
    f_fine = np.zeros(n_fine)
    f_fine[in_support] = h[idx[in_support]]
    
    conv_fine = fftconvolve(f_fine, f_fine, mode='full') * dt
    max_conv = np.max(conv_fine)
    
    return max_conv / l2_sq

def _project_simplex(v, s=1.0):
    """Project v onto simplex {x >= 0, sum(x) = s}."""
    n = len(v)
    if n == 0:
        return v
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - s))[0][-1]
    theta = (cssv[rho] - s) / (rho + 1.0)
    return np.maximum(v - theta, 0.0)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    # Start with uniform distribution (L1 normalized: sum(h_i * _dx) = 1)
    # For uniform: h_i = 1/(N*_dx) = 1/0.5 = 2... wait
    # Actually, let's work with raw heights and normalize properly.
    
    # The score is max(autoconv) / (sum(h^2)*dx)
    # This is scale-invariant! So we can fix sum(h) = N (or any constant)
    # and just minimize the ratio.
    
    # Let's fix sum(h_i) = N so that the "average height" is 1.
    # Then for uniform h_i = 1, score = max(autoconv)/(N*dx) = (N*dx)/(N*dx) = 1.
    
    # Gradient descent with simplex projection (sum(h) = N, h >= 0)
    h = np.ones(N)
    
    best_h = h.copy()
    best_score = _compute_score(h)
    
    # Gradient-based optimization
    # Objective: f(h) = max(autoconv(h)) / (sum(h^2)*dx)
    # We'll use numerical gradient with line search
    
    lr = 0.1
    for iteration in range(500):
        if time.time() > deadline - 10:
            break
        
        # Compute current score
        cur_score = _compute_score(h)
        if cur_score < best_score:
            best_score = cur_score
            best_h = h.copy()
        
        # Numerical gradient (use finite differences on a subset for speed)
        # Actually, let's use a simpler approach: perturb and check
        
        # Try random perturbations projected onto the simplex
        n_perturb = 50
        improved = False
        for _ in range(n_perturb):
            if time.time() > deadline - 10:
                break
            
            # Random perturbation
            delta = np.random.randn(N) * 0.01
            h_new = h + lr * delta
            h_new = np.maximum(h_new, 0)
            h_new = h_new / np.sum(h_new) * N  # re-normalize to sum = N
            
            new_score = _compute_score(h_new)
            if new_score < cur_score:
                h = h_new
                cur_score = new_score
                improved = True
        
        if not improved:
            lr *= 0.9
            if lr < 1e-6:
                break
    
    # Try some structured initializations as well
    inits = [
        np.ones(N),
        np.cos(np.pi * (np.arange(N) - N/2) / N),  # cosine bump
        np.exp(-((np.arange(N) - N/2) ** 2) / (2 * (N/4) ** 2)),  # gaussian
        np.maximum(0, 1 - np.abs(np.arange(N) - N/2) / (N/2)),  # triangle
    ]
    
    for h0 in inits:
        if time.time() > deadline - 10:
            break
        h0 = np.maximum(h0, 0)
        if np.sum(h0) < 1e-10:
            continue
        h0 = h0 / np.sum(h0) * N
        
        score0 = _compute_score(h0)
        if score0 < best_score:
            best_score = score0
            best_h = h0.copy()
        
        # Quick gradient descent from this init
        h = h0.copy()
        lr = 0.05
        for _ in range(100):
            if time.time() > deadline - 10:
                break
            cur_score = _compute_score(h)
            if cur_score < best_score:
                best_score = cur_score
                best_h = h.copy()
            
            improved = False
            for _ in range(20):
                if time.time() > deadline - 10:
                    break
                delta = np.random.randn(N) * 0.01
                h_new = h + lr * delta
                h_new = np.maximum(h_new, 0)
                h_new = h_new / np.sum(h_new) * N
                new_score = _compute_score(h_new)
                if new_score < cur_score:
                    h = h_new
                    cur_score = new_score
                    improved = True
            if not improved:
                lr *= 0.9
                if lr < 1e-6:
                    break
    
    return list(best_h)
