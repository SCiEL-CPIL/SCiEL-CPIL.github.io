import numpy as np
import time
from scipy.optimize import minimize

N = 50
w = 1.0 / 100.0  # spacing on [-0.5, 0.5] with m+2=102 points

def compute_value(h):
    h = np.asarray(h, dtype=float)
    g = np.convolve(h, h)
    m = len(g)  # 99
    y = np.concatenate(([0.0], g, [0.0]))  # length m+2 = 101
    sq = y[:m+1]**2 + y[:m+1]*y[1:m+2] + y[1:m+2]**2
    l2 = (w / 3.0) * np.sum(sq)
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    if norm_inf == 0 or norm_1 == 0:
        return 0.0
    return l2 / (norm_1 * norm_inf)

def compute_value_fast(h):
    h = np.asarray(h, dtype=float)
    g = np.convolve(h, h)
    m = len(g)  # 99
    # y[i] for i=0..m+1, where y[0]=0, y[1..m]=g, y[m+1]=0
    # l2 = sum_{i=0}^{m} (w/3)*(y[i]^2 + y[i]*y[i+1] + y[i+1]^2)
    # Let's expand this:
    # For i=0: (w/3)*(0 + 0 + g[0]^2) = (w/3)*g[0]^2
    # For i=1..m-1: (w/3)*(g[i-1]^2 + g[i-1]*g[i] + g[i]^2)
    # For i=m: (w/3)*(g[m-1]^2 + 0 + 0) = (w/3)*g[m-1]^2
    # So l2 = (w/3) * [g[0]^2 + g[m-1]^2 + sum_{i=1}^{m-1}(g[i-1]^2 + g[i-1]*g[i] + g[i]^2)]
    #       = (w/3) * [g[0]^2 + g[m-1]^2 + sum_{i=0}^{m-2}(g[i]^2 + g[i]*g[i+1] + g[i+1]^2)]
    #       = (w/3) * [sum_{i=0}^{m-1} g[i]^2 + sum_{i=0}^{m-2}(g[i]*g[i+1] + g[i+1]^2) + g[m-1]^2]
    # Wait, let me redo this more carefully.
    # sum_{i=0}^{m} (y[i]^2 + y[i]*y[i+1] + y[i+1]^2)
    # = sum_{i=0}^{m} y[i]^2 + sum_{i=0}^{m} y[i]*y[i+1] + sum_{i=0}^{m} y[i+1]^2
    # = (y[0]^2 + ... + y[m]^2) + (y[0]*y[1] + ... + y[m]*y[m+1]) + (y[1]^2 + ... + y[m+1]^2)
    # y[0]=0, y[m+1]=0, y[1..m]=g[0..m-1]
    # First sum: sum_{i=0}^{m} y[i]^2 = 0 + sum_{j=0}^{m-1} g[j]^2 + 0 = sum(g^2)
    # Second sum: sum_{i=0}^{m} y[i]*y[i+1] = 0*y[1] + sum_{i=1}^{m-1} y[i]*y[i+1] + y[m]*0
    #   = sum_{i=1}^{m-1} g[i-1]*g[i] = sum_{j=0}^{m-2} g[j]*g[j+1]
    # Third sum: sum_{i=0}^{m} y[i+1]^2 = sum_{i=1}^{m+1} y[i]^2 = 0 + sum(g^2) + 0 = sum(g^2)
    # So total = 2*sum(g^2) + sum_{j=0}^{m-2} g[j]*g[j+1]
    g2 = np.sum(g*g)
    if m > 1:
        cross = np.sum(g[:-1]*g[1:])
    else:
        cross = 0.0
    l2 = (w / 3.0) * (2.0 * g2 + cross)
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    if norm_inf == 0 or norm_1 == 0:
        return 0.0
    return l2 / (norm_1 * norm_inf)

def neg_value(log_h):
    h = np.exp(log_h)
    return -compute_value_fast(h)

def neg_value_direct(h):
    return -compute_value_fast(h)

def construct():
    deadline = time.time() + 140.0  # 10s safety margin before 150s kill
    best_h = np.ones(N)
    best_val = compute_value_fast(best_h)  # should be 1.0

    rng = np.random.RandomState(42)

    def try_optimize_log(start_log_h, maxiter):
        nonlocal best_h, best_val
        t0 = time.time()
        try:
            res = minimize(
                neg_value,
                start_log_h,
                method='L-BFGS-B',
                options={'maxiter': maxiter, 'maxfun': maxiter * 5, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            h_cand = np.exp(res.x)
            v_cand = compute_value_fast(h_cand)
            if v_cand > best_val:
                best_val = v_cand
                best_h = h_cand.copy()
        except Exception:
            pass
        return time.time() - t0

    def try_optimize_direct(start_h, maxiter):
        nonlocal best_h, best_val
        t0 = time.time()
        try:
            res = minimize(
                neg_value_direct,
                start_h,
                method='L-BFGS-B',
                bounds=[(0, None)] * N,
                options={'maxiter': maxiter, 'maxfun': maxiter * 5, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            h_cand = np.maximum(res.x, 0)
            v_cand = compute_value_fast(h_cand)
            if v_cand > best_val:
                best_val = v_cand
                best_h = h_cand.copy()
        except Exception:
            pass
        return time.time() - t0

    # Start from uniform
    try_optimize_log(np.zeros(N), maxiter=5000)

    # Try a few multi-start perturbations if time remains
    n_starts = 16
    for i in range(n_starts):
        if time.time() > deadline - 25.0:
            break
        
        # Generate diverse starting points
        if i == 0:
            # Smooth random
            x = rng.randn(N)
            kernel = np.ones(5)/5.0
            x = np.convolve(x, kernel, mode='same')
            x = x - x.mean()
            x = 0.5 * x
        elif i == 1:
            # Sine shape
            x = np.sin(np.pi * np.arange(N) / N)
            x = x - x.mean()
        elif i == 2:
            # Cosine shape
            x = np.cos(2*np.pi*np.arange(N)/N) * 0.5
            x = x - x.mean()
        elif i == 3:
            # Triangle shape
            x = 1 - 2*np.abs(np.arange(N)/N - 0.5)
            x = x - x.mean()
        elif i == 4:
            # Gaussian bump
            x = np.exp(-((np.arange(N)-N//2)**2)/30.0)
            x = x - x.mean()
        elif i == 5:
            # Step function
            x = np.where(np.arange(N) < N//2, 0.2, 0.8)
            x = x - x.mean()
        elif i == 6:
            # Two bumps
            x = np.exp(-((np.arange(N)-N//4)**2)/20.0) + np.exp(-((np.arange(N)-3*N//4)**2)/20.0)
            x = x - x.mean()
        elif i == 7:
            # Plateau
            x = np.zeros(N)
            x[N//4:3*N//4] = 1.0
            x = x - x.mean()
        elif i == 8:
            # Linear ramp
            x = np.linspace(-1, 1, N)
            x = x - x.mean()
        elif i == 9:
            # Quadratic
            x = (np.arange(N)/N - 0.5)**2 * 4 - 1
            x = x - x.mean()
        elif i == 10:
            # Random with smoothing
            x = rng.randn(N)
            kernel = np.ones(7)/7.0
            x = np.convolve(x, kernel, mode='same')
            x = x - x.mean()
        elif i == 11:
            # Alternating
            x = np.array([1.0 if j % 2 == 0 else -0.5 for j in range(N)])
            x = x - x.mean()
        elif i == 12:
            # Wide Gaussian
            x = np.exp(-((np.arange(N)-N//2)**2)/80.0)
            x = x - x.mean()
        elif i == 13:
            # Narrow Gaussian
            x = np.exp(-((np.arange(N)-N//2)**2)/10.0)
            x = x - x.mean()
        elif i == 14:
            # Random with more noise
            x = rng.randn(N)
            x = x - x.mean()
        else:
            # Another smooth random
            x = rng.randn(N)
            kernel = np.ones(3)/3.0
            x = np.convolve(x, kernel, mode='same')
            x = x - x.mean()
        
        x = np.maximum(x, -5.0)  # avoid underflow
        try_optimize_log(x, maxiter=3000)
        
        # Also try direct optimization from a positive version
        if time.time() < deadline - 30.0:
            h_start = np.exp(x)
            try_optimize_direct(h_start, maxiter=2000)

    return list(np.asarray(best_h, dtype=float))
