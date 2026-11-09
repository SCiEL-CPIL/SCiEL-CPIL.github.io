import numpy as np
import time
from scipy.optimize import minimize

N = 600

def compute_score(h):
    h = np.maximum(h, 0)
    area = np.sum(h)
    max_val = np.max(h)
    if area == 0 or max_val == 0:
        return float('inf')
    conv = np.convolve(h, h, mode='full')
    return np.max(conv) / (area * max_val)

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    i = np.arange(N)
    
    def param_to_h(params):
        """Convert parameters to step function heights."""
        # params: [A, center, sigma, B]
        # h[i] = max(0, A * exp(-(i-center)^2 / (2*sigma^2)) + B)
        A, center, sigma, B = params
        sigma = max(abs(sigma), 0.5)  # Avoid division by zero
        h = A * np.exp(-(i - center)**2 / (2 * sigma**2)) + B
        h = np.maximum(h, 0)
        if np.sum(h) > 0:
            h = h / np.sum(h)  # Normalize to sum=1
        return h
    
    def objective(params):
        h = param_to_h(params)
        return compute_score(h)
    
    best_h = None
    best_score = float('inf')
    
    # Initial best: single spike (seed approach)
    M = 1.0 / np.sqrt(N)
    h1 = np.full(N, (1.0 - M) / (N - 1))
    h1[0] = M
    s1 = compute_score(h1)
    best_score = s1
    best_h = h1.copy()
    
    # Try multiple initializations for L-BFGS-B
    inits = [
        # [A, center, sigma, B]
        [1.0, N//2, 5.0, 0.01],
        [1.0, N//2, 10.0, 0.01],
        [1.0, N//2, 20.0, 0.01],
        [0.5, N//4, 10.0, 0.01],
        [0.5, 3*N//4, 10.0, 0.01],
        [1.0, N//2, 3.0, 0.005],
        [1.0, N//2, 1.0, 0.005],
        [0.3, N//2, 50.0, 0.02],
        [0.3, N//2, 100.0, 0.02],
    ]
    
    bounds = [(0.01, 10.0), (0.0, float(N)), (0.5, float(N)), (-0.5, 0.5)]
    
    for x0 in inits:
        if time.time() > deadline:
            break
        try:
            result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                            options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
            if result.fun < best_score:
                best_score = result.fun
                best_h = param_to_h(result.x)
        except:
            pass
    
    # Also try a different parameterization: two bumps
    def param_to_h2(params):
        """Two Gaussian bumps."""
        A1, c1, s1, A2, c2, s2 = params
        s1 = max(abs(s1), 0.5)
        s2 = max(abs(s2), 0.5)
        h = A1 * np.exp(-(i - c1)**2 / (2 * s1**2)) + A2 * np.exp(-(i - c2)**2 / (2 * s2**2))
        h = np.maximum(h, 0)
        if np.sum(h) > 0:
            h = h / np.sum(h)
        return h
    
    def objective2(params):
        h = param_to_h2(params)
        return compute_score(h)
    
    bounds2 = [(0.01, 10.0), (0.0, float(N)), (0.5, float(N)),
               (0.01, 10.0), (0.0, float(N)), (0.5, float(N))]
    
    inits2 = [
        [1.0, N//4, 10.0, 1.0, 3*N//4, 10.0],
        [1.0, N//3, 5.0, 1.0, 2*N//3, 5.0],
        [0.5, 0.0, 5.0, 0.5, float(N-1), 5.0],
    ]
    
    for x0 in inits2:
        if time.time() > deadline:
            break
        try:
            result = minimize(objective2, x0, method='L-BFGS-B', bounds=bounds2,
                            options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
            if result.fun < best_score:
                best_score = result.fun
                best_h = param_to_h2(result.x)
        except:
            pass
    
    if best_h is None:
        best_h = h1
    
    return list(best_h)
