import time
import numpy as np
from scipy.optimize import minimize, linear_sum_assignment

N = 50
T = 0.5  # interval [-1/4, 1/4]
dt = T / N

def compute_ratio(h):
    """
    Compute the ratio: ||h||_2^2 / (||h||_1 * ||h * h||_inf)
    where h is a sequence of nonnegative heights on equal intervals.
    
    The autoconvolution (h * h) at discrete level:
    (h*h)[k] = sum_i h[i] * h[k-i]  (with appropriate scaling by dt for continuous approx)
    
    For the discrete version, we need to be careful about scaling.
    Let's think of this as: we have a function f(x) that is piecewise constant,
    f(x) = h[i] * dt for x in [i*dt, (i+1)*dt) after shifting to [0, T).
    
    Actually, let's work directly with the discrete convolution.
    The L2 norm squared of the piecewise constant function:
    ||f||_2^2 = sum_i (h[i]*dt)^2 * (1/dt) = sum_i h[i]^2 * dt
    
    Wait, let me reconsider. If f(x) = h[i] for x in [(i-1)*dt, i*dt) (or similar),
    then ||f||_2^2 = integral |f(x)|^2 dx = sum_i h[i]^2 * dt
    
    ||f||_1 = integral |f(x)| dx = sum_i h[i] * dt
    
    The autoconvolution g(x) = (f*f)(x) = integral f(y)f(x-y) dy
    ||g||_inf = max_x g(x)
    
    For discrete approximation:
    g(k*dt) ≈ sum_i f(i*dt) * f((k-i)*dt) * dt = dt * sum_i h[i] * h[k-i]
    
    So the ratio is:
    R = (sum_i h[i]^2 * dt) / (sum_i h[i] * dt * max_k (dt * sum_i h[i] h[k-i]))
      = (sum_i h[i]^2) / (sum_i h[i] * dt * max_k (sum_i h[i] h[k-i]))
      = (sum_i h[i]^2) / (dt * sum_i h[i] * max_k (sum_i h[i] h[k-i]))
    
    Let's define:
    - num = sum(h^2)
    - den = dt * sum(h) * max(conv(h, h))
    
    where conv is the linear convolution.
    """
    h = np.asarray(h, dtype=float)
    h = np.maximum(h, 0.0)
    
    if np.sum(h) < 1e-15:
        return 0.0
    
    num = np.sum(h**2)
    den_sum = np.sum(h)
    
    # Linear convolution
    conv = np.convolve(h, h, mode='full')
    max_conv = np.max(conv)
    
    if max_conv < 1e-15:
        return 0.0
    
    den = dt * den_sum * max_conv
    return num / den

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # 120 second limit, with safety margin
    
    best_h = np.ones(N)
    best_ratio = compute_ratio(best_h)
    
    # Try some structured initial candidates
    candidates = []
    
    # Uniform
    candidates.append(np.ones(N))
    
    # Concentrated in center
    center = np.zeros(N)
    center[N//2] = 1.0
    candidates.append(center)
    
    # Two peaks
    for gap in range(1, N):
        two_peak = np.zeros(N)
        two_peak[0] = 1.0
        two_peak[gap] = 1.0
        candidates.append(two_peak)
    
    # Random candidates
    np.random.seed(42)
    for _ in range(100):
        h = np.random.exponential(1.0, N)
        candidates.append(h)
    
    for h in candidates:
        if time.time() > deadline - 10:
            break
        r = compute_ratio(h)
        if r > best_ratio:
            best_ratio = r
            best_h = h.copy()
    
    # Local optimization using gradient-free methods
    # Use Nelder-Mead or Powell's method
    
    def neg_ratio(h):
        return -compute_ratio(h)
    
    # Try multiple local optimizations from different starting points
    start_points = [
        np.ones(N),
        np.random.exponential(1.0, N),
        np.zeros(N),
    ]
    start_points[N//2] = 1.0  # This modifies the last element, let's fix
    
    start_points = []
    start_points.append(np.ones(N))
    
    np.random.seed(123)
    for _ in range(20):
        h = np.random.exponential(1.0, N)
        start_points.append(h)
    
    # Add some structured starts
    for pos in range(N):
        h = np.zeros(N)
        h[pos] = 1.0
        start_points.append(h)
    
    for pos1 in range(0, N, 5):
        for pos2 in range(pos1+1, N, 5):
            h = np.zeros(N)
            h[pos1] = 1.0
            h[pos2] = 1.0
            start_points.append(h)
    
    for h0 in start_points:
        if time.time() > deadline - 5:
            break
        
        h0 = np.maximum(h0, 0.0)
        
        # Use Powell's method for local optimization
        try:
            res = minimize(neg_ratio, h0, method='Powell', 
                          options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8})
            if res.success or res.fun < neg_ratio(h0):
                h_opt = np.maximum(res.x, 0.0)
                r = compute_ratio(h_opt)
                if r > best_ratio:
                    best_ratio = r
                    best_h = h_opt.copy()
        except Exception:
            pass
    
    # Final polish: try to improve with a few more iterations
    if time.time() < deadline - 5:
        try:
            res = minimize(neg_ratio, best_h, method='Nelder-Mead', 
                          options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10})
            h_polished = np.maximum(res.x, 0.0)
            r = compute_ratio(h_polished)
            if r > best_ratio:
                best_ratio = r
                best_h = h_polished.copy()
        except Exception:
            pass
    
    # Return as a list of floats
    return list(np.maximum(best_h, 0.0))
