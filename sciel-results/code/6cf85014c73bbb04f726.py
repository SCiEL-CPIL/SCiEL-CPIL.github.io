"""Optimize 50 nonnegative step heights on [-1/4,1/4] to maximize
   (sum h_i^2) / (sum h_i * sup_k (h*h)_k)
   where (h*h)_k = sum_i h_i h_{i+k} is the autoconvolution at lag k.
"""
import time
import numpy as np

N = 50
# Grid points for the 50 intervals: x_i = -1/4 + (i + 0.5)/50 for i in 0..49
# Actually "50 nonnegative heights on equal intervals of [-1/4, 1/4]"
# This means we have 50 steps, each of width 1/50, covering [-1/4, 1/4].
# The autoconvolution of a step function: if h_i is the height on interval i,
# then the convolution at offset k (in units of intervals) is sum_i h_i * h_{i+k}
# But we need to be careful: the autoconvolution is a function, and we take its supremum.
# For a step function with N steps, the autoconvolution at continuous offset t is:
#   (f*f)(t) = integral f(x) f(x-t) dx
# If f is constant h_i on interval [x_i, x_{i+1}], then:
#   (f*f)(t) = sum over pairs of intervals of h_i h_j * (overlap length of interval i and interval j shifted by t)
# The supremum over t of this convolution.

# For simplicity and speed, let's compute the discrete autoconvolution:
# c[k] = sum_i h[i] * h[i+k] for k = -(N-1)..(N-1)
# But the true supremum of the continuous convolution might be higher than the max of c[k].
# However, for step functions, the convolution is piecewise linear (actually piecewise constant? no, piecewise linear with slope changes at interval boundaries).
# Actually, the convolution of two step functions is a piecewise linear function. Its maximum occurs either at a point where the derivative changes (i.e., at an interval boundary offset) or where the derivative is zero in between.

# For a practical approach: compute the convolution on a fine grid and take the max.
# But that's slow for optimization. Let's use the discrete approximation first, then refine.

# Actually, let's think more carefully. The step function f(x) = h_i for x in [x_i, x_{i+1}).
# The autoconvolution (f*f)(t) = integral f(x) f(x-t) dx.
# This is a piecewise linear function of t. The breakpoints are at t = x_j - x_i for all i,j.
# Since x_i are equally spaced with spacing dx = 1/50, the breakpoints are at multiples of dx.
# So (f*f)(t) is piecewise linear with breakpoints at t = k*dx for integer k.
# In each interval between consecutive breakpoints, (f*f)(t) is linear.
# Therefore, the maximum occurs at a breakpoint t = k*dx.
# At t = k*dx, (f*f)(k*dx) = sum_i h_i * h_{i-k} * dx ... wait, let me recompute.

# Let dx = 1/50. Interval i covers [i*dx - 1/4, (i+1)*dx - 11/4]... no.
# Let's say interval i covers [a_i, a_{i+1}] where a_i = -1/4 + i*dx.
# f(x) = h_i for x in [a_i, a_{i+1}).
# (f*f)(t) = integral f(x) f(x-t) dx.
# For t = k*dx, the shift aligns interval i with interval i-k (partially).
# Actually, the overlap between interval [a_i, a_{i+1}] and [a_j + t, a_{j+1} + t] is:
#   max(0, min(a_{i+1}, a_{j+1}+t) - max(a_i, a_j + t))
# For t = k*dx, a_{j+1}+t = a_j + (j+1-k)*dx... this is getting complicated.

# Simpler: since the grid is uniform, the autoconvolution at offset k*dx is:
#   c[k] = dx * sum_i h_i * h_{i-k}  (with h out of range = 0)
# And since the function is piecewise linear between these points, and linear on each segment,
# the max is at one of the k*dx points. So sup_t (f*f)(t) = max_k c[k] = dx * max_k sum_i h_i h_{i-k}.

# So the ratio is:
#   R = (sum h_i^2 * dx) / (sum h_i * dx * max_k sum_i h_i h_{i-k})
#   = (sum h_i^2) / (sum h_i * max_k sum_i h_i h_{i-k})
# The dx cancels! So we can just work with the discrete autoconvolution.

def _autoconv(h):
    """Compute full autoconvolution: c[k] = sum_i h[i] * h[i-k]."""
    return np.correlate(h, h, mode='full')

def _ratio_and_grad(h):
    """Compute ratio and its gradient.
    
    R = ||h||^2 / (||h||_1 * max_k c_k)
    
    We use the fact that R is homogeneous of degree 0, so we can normalize.
    Let A = ||h||^2, B = ||h||_1, C = max_k c_k.
    R = A / (B * C)
    
    dA/dh_i = 2*h_i
    dB/dh_i = 1
    dC/dh_i: C = c[k*] where k* is the argmax.
             c[k] = sum_j h[j] * h[j-k]
             dc[k]/dh_i = h[i-k] + h[i+k]  (with out-of-range = 0)
    
    dR/dh_i = (dA * B * C - A * (dB * C + B * dC)) / (B * C)^2
            = (dA * B * C - A * C * dB - A * B * dC) / (B^2 * C^2)
            = (dA * B - A * dB) / (B^2 * C) - (A * dC) / (B * C^2)
    """
    h = np.asarray(h, dtype=float)
    n = len(h)
    A = np.sum(h**2)
    B = np.sum(h)
    if B < 1e-12 or A < 1e-12:
        return 0.0, np.zeros(n)
    
    c = _autoconv(h)
    C = c.max()
    if C < 1e-12:
        return 0.0, np.zeros(n)
    
    # Find which lag(s) achieve the max
    k_max = np.argmax(c)
    # Convert to lag: c[k] corresponds to lag k - (n-1)
    lag = k_max - (n - 1)
    
    R = A / (B * C)
    
    # Gradient
    # dA/dh_i = 2*h_i
    # dB/dh_i = 1
    # dC/dh_i = h[i-lag] + h[i+lag] (with boundary handling)
    
    dA = 2.0 * h
    dB = np.ones(n)
    
    # dC/dh_i: for the max lag k*, c[k*] = sum_j h[j]*h[j-k*]
    # dc[k*]/dh_i = h[i-k*] + h[i+k*] if both in range, else just one term
    dC = np.zeros(n)
    for i in range(n):
        j1 = i - lag
        j2 = i + lag
        if 0 <= j1 < n:
            dC[i] += h[j1]
        if 0 <= j2 < n:
            dC[i] += h[j2]
    
    # dR/dh_i = (dA_i * B - A * dB_i) / (B^2 * C) - (A * dC_i) / (B * C^2)
    grad = (dA * B - A * dB) / (B**2 * C) - (A * dC) / (B * C**2)
    
    return R, grad

def _ratio(h):
    """Compute the ratio only."""
    R, _ = _ratio_and_grad(h)
    return R

def _project(h, min_val=0.0):
    """Project onto nonnegative orthant."""
    return np.maximum(h, min_val)

def _gradient_ascent(h, max_iter=5000, lr_init=0.01, momentum=0.9, deadline=None):
    """Gradient ascent with momentum and adaptive step size."""
    h = np.asarray(h, dtype=float).copy()
    n = len(h)
    velocity = np.zeros(n)
    lr = lr_init
    best_h = h.copy()
    best_R = _ratio(h)
    patience = 0
    max_patience = 500
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        R, grad = _ratio_and_grad(h)
        if R > best_R:
            best_R = R
            best_h = h.copy()
            patience = 0
        else:
            patience += 1
        
        if patience > max_patience:
            break
        
        # Gradient norm for adaptive step
        gnorm = np.linalg.norm(grad)
        if gnorm < 1e-15:
            break
        
        # Adaptive learning rate: scale by 1/sqrt(iteration)
        lr_eff = lr / (1.0 + 0.001 * it)
        
        # Momentum update
        velocity = momentum * velocity + lr_eff * grad
        h_new = h + velocity
        
        # Project to nonnegative
        h_new = _project(h_new)
        
        # Check if improvement; if not, reduce step
        R_new = _ratio(h_new)
        if R_new > R:
            h = h_new
        else:
            # Reduce learning rate
            lr *= 0.9
            if lr < 1e-8:
                break
    
    return best_h, best_R

def _generate_starts(rng, num_starts=30):
    """Generate diverse initial points."""
    starts = []
    x = np.arange(N, dtype=float)
    
    # Sparse random: few nonzero entries
    for _ in range(8):
        h = np.zeros(N)
        k = rng.integers(1, 10)
        idx = rng.choice(N, size=k, replace=False)
        h[idx] = rng.uniform(0.5, 3.0, size=k)
        starts.append(h)
    
    # Gaussian-like profiles
    for sigma in [1.5, 2.5, 4.0, 6.0, 10.0]:
        for center in [N//4, N//2, 3*N//4]:
            h = np.exp(-0.5 * ((x - center) / sigma)**2)
            starts.append(h)
    
    # Multi-peak
    for _ in range(5):
        h = np.zeros(N)
        n_peaks = rng.integers(2, 5)
        centers = rng.choice(N, size=n_peaks, replace=False)
        sigmas = rng.uniform(1.0, 4.0, size=n_peaks)
        for c, s in zip(centers, sigmas):
            h += np.exp(-0.5 * ((x - c) / s)**2)
        starts.append(h)
    
    # Random smooth (low-pass filtered)
    for _ in range(5):
        h = rng.uniform(0, 1, size=N)
        # Simple moving average for smoothing
        kernel = np.ones(5) / 5.0
        h = np.convolve(h, kernel, mode='same')
        starts.append(h)
    
    # Alternating patterns
    for period in [2, 3, 4]:
        h = np.zeros(N)
        h[::period] = 1.0
        starts.append(h)
    
    # Block patterns
    for blk in [3, 5, 10, 15, 25]:
        h = np.zeros(N)
        start = rng.integers(0, N - blk + 1)
        h[start:start+blk] = 1.0
        starts.append(h)
    
    # Pure random
    for _ in range(3):
        starts.append(rng.uniform(0, 2, size=N))
    
    # Sparse with large values
    for _ in range(4):
        h = np.zeros(N)
        k = rng.integers(1, 5)
        idx = rng.choice(N, size=k, replace=False)
        h[idx] = rng.uniform(1.0, 5.0, size=k)
        starts.append(h)
    
    return starts

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    best_h = np.ones(N)
    best_score = _ratio(best_h)
    rng = np.random.default_rng(42)
    
    # Phase 1: Quick evaluation of simple structured starts
    simple_starts = [
        np.ones(N),
        np.zeros(N),  # will be handled
    ]
    # Single spike
    for pos in range(N):
        h = np.zeros(N)
        h[pos] = 1.0
        r = _ratio(h)
        if r > best_score:
            best_score = r
            best_h = h.copy()
    
    # Two spikes
    for i in range(N):
        for j in range(i+1, N):
            if time.time() > deadline - 80:
                break
            h = np.zeros(N)
            h[i] = 1.0
            h[j] = 1.0
            r = _ratio(h)
            if r > best_score:
                best_score = r
                best_h = h.copy()
    
    # Phase 2: Gradient ascent from diverse starts
    starts = _generate_starts(rng, num_starts=30)
    
    # Sort starts by initial score to prioritize promising ones
    start_scores = [(s, _ratio(s)) for s in starts]
    start_scores.sort(key=lambda x: x[1], reverse=True)
    
    for h0, s0 in start_scores:
        if time.time() > deadline - 15:
            break
        try:
            h_opt, R_opt = _gradient_ascent(h0, max_iter=3000, lr_init=0.05, 
                                            momentum=0.9, deadline=deadline - 10)
            if R_opt > best_score:
                best_score = R_opt
                best_h = h_opt
        except Exception:
            pass
    
    # Phase 3: Refine best solution with more iterations and smaller lr
    if time.time() < deadline - 5:
        try:
            h_refined, R_refined = _gradient_ascent(best_h, max_iter=5000, lr_init=0.01,
                                                    momentum=0.95, deadline=deadline - 3)
            if R_refined > best_score:
                best_score = R_refined
                best_h = h_refined
        except Exception:
            pass
    
    # Phase 4: Local search around best with random perturbations
    if time.time() < deadline - 3:
        for _ in range(20):
            if time.time() > deadline - 3:
                break
            h_pert = best_h.copy()
            # Add small random perturbation
            mask = rng.random(N) < 0.3
            h_pert[mask] += rng.normal(0, 0.1, size=mask.sum())
            h_pert = _project(h_pert)
            try:
                h_opt, R_opt = _gradient_ascent(h_pert, max_iter=1000, lr_init=0.02,
                                                momentum=0.9, deadline=deadline - 3)
                if R_opt > best_score:
                    best_score = R_opt
                    best_h = h_opt
            except Exception:
                pass
    
    best_h = np.maximum(best_h, 0.0)
    return list(best_h)
