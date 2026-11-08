"""Optimize 50 nonnegative step heights on [-1/4,1/4] to maximize
   (sum h_i^2) / (sum h_i * sup_k (h*h)_k)
   where (h*h)_k = sum_i h_i h_{i+k} is the autoconvolution at lag k.
"""
import time
import numpy as np
from scipy.optimize import minimize

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

def _autoconv_max(h):
    """Compute max_k sum_i h[i] * h[i-k] for the given height array."""
    c = np.correlate(h, h, mode='full')
    return c.max()

def _ratio(h):
    """Compute the ratio (sum h^2) / (sum h * max autoconv)."""
    h = np.asarray(h, dtype=float)
    l2_sq = np.sum(h**2)
    l1 = np.sum(h)
    if l1 < 1e-12:
        return 0.0
    ac_max = _autoconv_max(h)
    if ac_max < 1e-12:
        return 0.0
    return l2_sq / (l1 * ac_max)

def _neg_ratio(h):
    return -_ratio(h)

def _coordinate_descent(h, max_iter=500, tol=1e-10):
    """Coordinate descent: optimize one coordinate at a time."""
    h = np.asarray(h, dtype=float).copy()
    n = len(h)
    for it in range(max_iter):
        improved = False
        for i in range(n):
            h_old = h[i]
            # Try a few values around current
            candidates = [0.0]
            if h_old > 0:
                candidates.extend([h_old * 0.5, h_old * 1.5, h_old * 0.1, h_old * 5.0])
            else:
                candidates.extend([0.1, 0.5, 1.0])
            # Add some random perturbations
            for _ in range(3):
                candidates.append(np.random.exponential(scale=max(h_old, 0.1)))
            
            best_val = _ratio(h)
            best_h_i = h_old
            for v in candidates:
                h[i] = v
                r = _ratio(h)
                if r > best_val:
                    best_val = r
                    best_h_i = v
            h[i] = best_h_i
            if abs(best_h_i - h_old) > tol:
                improved = True
        if not improved:
            break
    return h

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    best_h = np.ones(N)
    best_score = _ratio(best_h)
    rng = np.random.default_rng(42)
    
    # Strategy 1: Try very sparse signals (single spike, few spikes)
    # Single spike should give ratio ~1.0
    for pos in range(N):
        h = np.zeros(N)
        h[pos] = 1.0
        r = _ratio(h)
        if r > best_score:
            best_score = r
            best_h = h.copy()
    
    # Two spikes at various distances
    for i in range(N):
        for j in range(i+1, N):
            if time.time() > deadline - 60:
                break
            h = np.zeros(N)
            h[i] = 1.0
            h[j] = 1.0
            r = _ratio(h)
            if r > best_score:
                best_score = r
                best_h = h.copy()
    
    # Strategy 2: Coordinate descent from diverse starts
    starts = []
    # Sparse random
    for _ in range(20):
        h = np.zeros(N)
        k = rng.integers(1, 8)
        idx = rng.choice(N, size=k, replace=False)
        h[idx] = rng.uniform(0.5, 2.0, size=k)
        starts.append(h)
    # Dense random
    for _ in range(10):
        starts.append(rng.exponential(scale=1.0, size=N))
    # Structured
    x = np.linspace(0, N-1, N)
    for sigma in [2, 4, 6]:
        starts.append(np.exp(-0.5 * ((x - N//2) / sigma)**2))
    # Alternating
    alt = np.zeros(N)
    alt[::2] = 1.0
    starts.append(alt)
    # Block
    for blk in [5, 10, 25]:
        h = np.zeros(N)
        h[:blk] = 1.0
        starts.append(h)
    
    for h0 in starts:
        if time.time() > deadline - 10:
            break
        try:
            h_cd = _coordinate_descent(h0, max_iter=200)
            r = _ratio(h_cd)
            if r > best_score:
                best_score = r
                best_h = h_cd
        except Exception:
            pass
    
    # Strategy 3: L-BFGS-B refinement from best
    if time.time() < deadline - 5:
        try:
            h0 = best_h.copy()
            if h0.max() > 0:
                h0 = h0 / h0.max()
            res = minimize(_neg_ratio, h0, method='L-BFGS-B',
                          bounds=[(0, None)] * N,
                          options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10})
            if res.fun < _neg_ratio(best_h):
                best_h = res.x
                best_score = -res.fun
        except Exception:
            pass
    
    # Strategy 4: More coordinate descent rounds from current best
    if time.time() < deadline - 3:
        try:
            h_cd = _coordinate_descent(best_h, max_iter=300)
            r = _ratio(h_cd)
            if r > best_score:
                best_score = r
                best_h = h_cd
        except Exception:
            pass
    
    best_h = np.maximum(best_h, 0.0)
    return list(best_h)
