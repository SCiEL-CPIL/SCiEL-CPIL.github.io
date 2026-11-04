import numpy as np
from scipy.optimize import minimize
import time

N = 50
DELTA_X = (0.25 - (-0.25)) / N  # 0.5/50 = 0.01
START_TIME = None

def _autoconv_sup(h):
    """Compute sup norm of autoconvolution of step function h on equal intervals."""
    # h is array of N nonnegative heights
    # Autoconvolution: (h*h)(x) = integral h(y)h(x-y) dy
    # For step functions on equal intervals, we can compute via convolution
    # The autoconvolution at discrete points: for each offset k (in units of dx),
    # (h*h) at offset k*dx = sum_i h[i] * h[i-k] * dx^2 (with appropriate bounds)
    # Actually for step functions: f(x) = h[i] for x in [x_i, x_{i+1})
    # (f*f)(t) = integral f(y) f(t-y) dy
    # At t = k*dx: sum over pairs where the overlap is computed
    
    # Simpler: use numpy convolve for the discrete approximation
    # The continuous autoconvolution sampled at grid points:
    # For step function, (f*f)(k*dx) = sum_{i,j: i-j=k} h[i]*h[j]*dx^2 ... 
    # Actually let me think more carefully.
    
    # f(x) = h[i] for x in [i*dx + offset, (i+1)*dx + offset)
    # (f*f)(t) = integral f(y) f(t-y) dy
    # = sum_i sum_j h[i] h[j] * length of {y: y in bin i, t-y in bin j}
    # = sum_i sum_j h[i] h[j] * max(0, min((i+1)*dx, t - j*dx) - max(i*dx, t - (j+1)*dx))
    
    # For t = k*dx (k integer), the overlap of bin i and bin (t - bin j) ...
    # This is essentially: for integer k, (f*f)(k*dx) = dx * sum_i h[i] * h[i-k] 
    # where h[i-k] = 0 if i-k out of range, but we need to be careful about the step function.
    
    # Actually for a step function on uniform grid, the autoconvolution at grid points k*dx:
    # (f*f)(k*dx) = sum_{i} h[i] * h[i-k] * dx  (for the part where both are defined)
    # Wait, let me reconsider. If f is piecewise constant with f(x) = h[i] on [x_i, x_{i+1}),
    # then (f*f)(t) = integral f(y) f(t-y) dy.
    # For t = k*dx: the integral becomes sum over i of h[i] * integral_{y in bin_i} f(t-y) dy
    # = sum_i h[i] * sum_j h[j] * length(bin_i ∩ (t - bin_j))
    
    # For t = k*dx and uniform grid, bin_i ∩ (k*dx - bin_j) has length:
    # bin_i = [i*dx, (i+1)*dx), k*dx - bin_j = [k*dx - (j+1)*dx, k*dx - j*dx)
    # Overlap length = max(0, min((i+1)*dx, (k-j)*dx) - max(i*dx, (k-j-1)*dx))
    # = dx * max(0, min(i+1, k-j) - max(i, k-j-1))
    # = dx if |i - (k-j)| < 1... this is getting complex.
    
    # Simpler approach: just use FFT-based convolution or direct computation
    # For N=50, direct O(N^2) is fine.
    
    h = np.asarray(h, dtype=float)
    n = len(h)
    # The autoconvolution at points t_k = k*dx for k = -(n-1), ..., n-1
    # (f*f)(k*dx) = sum_i h[i] * h[i-k] * dx  (with h[m]=0 for m out of [0,n-1])
    # This is the standard discrete convolution scaled by dx.
    
    # Using np.convolve:
    conv = np.convolve(h, h, mode='full')  # length 2n-1
    # conv[k] = sum_i h[i] * h[k-i] for the discrete case
    # The continuous autoconvolution at t=(k-(n-1))*dx is conv[k] * dx
    # Wait: np.convolve(h,h)[k] = sum_{i} h[i]*h[k-i]
    # For step functions on [0, (n-1)*dx] (shifted), (f*f)(t) at t = m*dx:
    # = dx * sum_i h[i] * h[i - m] where indices valid
    # The full convolution has 2n-1 elements, index 0 corresponds to t=0 (both at 0)
    # Actually np.convolve(h,h,'full')[k] = sum_{i=max(0,k-n+1)}^{min(n-1,k)} h[i]*h[k-i]
    # This corresponds to autoconvolution at t = (k - (n-1)) * dx if we center... 
    
    # Let me just compute it directly for clarity:
    # (f*f)(t) for t in [-(n-1)*dx, (n-1)*dx]
    # At t = m*dx for m = -(n-1), ..., n-1:
    # (f*f)(m*dx) = dx * sum_{i: 0<=i<n, 0<=i-m<n} h[i] * h[i-m]
    
    sup_val = 0.0
    for m in range(-(n-1), n):
        s = 0.0
        for i in range(n):
            j = i - m
            if 0 <= j < n:
                s += h[i] * h[j]
        val = s * dx  # wait, need to check the scaling
        # Actually: integral f(y)f(t-y)dy where f is step function
        # = sum_i h[i] * integral_{y in bin_i} f(t-y) dy
        # For t = m*dx: integral_{y in [i*dx,(i+1)*dx)} f(m*dx - y) dy
        # f(m*dx - y) = h[j] where m*dx - y in bin_j, i.e., j*dx <= m*dx-y < (j+1)*dx
        # => (m-j-1)*dx < y <= (m-j)*dx
        # Overlap of [i*dx, (i+1)*dx) and ((m-j-1)*dx, (m-j)*dx]
        # = [max(i, m-j-1)*dx, min(i+1, m-j)*dx) if max < min
        # length = max(0, min(i+1, m-j) - max(i, m-j-1)) * dx
        # For integer i, m, j: min(i+1, m-j) - max(i, m-j-1)
        # If m-j >= i+1: min=i+1, max=m-j-1, diff = i+1-(m-j-1) = i+2-m+j... 
        # This is getting messy. Let me just use the convolution approach which is standard.
        
        # For a step function on uniform grid, the autoconvolution at grid points is:
        # (f*f)(m*dx) = dx * (h * h)[m] where (h*h) is the linear convolution
        # and we index appropriately.
        pass
    
    # Let me just use the standard result: for step function f with values h[i] on bins,
    # the autoconvolution at t_k = k*dx (k=-(N-1),...,N-1) is:
    # (f*f)(t_k) = dx * sum_{i} h[i] * h[i-k] (with h out of range = 0)
    # This is dx times the autocorrelation of h.
    
    # Using np.correlate:
    corr = np.correlate(h, h, mode='full')  # corr[k] = sum_i h[i] * h[i-k] for appropriate k
    # np.correlate(h, h, 'full')[k] for k=0,...,2N-2
    # corr[N-1+m] = sum_i h[i]*h[i-m] for m=-(N-1),...,N-1
    # So (f*f)(m*dx) = dx * corr[N-1+m]
    
    sup_val = np.max(corr) * dx
    return sup_val

def _score(h):
    """Compute (L2)^2 / (L1 * sup_autoconv)"""
    h = np.asarray(h, dtype=float)
    h = np.maximum(h, 0)  # ensure nonnegative
    l2_sq = np.sum(h**2) * DELTA_X  # (L2 norm)^2 = integral f^2 = sum h[i]^2 * dx
    l1 = np.sum(h) * DELTA_X        # L1 norm = integral |f| = sum h[i] * dx
    sup_ac = _autoconv_sup(h)
    if l1 <= 0 or sup_ac <= 0:
        return 0.0
    return l2_sq / (l1 * sup_ac)

def _optimize_from_start(h0, time_budget):
    """Run Nelder-Mead + coordinate descent from h0, return best h and score."""
    h0 = np.array(h0, dtype=float)
    h0 = np.maximum(h0, 0)
    
    best_h = h0.copy()
    best_score = _score(h0)
    
    def neg_score(h):
        return -_score(h)
    
    # Nelder-Mead
    try:
        res = minimize(neg_score, h0, method='Nelder-Mead',
                       options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10})
        if res.fun < -best_score:
            best_score = -res.fun
            best_h = res.x.copy()
    except:
        pass
    
    # Coordinate descent
    h = best_h.copy()
    for _ in range(50):
        improved = False
        for i in range(N):
            old = h[i]
            # Try a few perturbations
            for factor in [0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 2.0, 0.0]:
                h[i] = max(0, old * factor) if old > 0 else (old + 0.01 if factor == 1.0 else 0)
                s = _score(h)
                if s > best_score:
                    best_score = s
                    best_h = h.copy()
                    improved = True
            h[i] = old  # restore
        if not improved:
            break
        if time.time() - START_TIME > time_budget:
            break
    
    return best_h, best_score

def construct():
    global START_TIME
    if START_TIME is None:
        START_TIME = time.time()
    
    DEADLINE = 130  # safety margin before 150s kill
    
    best_h = None
    best_score = 0.0
    
    # Initial guesses
    starts = []
    
    # 1. Pure spike at center
    h_spike = np.zeros(N)
    h_spike[N//2] = 1.0
    starts.append(h_spike)
    
    # 2. Spike at different positions
    for pos in [N//4, N//2, 3*N//4]:
        h = np.zeros(N)
        h[pos] = 1.0
        starts.append(h)
    
    # 3. Narrow block (3 consecutive bins)
    for center in [N//4, N//2, 3*N//4]:
        h = np.zeros(N)
        for i in range(max(0, center-1), min(N, center+2)):
            h[i] = 1.0
        starts.append(h)
    
    # 4. Gaussian-like profile
    x = np.linspace(-1, 1, N)
    for sigma in [0.2, 0.3, 0.5, 0.8]:
        h = np.exp(-x**2 / (2*sigma**2))
        h /= np.max(h)
        starts.append(h)
    
    # 5. Triangular profile
    h_tri = np.zeros(N)
    mid = N // 2
    for i in range(N):
        h_tri[i] = max(0, 1 - abs(i - mid) / (mid + 1))
    starts.append(h_tri)
    
    # 6. Two spikes
    for gap in [5, 10, 15]:
        h = np.zeros(N)
        h[N//2 - gap] = 1.0
        h[N//2 + gap] = 1.0
        starts.append(h)
    
    # 7. Random starts
    np.random.seed(42)
    for _ in range(10):
        h = np.random.exponential(1.0, N)
        starts.append(h)
    
    # 8. Sparse random
    for _ in range(10):
        h = np.zeros(N)
        idx = np.random.choice(N, size=5, replace=False)
        h[idx] = np.random.uniform(0.5, 1.0, size=5)
        starts.append(h)
    
    # Evaluate each start and optimize the best ones
    for h0 in starts:
        if time.time() - START_TIME > DEADLINE:
            break
        h_opt, s = _optimize_from_start(h0, DEADLINE)
        if s > best_score:
            best_score = s
            best_h = h_opt
    
    if best_h is None:
        best_h = np.ones(N)
    
    # Final check: ensure nonnegative
    best_h = np.maximum(best_h, 0)
    
    return list(best_h)

# Set start time when module loads
START_TIME = time.time()
