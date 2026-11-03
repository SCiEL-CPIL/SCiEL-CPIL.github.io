"""Rewritten program: optimize 50 nonnegative step heights to maximize
    ||h||_2^2 / (||h||_1 * ||h*h||_inf)
    over equal intervals on [-1/4, 1/4].
    Uses SLSQP with analytic gradients on log-heights (positive parametrization)
    plus a few restarts, with a hard deadline guard.
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 50
# Interval width
W = 0.5  # [-1/4, 1/4]
dx = W / N  # each step width

# Precompute the autoconvolution structure.
# h is a vector of N heights. The autoconvolution (h*h)(t) for t in [-1/4,1/4]
# can be computed via FFT or direct sum. Since N=50 is small, we can compute
# the full autoconvolution on a fine grid, but for optimization we need
# ||h*h||_inf. 

# Better: compute the autoconvolution as a function of the discrete autocorrelation.
# The autoconvolution of a step function with constant height h_i on interval i
# is a piecewise linear function. Its maximum occurs at some point.
# 
# Actually, let's compute the autoconvolution on a fine grid and take the max.
# But for gradient computation, we need differentiability. The max is not differentiable
# everywhere, but we can use a smooth approximation or just use the discrete max and
# accept that SLSQP will handle it approximately.
#
# Alternative: The autoconvolution of a step function can be expressed in terms of
# the autocorrelation coefficients. Let's define:
#   c_k = sum_{i=0}^{N-1-k} h_i * h_{i+k}   for k = 0, 1, ..., N-1
# The autoconvolution at position t (in units of dx) is a piecewise linear function
# whose values at integer positions are related to c_k.
#
# Actually, the autoconvolution (h*h)(t) = integral h(s) h(t-s) ds.
# For step functions on equal intervals, this is a piecewise linear function.
# The maximum of this function occurs either at a point where the slope changes
# (i.e., at integer multiples of dx) or at a point where the slope is zero.
#
# Let's just compute the autoconvolution on a fine grid (say, 2*N+1 points at
# half-integer and integer positions) and take the max. For the gradient, we can
# use the fact that the max is achieved at some point, and the gradient with respect
# to h_i is the sum of contributions.
#
# Simpler approach: Use a fine grid for the autoconvolution and compute the max.
# For the gradient, note that if the max is at grid point j, then
#   d(max)/dh_i = sum over appropriate pairs.
#
# Let's define the autoconvolution on a grid of M points.
# M = 2*N + 1, grid points at t_j = (j - N) * dx/2 for j = 0, ..., 2*N
# Wait, the support of h*h is [-1/2, 1/2], so we need to cover that.
# Actually, h is supported on [-1/4, 1/4], so h*h is supported on [-1/2, 1/2].
# Let's use a grid of 4*N+1 points from -1/2 to 1/2.

M = 4 * N + 1
t_grid = np.linspace(-0.5, 0.5, M)

def compute_autoconv(h):
    """Compute the autoconvolution of step function h on t_grid."""
    # h is an array of N heights, each on interval [i*dx - 1/4, (i+1)*dx - 1/4)
    # The autoconvolution at t is:
    #   sum_{i,j} h_i * h_j * length of intersection of [i*dx - 1/4, (i+1)*dx - 1/4)
    #                                       and [t - (j*dx - 1/4), t - ((j+1)*dx - 1/4))
    # This is complex. Let's use FFT-based approach.
    # Pad h to length 2*N, compute FFT, square, IFFT.
    h_padded = np.zeros(2 * N)
    h_padded[:N] = h
    H = np.fft.fft(h_padded)
    H2 = H * H
    conv = np.fft.ifft(H2).real
    # conv[k] corresponds to the autoconvolution at position (k - N) * dx/2? 
    # Actually, the linear convolution of h with h has length 2*N - 1.
    # The FFT gives circular convolution of length 2*N.
    # Let's just compute it directly for N=50.
    return None

def compute_autoconv_direct(h):
    """Direct computation of autoconvolution on t_grid."""
    # h[i] is the height on interval [i*dx - 0.25, (i+1)*dx - 0.25)
    result = np.zeros(M)
    for j, t in enumerate(t_grid):
        val = 0.0
        for i in range(N):
            h_i = h[i]
            if h_i == 0:
                continue
            # Interval for h_i: [i*dx - 0.25, (i+1)*dx - 0.25)
            a_i = i * dx - 0.25
            b_i = (i + 1) * dx - 0.25
            # We need h(t - s) for s in [a_i, b_i)
            # h(t - s) = h_k where k is such that t - s in [k*dx - 0.25, (k+1)*dx - 0.25)
            # So s in [t - (k+1)*dx + 0.25, t - k*dx + 0.25)
            # Intersection of [a_i, b_i) and [t - (k+1)*dx + 0.25, t - k*dx + 0.25)
            for k in range(N):
                h_k = h[k]
                if h_k == 0:
                    continue
                a_k = k * dx - 0.25
                b_k = (k + 1) * dx - 0.25
                # s such that s in [a_i, b_i) and t - s in [a_k, b_k)
                # => s in [t - b_k, t - a_k)
                lo = max(a_i, t - b_k)
                hi = min(b_i, t - a_k)
                if lo < hi:
                    val += h_i * h_k * (hi - lo)
        result[j] = val
    return result

def objective_and_grad(x):
    """x = log(h_i + eps) for positivity. Returns (f, grad) where f is negative of the ratio."""
    h = np.exp(x)
    norm2 = np.sum(h * h)
    norm1 = np.sum(h)
    
    # Compute autoconvolution and its max
    conv = compute_autoconv_direct(h)
    conv_max = np.max(conv)
    
    if conv_max < 1e-15:
        return 0.0, np.zeros(N)
    
    ratio = norm2 / (norm1 * conv_max)
    f = -ratio
    
    # Gradient with respect to x: df/dx_i = df/dh_i * h_i
    # df/dh_i = -d(norm2 / (norm1 * conv_max))/dh_i
    # = -(2*h_i * norm1*conv_max - norm2 * (dh1/dh_i * conv_max + norm1 * dconv_max/dh_i)) / (norm1*conv_max)^2
    # where dh1/dh_i = 1
    # dconv_max/dh_i: if conv_max is at index j*, then dconv_max/dh_i = dconv[j*]/dh_i
    
    j_star = np.argmax(conv)
    # dconv[j_star]/dh_i = sum over k of h_k * (length of intersection for pair (i,k)) 
    #   + h_i * (sum over k of length of intersection for pair (k,i))
    # By symmetry, this is 2 * sum_k h_k * L_{i,k} where L_{i,k} is the length
    # of intersection for the pair, but careful: when i=k, it's h_i * L_{i,i} * 2? 
    # Actually, conv = sum_{i,k} h_i h_k L_{i,k}(t), so dconv/dh_i = sum_k h_k L_{i,k}(t) + sum_k h_k L_{k,i}(t)
    # But L_{i,k}(t) = L_{k,i}(t) by symmetry of the intersection length? 
    # Not exactly, because the intersection depends on t. Let's just compute it.
    
    dconv_max_dh = np.zeros(N)
    t = t_grid[j_star]
    for i in range(N):
        h_i = h[i]
        a_i = i * dx - 0.25
        b_i = (i + 1) * dx - 0.25
        s = 0.0
        for k in range(N):
            h_k = h[k]
            a_k = k * dx - 0.25
            b_k = (k + 1) * dx - 0.25
            lo = max(a_i, t - b_k)
            hi = min(b_i, t - a_k)
            if lo < hi:
                s += h_k * (hi - lo)
        dconv_max_dh[i] = s
    
    # Now: f = -norm2 / (norm1 * conv_max)
    # df/dh_i = -(2*h_i * norm1*conv_max - norm2 * (conv_max + norm1 * dconv_max_dh[i])) / (norm1*conv_max)^2
    
    denom = (norm1 * conv_max) ** 2
    grad_h = np.zeros(N)
    for i in range(N):
        numerator = 2 * h[i] * norm1 * conv_max - norm2 * (conv_max + norm1 * dconv_max_dh[i])
        grad_h[i] = -numerator / denom
    
    grad_x = grad_h * h
    return f, grad_x

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    best_h = np.ones(N)
    best_score = -np.inf
    
    def score(h):
        norm2 = np.sum(h * h)
        norm1 = np.sum(h)
        conv = compute_autoconv_direct(h)
        conv_max = np.max(conv)
        if conv_max < 1e-15:
            return 0.0
        return norm2 / (norm1 * conv_max)
    
    best_score = score(best_h)
    
    # Try a few initial points
    np.random.seed(42)
    for restart in range(5):
        if time.time() > deadline - 30:
            break
        
        if restart == 0:
            x0 = np.zeros(N)
        else:
            # Random positive initial point
            x0 = np.random.uniform(-2, 2, N)
        
        try:
            res = minimize(objective_and_grad, x0, method='SLSQP',
                           jac=True, options={'maxiter': 50, 'ftol': 1e-10})
            if res.success or res.fun < -best_score:
                h_new = np.exp(res.x)
                s = score(h_new)
                if s > best_score:
                    best_score = s
                    best_h = h_new
        except Exception:
            pass
        
        if time.time() > deadline - 20:
            break
    
    # Ensure nonnegative
    best_h = np.maximum(best_h, 0.0)
    return list(best_h)
