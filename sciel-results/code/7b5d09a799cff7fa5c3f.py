"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import differential_evolution

N = 50
DT = (1.0 / 4 - (-1.0 / 4)) / N  # interval length
START = -0.25
TIME_LIMIT = 110.0  # safety margin below 120s / 150s

# EVOLVE-BLOCK-START

def compute_ratio(h):
    """Compute the ratio ||h||_2^2 / (||h||_1 * ||h*h||_infty) for step heights h on equal intervals."""
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h * h)
    
    # Autoconvolution: h*h at discrete offsets.
    # The step function is constant on [START + i*DT, START + (i+1)*DT) with height h[i].
    # The autoconvolution (h*h)(t) = integral h(s) h(t-s) ds.
    # Since h is a step function, h*h is piecewise linear (a "tent" function for each pair).
    # The maximum of h*h occurs at one of the interval boundaries or midpoints?
    # Actually, for step functions, the convolution is a piecewise quadratic? No, it's piecewise linear?
    # Let's think. h(s) = sum h[i] * I_{[x_i, x_{i+1})}(s).
    # (h*h)(t) = sum_i sum_j h[i] h[j] * I_{[x_i, x_{i+1})}(s) * I_{[x_j, x_{j+1})}(t-s) integrated.
    # The overlap of [x_i, x_{i+1}) and [t-x_{j+1}, t-x_j) is a function of t.
    # The width of overlap is piecewise linear in t, with kinks at t = x_i - x_{j+1} and t = x_i - x_j.
    # The peak of the convolution of two step functions on uniform grid is achieved at t=0 (for even functions) 
    # but more generally, the maximum is at one of the grid points t = k*DT or half-grid points?
    # Actually, for step functions, the convolution is a piecewise linear function with kinks at 
    # t = (i-j)*DT and t = (i-j+1)*DT. The maximum is at one of these kinks.
    # The kinks are at integer multiples of DT. So we only need to evaluate at t = k*DT for integer k.
    # (h*h)(k*DT) = sum_{i,j} h[i] h[j] * width of overlap of [x_i, x_{i+1}) and [x_j + k*DT, x_{j+1} + k*DT).
    # x_i = START + i*DT.
    # The overlap of [START + i*DT, START + (i+1)*DT) and [START + (j+k)*DT, START + (j+k+1)*DT).
    # These intervals overlap iff i == j+k or |i - (j+k)| == 1? No, they can only overlap if they are adjacent or identical.
    # Since the grid is uniform, [x_i, x_{i+1}) and [x_m, x_{m+1}) overlap iff i == m.
    # Wait, that's not right. The shift is by k*DT.
    # Let m = j + k. Then the second interval is [x_m, x_{m+1}).
    # The first is [x_i, x_{i+1}).
    # They overlap iff i == m. If i == m, the overlap is the full interval of length DT.
    # If i != m, the overlap is 0.
    # So (h*h)(k*DT) = sum_i h[i] h[i-k] * DT, where h[i-k] = 0 if i-k is out of bounds.
    # This is the discrete autocorrelation!
    # So ||h*h||_infty is at most max_k (DT * sum_i h[i] h[i-k]).
    # But is it possible that the maximum occurs at non-grid points?
    # The function (h*h)(t) is continuous and piecewise linear with kinks at t = (i-j)*DT.
    # So the maximum must be at one of these kinks, i.e., at t = k*DT.
    # Therefore, ||h*h||_infty = DT * max_k (sum_i h[i] h[i-k]).
    
    # Compute discrete autocorrelation
    # h is length N, indices 0..N-1.
    # For lag k in -(N-1)..(N-1), autocorr[k] = sum_i h[i] h[i-k] (with out-of-bounds as 0)
    # Use FFT for efficiency? N=50 is small, direct computation is fine.
    
    max_auto = 0.0
    for k in range(-(N-1), N):
        s = 0.0
        for i in range(N):
            j = i - k
            if 0 <= j < N:
                s += h[i] * h[j]
        if s > max_auto:
            max_auto = s
    conv_max = DT * max_auto
    if conv_max == 0:
        return 0.0
    return l2_sq / (l1 * conv_max)

def neg_ratio(h):
    return -compute_ratio(h)

def construct():
    start_time = time.time()
    
    # Initial candidate: uniform
    best_h = np.ones(N)
    best_score = compute_ratio(best_h)
    
    # Also try single spike
    spike = np.zeros(N)
    spike[0] = 1.0
    spike_score = compute_ratio(spike)
    if spike_score > best_score:
        best_h = spike
        best_score = spike_score
    
    # Try a few random initializations and use local search (Nelder-Mead or similar)
    # Or use differential evolution with a time limit.
    
    # Let's try a simple local search from a few random starts.
    from scipy.optimize import minimize
    
    bounds = [(0.0, 1.0)] * N  # heights are nonnegative; scale doesn't matter for ratio, so cap at 1
    
    n_trials = 20
    for trial in range(n_trials):
        if time.time() - start_time > TIME_LIMIT - 5:
            break
        h0 = np.random.random(N)
        res = minimize(neg_ratio, h0, method='Nelder-Mead', 
                       options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-4, 'adaptive': True})
        if res.success or res.fun < -best_score:
            candidate = np.clip(res.x, 0, 1)
            score = compute_ratio(candidate)
            if score > best_score:
                best_score = score
                best_h = candidate
        
    # Convert to list of floats
    return [float(x) for x in best_h]

# EVOLVE-BLOCK-END
