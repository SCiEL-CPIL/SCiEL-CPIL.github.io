import numpy as np
from scipy.optimize import minimize

N = 50

def _autoconv_peak(h):
    """
    Compute the supremum norm of the autoconvolution of a piecewise-constant
    function defined by heights h on N equal intervals over [-1/4, 1/4].
    
    The autoconvolution of two box functions of width w centered at c_i and c_j
    is a triangle function with peak h_i*h_j*w at c_i+c_j.
    The total autoconvolution is the sum of these triangles.
    The peak of the sum of triangles is not simply the sum of peaks, but for
    non-negative h, the maximum of the sum occurs at one of the centers c_i+c_j.
    However, a more robust approach for the "supremum norm" in the context of
    the L1/L2 ratio bound is often evaluated at the center of mass or simply
    using the discrete convolution proxy if the grid is fine, but here we can
    compute the exact value at the candidate peak locations.
    
    Actually, for the inequality C2 * ||h||_1 * ||h||_inf >= ||h*h||_inf,
    we want to maximize ||h||_2^2 / (||h||_1 * ||h*h||_inf).
    
    Let's compute the exact autoconvolution values at the grid points that are
    sums of centers. The centers are x_i = -1/4 + (i+0.5)*w, where w = 1/25.
    The autoconvolution (h*h)(x) = sum_{i,j} h_i h_j * max(0, 1 - |x - (x_i+x_j)|/w).
    
    The maximum of this function must occur at one of the points x_i + x_j.
    There are N^2 such points. We can compute the value at all of them and take the max.
    """
    w = 1.0 / N
    # Centers of the intervals
    centers = -0.25 + (np.arange(N) + 0.5) * w
    
    # All pairwise sums of centers
    # This is an N x N array
    sum_centers = centers[:, None] + centers[None, :]
    
    # For each pair (i, j), the contribution to the autoconvolution at a point x
    # is h_i * h_j * max(0, 1 - |x - sum_centers[i,j]|/w)
    
    # We want to find max_x sum_{i,j} h_i h_j max(0, 1 - |x - sum_centers[i,j]|/w)
    # Since the function is piecewise linear and concave between the sum_centers,
    # the maximum must be at one of the sum_centers.
    
    # Let's compute the autoconvolution value at each sum_center.
    # This is an N^2 x N^2 matrix which is 2500x2500 = 6.25 million elements.
    # This is feasible.
    
    # But we can be smarter.
    # Let S be the set of sum_centers.
    # For a fixed x in S, the value is sum_{i,j} h_i h_j * max(0, 1 - |x - (x_i+x_j)|/w).
    
    # To avoid memory issues, we can iterate over the candidate x values (which are the sum_centers).
    # But 2500 candidates * 2500 pairs = 6.25M operations per evaluation.
    # This is fast enough for a few hundred iterations.
    
    # Vectorized computation:
    # diff = x - sum_centers (shape: N^2)
    # tri = max(0, 1 - |diff|/w)
    # val = sum(h_i h_j * tri)
    
    # We can precompute the h_i h_j matrix (N x N)
    hh = np.outer(h, h)
    
    max_val = 0.0
    # To save time, we can sample a subset of the sum_centers if needed,
    # but let's try all of them first.
    for x in sum_centers.flatten():
        diff = np.abs(x - sum_centers)
        tri = np.maximum(0, 1 - diff / w)
        val = np.sum(hh * tri)
        if val > max_val:
            max_val = val
            
    return max_val

def construct():
    # Objective function to minimize: - (||h||_2^2 / (||h||_1 * ||h*h||_inf))
    def objective(h_flat):
        h = np.maximum(0, h_flat)  # Enforce non-negativity
        l1 = np.sum(h)
        if l1 < 1e-12:
            return 1e6
        l2_sq = np.sum(h**2)
        peak = _autoconv_peak(h)
        if peak < 1e-12:
            return 1e6
        ratio = l2_sq / (l1 * peak)
        return -ratio
    
    # Initial guess: single spike
    h0 = np.zeros(N)
    h0[N // 2] = 1.0
    
    # Bounds: [0, inf)
    bounds = [(0, None)] * N
    
    # Use L-BFGS-B for optimization
    try:
        res = minimize(objective, h0, method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 500, 'ftol': 1e-9, 'gtol': 1e-8})
        h_opt = np.maximum(0, res.x)
    except Exception:
        h_opt = h0
        
    # Normalize to ensure non-negative and reasonable scale
    # The ratio is scale-invariant, so we can return as is, or normalize to sum=1
    # The problem asks for "heights", so any non-negative vector works.
    # Let's normalize to sum=1 for stability
    s = np.sum(h_opt)
    if s > 0:
        h_opt = h_opt / s
    else:
        h_opt = h0
        
    return h_opt
