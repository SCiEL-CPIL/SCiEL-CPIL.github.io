"""Second autocorrelation inequality: maximize ||h||_2^2 / (||h||_1 * ||h*h||_inf) for 50 nonnegative steps on [-1/4,1/4]."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import expit

N = 50
W = 1.0 / N  # width of each step, since [-1/4,1/4] has length 1/2 and 50 intervals

# Centers of the 50 intervals
centers = -0.25 + (np.arange(N) + 0.5) * W


def _autoconv_peak(h):
    """
    Compute the supremum of the autoconvolution h*h for piecewise-constant
    functions with heights h on N equal intervals of width W over [-1/4, 1/4].
    
    For piecewise-constant functions on a uniform grid, the autoconvolution at
    point x is:
        (h*h)(x) = sum_{i,j} h_i * h_j * overlap(i,j,x)
    where overlap is the length of intersection of [x - c_j, x - c_j + W] with
    [c_i - W/2, c_i + W/2]... 
    
    Actually, let me think more carefully. The function f is defined as:
        f(t) = h_i for t in [c_i - W/2, c_i + W/2)
    i.e., f is a step function with N steps of width W.
    
    The autoconvolution is (f*f)(x) = integral f(t) f(x-t) dt.
    
    For piecewise-constant f, this can be computed exactly. The key insight:
    (f*f)(x) = sum_i sum_j h_i h_j * measure({t : t in support_i and x-t in support_j})
    
    support_i = [c_i - W/2, c_i + W/2]
    x - t in support_j means t in [x - c_j - W/2, x - c_j + W/2]
    
    So the overlap is the length of intersection of [c_i - W/2, c_i + W/2] and [x - c_j - W/2, x - c_j + W/2].
    
    This is a piecewise-linear function of x. The maximum of (f*f) occurs at points
    where the derivative changes sign, which are at x = c_i + c_j (the peaks of the
    individual triangular contributions).
    
    Actually, for nonnegative h, the autoconvolution is maximized at x = c_i + c_j
    for some i,j, because each term h_i h_j * overlap is a triangular bump centered
    at c_i + c_j with peak value h_i h_j * W.
    
    Wait, that's not quite right either. The overlap of two intervals of width W 
    centered at a and b, as a function of the shift x, is a triangular function 
    peaked at x = a+b with peak W. So (f*f)(x) = sum_{i,j} h_i h_j * tri(x - c_i - c_j)
    where tri(z) = max(0, W - |z|) for |z| < W.
    
    So the maximum of (f*f) is max over x of sum_{i,j} h_i h_j * max(0, W - |x - c_i - c_j|).
    
    The maximum must occur at one of the points x = c_i + c_j for some i,j, because
    between consecutive such points the function is linear (sum of triangles with 
    slopes that are piecewise constant, changing only at the c_i + c_j points).
    
    Actually the function is piecewise linear with breakpoints at all c_i + c_j.
    The maximum of a piecewise-linear function on a bounded interval occurs at a
    breakpoint or at the boundary. The breakpoints are {c_i + c_j : i,j = 0,...,N-1}.
    
    So we just need to evaluate (f*f) at all N^2 points c_i + c_j and take the max.
    """
    h = np.asarray(h, dtype=np.float64)
    # c_i + c_j for all i,j
    sums = centers[:, None] + centers[None, :]  # shape (N, N)
    
    # For each pair (i,j), the contribution at point x is h_i * h_j * max(0, W - |x - (c_i+c_j)|)
    # We need max over x of sum_{i,j} h_i h_j max(0, W - |x - s_{ij}|)
    # where s_{ij} = c_i + c_j.
    # 
    # Evaluate at all candidate points x = s_{kl} for k,l in 0..N-1.
    # But that's N^2 candidate points and N^2 terms each = N^4 which is 6.25M for N=50.
    # That's manageable.
    
    # More efficiently: for a fixed x, (f*f)(x) = sum_i h_i * sum_j h_j * max(0, W - |x - c_i - c_j|)
    # = sum_i h_i * (h * tri_shift)(c_i) where tri_shift depends on x.
    
    # Let's just do the direct N^2 x N^2 computation but vectorized.
    # Actually, let's think about this differently.
    
    # (f*f)(x) = integral f(t) f(x-t) dt
    # For our step function, f(t) = h_i for t in [(i-0.5)W - 0.25, (i+0.5)W - 0.25]
    
    # Let me use a cleaner formulation. Define the intervals as I_i = [a_i, b_i] 
    # where a_i = -0.25 + i*W, b_i = -0.25 + (i+1)*W. So width is W.
    # f(t) = h_i for t in I_i.
    
    # (f*f)(x) = sum_i sum_j h_i h_j * length(I_i ∩ (x - I_j))
    # x - I_j = [x - b_j, x - a_j]
    # I_i ∩ [x-b_j, x-a_j] has length max(0, min(b_i, x-a_j) - max(a_i, x-b_j))
    
    # For the maximum, we evaluate at candidate points. The breakpoints of (f*f) 
    # as a function of x are at x = a_i + a_j, a_i + b_j, b_i + a_j, b_i + b_j for all i,j.
    # That's 4*N^2 breakpoints. For N=50, that's 10000 breakpoints.
    # At each breakpoint, evaluating (f*f) takes O(N^2). Total: O(N^4) = 6.25M. Fine.
    
    a = -0.25 + np.arange(N) * W  # left edges
    b = -0.25 + (np.arange(N) + 1) * W  # right edges
    
    # Candidate x values: all a_i + a_j, a_i + b_j, b_i + a_j, b_i + b_j
    # But many of these are the same. Let's just use a_i + a_j and b_i + b_j 
    # and also a_i + b_j and b_i + a_j.
    # Actually, the distinct breakpoints are at x = a_i + a_j, a_i + b_j, b_i + a_j, b_i + b_j.
    # Note a_i + b_j = a_i + a_j + W, and b_i + a_j = a_i + a_j + W, and b_i + b_j = a_i + a_j + 2W.
    # So the breakpoints are at {a_i + a_j + kW : k=0,1,2} for all i,j.
    # a_i + a_j = -0.5 + (i+j)W, so breakpoints are at -0.5 + (i+j+k)W for i,j in 0..N-1, k in 0,1,2.
    # i+j ranges from 0 to 2(N-1) = 98. So i+j+k ranges from 0 to 100.
    # Breakpoints are at -0.5 + m*W for m = 0, 1, ..., 100. That's 101 distinct points!
    
    # Much better! Let's verify: a_i = -0.25 + iW, b_i = -0.25 + (i+1)W = a_i + W.
    # a_i + a_j = -0.5 + (i+j)W
    # a_i + b_j = -0.5 + (i+j+1)W
    # b_i + a_j = -0.5 + (i+j+1)W
    # b_i + b_j = -0.5 + (i+j+2)W
    # So breakpoints at -0.5 + m*W for m in {i+j, i+j+1, i+j+2} for all i,j.
    # min m = 0+0 = 0, max m = (N-1)+(N-1)+2 = 2N = 100.
    # So 101 candidate points: x_m = -0.5 + m*W for m=0,...,100.
    
    m_vals = np.arange(2 * N + 1)  # 0 to 100
    x_candidates = -0.5 + m_vals * W  # 101 points
    
    # For each candidate x, compute (f*f)(x) = sum_i sum_j h_i h_j * overlap(I_i, x - I_j)
    # overlap = max(0, min(b_i, x - a_j) - max(a_i, x - b_j))
    
    # Vectorize: for a fixed x, 
    # min(b_i, x - a_j) - max(a_i, x - b_j) for all i,j
    # = min(b_i, x-a_j) - max(a_i, x-b_j)
    
    # Let's compute for all x_candidates at once.
    # x_candidates shape: (101,)
    # a, b shape: (N,)
    
    # For efficiency, compute in batches or use full vectorization.
    # x shape: (101, 1, 1), a shape: (1, N, 1), b shape: (1, N, 1)
    # x - a_j: (101, 1, N)
    # x - b_j: (101, 1, N)
    # b_i: (1, N, 1)
    # a_i: (1, N, 1)
    
    # min(b_i, x - a_j): (101, N, N)
    # max(a_i, x - b_j): (101, N, N)
    
    # This is 101 * 50 * 50 = 252500 elements. Fine.
    
    x3 = x_candidates[:, None, None]  # (101, 1, 1)
    ai = a[None, :, None]  # (1, N, 1)
    bi = b[None, :, None]  # (1, N, 1)
    aj = a[None, None, :]  # (1, 1, N)
    bj = b[None, None, :]  # (1, 1, N)
    
    upper = np.minimum(bi, x3 - aj)  # (101, N, N)
    lower = np.maximum(ai, x3 - bj)  # (101, N, N)
    overlap = np.maximum(0.0, upper - lower)  # (101, N, N)
    
    # (f*f)(x_m) = sum_{i,j} h_i h_j * overlap[m, i, j]
    # = h @ overlap[m] @ h  (where overlap[m] is an N x N matrix)
    
    h2 = h[None, :, None] * h[None, None, :]  # (1, N, N)
    autoconv = np.sum(overlap * h2, axis=(1, 2))  # (101,)
    
    return np.max(autoconv)


def _objective(h):
    """Negative of the ratio to minimize."""
    h = np.maximum(h, 0.0)  # enforce nonnegativity in evaluation
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    l2_sq = np.sum(h ** 2)
    peak = _autoconv_peak(h)
    if peak < 1e-15:
        return 0.0
    return -(l2_sq / (l1 * peak))


def _smooth_objective(z, sigma=0.1):
    """
    Smoothed objective using soft-max to make it differentiable.
    z is unconstrained, h = sigmoid(z) * scale to ensure nonnegativity.
    """
    h = expit(z)  # in (0, 1)
    # Scale to have reasonable magnitude
    h = h * 2.0  # scale up a bit
    
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    l2_sq = np.sum(h ** 2)
    peak = _autoconv_peak(h)
    if peak < 1e-15:
        return 0.0
    return -(l2_sq / (l1 * peak))


def _smooth_gradient(z, sigma=0.1):
    """Numerical gradient of the smoothed objective."""
    eps = 1e-7
    grad = np.zeros_like(z)
    f0 = _smooth_objective(z)
    for i in range(N):
        z_plus = z.copy()
        z_plus[i] += eps
        f_plus = _smooth_objective(z_plus)
        grad[i] = (f_plus - f0) / eps
    return grad


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_h = np.ones(N)
    best_score = _objective(best_h)
    
    # Multiple restarts with different initializations
    rng = np.random.default_rng(42)
    
    # Strategy 1: Uniform start
    starts = [np.zeros(N)]
    
    # Strategy 2: Multi-peak structures
    for n_peaks in [2, 3, 5, 7, 10]:
        z = np.zeros(N)
        peak_positions = rng.choice(N, size=n_peaks, replace=False)
        z[peak_positions] = 3.0  # sigmoid(3) ≈ 0.95
        starts.append(z)
    
    # Strategy 3: Random starts
    for _ in range(5):
        starts.append(rng.normal(0, 1, N))
    
    # Strategy 4: Alternating patterns
    z_alt = np.zeros(N)
    z_alt[::2] = 2.0
    starts.append(z_alt)
    
    z_alt2 = np.zeros(N)
    z_alt2[::3] = 2.0
    starts.append(z_alt2)
    
    # Strategy 5: Block structures
    for block_size in [2, 4, 5, 10]:
        z_block = np.zeros(N)
        for start in range(0, N, block_size * 2):
            z_block[start:start+block_size] = 2.0
        starts.append(z_block)
    
    for z0 in starts:
        if time.time() > deadline:
            break
        
        try:
            result = minimize(
                _smooth_objective,
                z0,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            h_opt = expit(result.x) * 2.0
            h_opt = np.maximum(h_opt, 0.0)
            score = _objective(h_opt)
            
            if score > best_score:
                best_score = score
                best_h = h_opt.copy()
                
        except Exception:
            continue
    
    # Also try a few more random restarts if time permits
    if time.time() < deadline - 10.0:
        for _ in range(5):
            if time.time() > deadline:
                break
            z0 = rng.normal(0, 1.5, N)
            try:
                result = minimize(
                    _smooth_objective,
                    z0,
                    method='L-BFGS-B',
                    options={'maxiter': 150, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                h_opt = expit(result.x) * 2.0
                h_opt = np.maximum(h_opt, 0.0)
                score = _objective(h_opt)
                if score > best_score:
                    best_score = score
                    best_h = h_opt.copy()
            except Exception:
                continue
    
    # Ensure nonnegativity and return as list of floats
    best_h = np.maximum(best_h, 0.0)
    return list(best_h.astype(float))
