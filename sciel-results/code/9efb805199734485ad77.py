"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 600

def _compute_ratio(heights):
    """Compute the normalized max autoconvolution ratio."""
    h = np.array(heights, dtype=np.float64)
    n = len(h)
    # Autoconvolution via FFT for speed
    # Pad to avoid circular convolution
    size = 2 * n
    f = np.zeros(size)
    f[:n] = h
    conv = np.real(np.fft.ifft(np.fft.fft(f) ** 2))
    # The linear autoconvolution of h with h has length 2n-1.
    # The FFT result `conv` corresponds to circular convolution of length 2n.
    # Since h is zero-padded to length 2n, the first n values of the circular convolution
    # are not necessarily the linear ones if n is not small enough relative to 2n.
    # Actually, for linear convolution of two length-n signals, we need FFT size >= 2n-1.
    # Size 2n is sufficient.
    # The linear convolution c[k] = sum_j h[j] h[k-j].
    # The FFT method gives c[0..2n-2] in the first 2n-1 entries if no wrap-around occurs.
    # With size 2n, wrap-around affects indices >= n.
    # Specifically, conv[0..n-1] are correct linear conv values?
    # Let's check: c[k] for k=0..n-1.
    # In circular conv of size 2n, result[k] = sum_j h[j] h[(k-j) mod 2n].
    # If k < n, and we assume h is zero outside [0, n-1], then for j in [0, n-1],
    # (k-j) mod 2n is in [-(n-1), n-1].
    # If k-j < 0, it wraps to 2n + k - j. Since k < n and j < n, 2n+k-j > n.
    # h[2n+k-j] is 0 because h is only non-zero in [0, n-1] and we padded to 2n.
    # So for k < n, result[k] = sum_{j: k-j >= 0} h[j] h[k-j] = linear conv[k].
    # So conv[0:n] are the first n linear autoconv coefficients.
    # The max of the autoconvolution is likely at k=0 (peak) or nearby.
    # We should check the whole linear convolution.
    # The linear conv has 2n-1 terms.
    # The FFT result of size 2n contains:
    # conv[0..n-1] = linear[0..n-1]
    # conv[n..2n-1] = linear[n..2n-2] + linear[0..n-1] (wrapped)
    # Wait, standard FFT convolution:
    # If we pad to size M >= 2n-1, then ifft(fft(a)*fft(b)) gives linear conv in [0, 2n-2].
    # Here M=2n. So conv[0..2n-1] contains linear conv values, but wrapped?
    # No, if M >= 2n-1, there is no wrap-around in the linear sense.
    # The linear conv has indices 0 to 2n-2.
    # The FFT result has indices 0 to 2n-1.
    # Indices 0 to 2n-2 are the linear conv.
    # Index 2n-1 is 0 (since linear conv ends at 2n-2).
    # So we can just take max of conv[0:2n-1].
    
    # However, due to floating point errors, conv might have tiny negative values.
    # h is non-negative, so conv must be non-negative.
    max_conv = np.max(conv[:2*n-1])
    l1_norm = np.sum(h)
    if l1_norm <= 0:
        return 0.0
    return max_conv / (l1_norm * l1_norm)

def _param_to_h(width_frac, power):
    """Generate step heights from a parametric bump function."""
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 1.0/(2*N)
    half_width = 0.25 * width_frac
    arg = np.abs(x) / (2.0 * half_width)
    arg = np.clip(arg, 0, 1)
    bump = np.cos(np.pi * arg / 2.0) ** power
    bump[arg > 1.0] = 0.0
    return bump

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    # 1. Get a good initial guess using the parametric method
    # We know a narrow bump is better. Let's start with a reasonably narrow one.
    # The optimal shape is roughly triangular or Gaussian-like.
    # Let's use width_frac=0.5, power=2.0 as a solid start.
    best_h = _param_to_h(0.5, 2.0)
    best_ratio = _compute_ratio(best_h)
    
    # Try a few parametric starts to pick the best seed for optimization
    seeds = [
        (0.3, 1.0),
        (0.4, 1.5),
        (0.5, 2.0),
        (0.6, 2.5),
        (0.5, 1.0), # Triangular-ish
    ]
    
    for wf, pw in seeds:
        if time.time() > deadline:
            break
        h = _param_to_h(wf, pw)
        r = _compute_ratio(h)
        if r < best_ratio:
            best_ratio = r
            best_h = h

    # 2. Direct Convex Optimization
    # Objective: f(x) = max(conv(x)) / (sum(x))^2
    # This function is convex for x >= 0.
    # We use L-BFGS-B which handles bounds.
    
    x0 = best_h.copy()
    
    # We need to compute the objective and gradient for L-BFGS-B.
    # For speed, we can use finite differences for gradient if analytic is too complex,
    # but L-BFGS-B can do finite differences.
    # However, 600 variables is small enough that even finite differences might be okay
    # if we limit iterations.
    # Better: Use SLSQP or trust-constr? L-BFGS-B is usually fastest for large N.
    
    # Let's define the objective function for scipy
    def objective(x):
        # Ensure non-negative (bounds will handle this, but just in case)
        x = np.maximum(x, 0)
        return _compute_ratio(x)
        
    # We can also provide a gradient for better performance.
    # Gradient of max(conv) / S^2 where S = sum(x).
    # Let M = max(conv). Let k* be the argmax.
    # f(x) = conv[k*](x) / S^2
    # df/dx_i = (d(conv[k*])/dx_i * S^2 - conv[k*] * 2*S * dS/dx_i) / S^4
    #        = (d(conv[k*])/dx_i * S - 2*conv[k*] * x_i) / S^3
    # d(conv[k*])/dx_i = h[(k*-i) mod N] + h[(k*-i+N) mod N] ? 
    # Convolution: (h*h)[k] = sum_j h[j] h[k-j].
    # d/dx_i (h*h)[k] = h[k-i] + h[k-i+N] (with appropriate indexing/wrapping for linear conv).
    # For linear conv of length 2N-1, index k is in [0, 2N-2].
    # h is defined on [0, N-1].
    # Term 1: h[k-i] if 0 <= k-i < N else 0.
    # Term 2: h[k-i+N] if 0 <= k-i+N < N => -N <= k-i < 0.
    
    def objective_and_grad(x):
        x = np.maximum(x, 0)
        S = np.sum(x)
        if S == 0:
            return 0, np.zeros_like(x)
        
        # Compute convolution
        size = 2 * N
        f = np.zeros(size)
        f[:N] = x
        conv = np.real(np.fft.ifft(np.fft.fft(f) ** 2))
        
        # Find max
        k_max = np.argmax(conv[:2*N-1])
        M = conv[k_max]
        
        # Gradient
        grad = np.zeros(N)
        # d(conv[k_max])/dx_i = x[k_max - i] + x[k_max - i + N] (with bounds check)
        # We need to compute this for all i efficiently.
        # Let v = np.zeros(N)
        # For a fixed k, the vector of derivatives is the circular convolution of x with a delta at k?
        # Actually, d(conv[k])/dx_i is non-zero only if k-i is in [0, N-1] or k-i+N is in [0, N-1].
        # This is equivalent to:
        # g_i = x[(k - i) % N] if (k - i) % N corresponds to valid linear index?
        # Let's just loop or use vectorized ops. N=600 is small.
        
        # Efficient way:
        # The derivative vector d is the linear convolution of x with a delta function at k?
        # No, it's the cross-correlation.
        # d_i = x[k-i] + x[k-i+N]
        
        # We can compute this by shifting x.
        # Let's construct the derivative vector d of length N.
        # d[i] = x[k-i] (if valid) + x[k-i+N] (if valid)
        
        # Vectorized approach:
        # indices1 = k - np.arange(N)
        # valid1 = (indices1 >= 0) & (indices1 < N)
        # d[valid1] += x[indices1[valid1]]
        
        # indices2 = k - np.arange(N) + N
        # valid2 = (indices2 >= 0) & (indices2 < N)
        # d[valid2] += x[indices2[valid2]]
        
        idx = k_max - np.arange(N)
        valid1 = (idx >= 0) & (idx < N)
        d = np.zeros(N)
        d[valid1] += x[idx[valid1]]
        
        idx2 = idx + N
        valid2 = (idx2 >= 0) & (idx2 < N)
        d[valid2] += x[idx2[valid2]]
        
        # grad = (d * S - 2 * M * x) / S**3
        grad = (d * S - 2.0 * M * x) / (S ** 3)
        
        return M / (S * S), grad

    # Optimization
    # We might run multiple local optimizations from slightly perturbed starts to escape local minima
    # (though convexity suggests global min, numerical issues or non-convexity of the max operator
    # switching active constraints could cause issues, but generally it's fine).
    
    # Let's try one robust run.
    # Options for L-BFGS-B:
    # maxiter: limit iterations to stay within time.
    # Each iteration involves a forward FFT (O(N log N)) and gradient computation.
    # N=600, N log N ~ 6000. Very fast.
    # We can afford many iterations.
    
    options = {
        'maxiter': 2000,
        'ftol': 1e-12,
        'gtol': 1e-12,
        'disp': False
    }
    
    # Run optimization
    try:
        res = minimize(objective_and_grad, x0, method='L-BFGS-B', jac=True, bounds=[(0, None)] * N, options=options)
        if res.success or res.status == 0:
            h_opt = res.x
            # Ensure non-negative
            h_opt = np.maximum(h_opt, 0)
            r_opt = _compute_ratio(h_opt)
            if r_opt < best_ratio:
                best_ratio = r_opt
                best_h = h_opt
    except Exception:
        pass

    # If we have time left, try a second optimization from a different start
    if time.time() < deadline - 20.0:
        # Perturb the best solution
        noise = np.random.normal(0, 0.01 * np.max(best_h), N)
        x1 = np.maximum(best_h + noise, 0)
        try:
            res2 = minimize(objective_and_grad, x1, method='L-BFGS-B', jac=True, bounds=[(0, None)] * N, options=options)
            if res2.success or res2.status == 0:
                h_opt2 = np.maximum(res2.x, 0)
                r_opt2 = _compute_ratio(h_opt2)
                if r_opt2 < best_ratio:
                    best_ratio = r_opt2
                    best_h = h_opt2
        except Exception:
            pass

    return best_h.tolist()
# EVOLVE-BLOCK-END
