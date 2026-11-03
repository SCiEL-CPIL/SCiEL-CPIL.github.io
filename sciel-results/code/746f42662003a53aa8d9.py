import time
import numpy as np
from scipy.optimize import minimize

N = 600
# Intervals on [-1/4, 1/4], so width of each interval is (1/2) / 600 = 1/1200
dx = 1.0 / (2 * N)

def compute_autoconvolution_max(f):
    """
    f: array of shape (N,) of nonnegative step heights on intervals of [-1/4, 1/4].
    The function is defined as f(x) = f[i] for x in [-1/4 + i*dx, -1/4 + (i+1)*dx).
    
    The autoconvolution is (f * f)(t) = integral f(x) f(t-x) dx.
    Since f is a step function, we can compute this efficiently.
    
    The support of f is [-1/4, 1/4], so the support of f*f is [-1/2, 1/2].
    We want to minimize the normalized maximum of f*f. The normalization is likely
    such that the integral of f is 1 (probability density) or similar. Let's assume
    we normalize f so that its integral is 1, then minimize max(f*f).
    
    Actually, the task says "minimize the normalized maximum of its autoconvolution".
    A common normalization is to set the integral of f to 1. Then the autoconvolution
    at t=0 is the integral of f(x)^2 dx, and the maximum is at least this value.
    
    Let's compute f*f on a grid. The grid for t should cover [-1/2, 1/2].
    The number of points for the convolution: since f has N steps, f*f will have 2N-1
    non-zero intervals if we consider the discrete convolution of the step heights
    with the interval width.
    
    Let's use the discrete convolution approach.
    Let g = f * dx. Then g is the integral of f over each interval, so sum(g) = integral(f).
    The autoconvolution (f*f)(t) can be approximated by (1/dx) * (g * g)(t) where * is
    discrete convolution, but we need to be careful with the scaling.
    
    Actually, let's think of f as a piecewise constant function.
    f(x) = sum_i f[i] * 1_{[-1/4 + i*dx, -1/4 + (i+1)*dx)}(x)
    
    Then (f*f)(t) = sum_i sum_j f[i] f[j] * length of intersection of
    [-1/4 + i*dx, -1/4 + (i+1)*dx) and [t - (-1/4 + j*dx), t - (-1/4 + (i+1)*dx))
    Wait, the second interval is [t - x] for x in the j-th interval, so it's
    [t - (-1/4 + (j+1)*dx), t - (-1/4 + j*dx)].
    
    This is getting complicated. Let's use the FFT-based convolution.
    Pad f to length 2N, zero-pad to 4N for linear convolution, compute FFT, square, inverse FFT.
    The result will be the convolution of the step function with itself, sampled at the
    interval boundaries.
    
    Let's do it properly:
    Create an array F of length 4N.
    F[0:N] = f
    F[N:2N] = 0
    F[2N:3N] = 0
    F[3N:4N] = 0
    Wait, linear convolution of two length-N arrays has length 2N-1.
    So we need at least 2N-1 points. Let's use 4N to be safe.
    
    Actually, numpy.convolve(f, f, mode='full') gives the linear convolution of the step heights.
    The result has length 2N-1. Each element corresponds to the integral of the product
    over the overlapping intervals. But we need to multiply by dx^2 to get the actual
    integral value? No.
    
    Let's reconsider. f is a step function. The value of f*f at a point t is:
    integral f(x) f(t-x) dx.
    
    If we define the discrete sequence f_i = f[i], then the convolution sum is:
    sum_i f_i f_j where the intervals overlap.
    The length of overlap between interval i and the reflected interval j (shifted by t)
    is a function of t.
    
    It's easier to use the FFT method.
    Let's create an array of length M >= 2N-1.
    Put f in the first N positions, zero elsewhere.
    Compute FFT, square, inverse FFT.
    The result is the circular convolution, but if M >= 2N-1, it's the same as linear convolution.
    The values are the integrals of the product, but scaled by dx^2? 
    Actually, the FFT convolution gives sum_i f[i] f[j] for the overlapping parts, but since
    f is constant on intervals, the integral is sum over i,j of f[i] f[j] * (overlap length).
    The FFT method computes sum_i f[i] f[j] * (dx * dx) if we interpret f[i] as the value
    and dx as the spacing? No.
    
    Let's use a simple approach:
    The autoconvolution at t is a piecewise linear function. Its maximum occurs at one of
    the breakpoints, which are at t = k*dx for integer k, where k ranges from -N to N.
    Actually, the breakpoints are at t = (i - j)*dx for i,j in [0, N-1], so t = k*dx for
    k in [-(N-1), N-1].
    
    Let's compute f*f at t = k*dx for k = 0, 1, ..., 2N-2.
    For t = k*dx, the overlap between interval i and the reflected interval j is:
    The i-th interval is [i*dx, (i+1)*dx) in local coordinates (shifted by -1/4).
    The reflected j-th interval for t = k*dx is [k*dx - (j+1)*dx, k*dx - j*dx) = [(k-j-1)*dx, (k-j)*dx).
    The overlap length is the length of intersection of [i*dx, (i+1)*dx) and [(k-j-1)*dx, (k-j)*dx).
    This is a standard discrete convolution with a triangular kernel? No, it's just the
    number of pairs (i,j) such that the intervals overlap, weighted by the overlap length.
    
    Actually, for step functions, the autoconvolution at t = k*dx is:
    sum_{i,j} f[i] f[j] * max(0, 1 - |i - j - k|) * dx
    Wait, let's check. The overlap length between interval i and the shifted interval j
    (shifted by k*dx) is max(0, 1 - |i - j - k|) * dx.
    So (f*f)(k*dx) = dx * sum_{i,j} f[i] f[j] * max(0, 1 - |i - j - k|).
    
    This is equivalent to: let h = f * f (discrete convolution of the arrays f and f).
    Then (f*f)(k*dx) = dx * (h * tri)(k) where tri is [0, 1, 1, ..., 1, 0]? No.
    
    Let's just compute it directly for now, and optimize later.
    """
    # Compute discrete convolution of f with f
    conv = np.convolve(f, f, mode='full')
    # The autoconvolution at t = k*dx is conv[k] * dx^2? 
    # Let's test with a simple case.
    # If f = [1, 0, 0, ..., 0], then f*f should be a rectangle of width dx and height 1.
    # conv = [1, 0, 0, ..., 0]
    # So (f*f)(0) = 1 * dx? But the integral of f*f over t should be (integral f)^2 = 1.
    # The area of f*f is dx * 1 = dx. That's not 1.
    # So we need to multiply by 1/dx? 
    # Actually, the autoconvolution is (1/dx) * conv * dx^2 = dx * conv? 
    # Let's think: f(x) = 1 for x in [0, dx), 0 otherwise.
    # f*f(t) = integral f(x) f(t-x) dx.
    # For t in [0, dx), f(x) f(t-x) = 1 when x in [0, t] and t-x in [0, dx), i.e., x in [0, t].
    # So f*f(t) = t for t in [0, dx], and f*f(t) = dx - t for t in [dx, 2dx].
    # The maximum is dx at t = dx/2.
    # conv = [1, 0, ...]
    # So (f*f)(t) is not simply conv[k] * dx.
    
    # This is getting too complicated. Let's use the FFT method with proper scaling.
    # The correct formula is: (f*f)(t) = (1/dx) * (F * F)(t) where F is the FFT of f.
    # Actually, the discrete convolution theorem says that the FFT of the convolution is
    # the product of the FFTs. But we need to account for the dx factors.
    
    # Let's use a different approach: compute the autoconvolution on a fine grid using
    # the fact that it's a piecewise linear function.
    # The breakpoints are at t = k*dx for k = -N+1, ..., N-1.
    # Let's compute f*f at these points.
    
    # For t = k*dx, the value is:
    # sum_{i=0}^{N-1} sum_{j=0}^{N-1} f[i] f[j] * overlap(i, j, k)
    # where overlap(i, j, k) = max(0, 1 - |i - j - k|) * dx
    
    # This can be computed as a convolution of f with a triangular kernel.
    # Let tri = [1, 1, ..., 1] (length N)
    # Then overlap sum = dx * conv(f, f, mode='full') * something?
    
    # Actually, let's just compute it directly for now.
    # We'll compute f*f at t = k*dx for k = 0, 1, ..., 2N-2.
    # For each k, we sum over i and j such that the intervals overlap.
    
    # This is O(N^3) which is too slow for N=600.
    
    # Let's use the FFT method.
    # The autoconvolution of a step function can be computed as:
    # (f*f)(t) = dx * (f_discrete * f_discrete)(t/dx) where * is discrete convolution,
    # but with a triangular kernel?
    
    # I found a reference: for a step function with values f_i on intervals of width dx,
    # the autoconvolution at t = k*dx is:
    # (f*f)(k*dx) = dx * sum_{i} f[i] f[i-k] for k >= 0? No.
    
    # Let's just use the continuous convolution formula.
    # f*f(t) = integral f(x) f(t-x) dx.
    # Since f is piecewise constant, we can compute this by summing over the intervals.
    # For a given t, the integral is sum over i,j of f[i] f[j] * length of intersection
    # of [i*dx, (i+1)*dx) and [t - (j+1)*dx, t - j*dx).
    
    # The length of intersection is:
    # max(0, min((i+1)*dx, t - j*dx) - max(i*dx, t - (j+1)*dx))
    
    # This is O(N^2) per t, and we have O(N) values of t, so O(N^3) total.
    # For N=600, that's 216 million operations, which might be too slow in Python.
    
    # Let's use NumPy to vectorize the computation.
    
    # Create arrays for i and j
    i = np.arange(N)
    j = np.arange(N)
    I, J = np.meshgrid(i, j, indexing='ij')
    
    # For t = k*dx, k = 0, 1, ..., 2N-2
    K = np.arange(2*N-1)
    
    # The intersection length for given i, j, k is:
    # max(0, min((i+1)*dx, k*dx - j*dx) - max(i*dx, k*dx - (j+1)*dx))
    # = dx * max(0, min(i+1, k-j) - max(i, k-j-1))
    
    # Let's compute this for all i, j, k.
    # This is a 3D array of size N x N x (2N-1), which is 600*600*1199 = 431 million elements.
    # Too large.
    
    # Let's compute the autoconvolution using FFT.
    # The correct formula is:
    # Let F = FFT(f, n=4*N)
    # Then the autoconvolution at t = k*dx is:
    # (f*f)(k*dx) = dx * IFFT(F * F, n=4*N)[k] for k = 0, ..., 2N-2
    # Wait, let's check the scaling.
    
    # Test with f = [1, 0, 0, ..., 0]
    # F = FFT([1, 0, ..., 0]) = [1, 1, 1, ..., 1]
    # F*F = [1, 1, 1, ..., 1]
    # IFFT(F*F) = [1, 0, 0, ..., 0]
    # So (f*f)(0) = dx * 1 = dx.
    # But earlier we computed that the maximum of f*f is dx, which matches.
    # And the integral of f*f is sum_k (f*f)(k*dx) * dx = dx * 1 * dx = dx^2.
    # But (integral f)^2 = (1*dx)^2 = dx^2. So it matches.
    
    # So the formula is:
    # (f*f)(k*dx) = dx * IFFT(FFT(f, n=4*N) ** 2, n=4*N)[k]
    
    # Let's implement this.
    n_fft = 4 * N
    F = np.fft.fft(f, n=n_fft)
    conv = np.fft.ifft(F * F, n=n_fft).real
    # The autoconvolution values at t = k*dx for k = 0, ..., 2N-2 are conv[0:2*N-1] * dx
    autoconv = conv[:2*N-1] * dx
    return np.max(autoconv)

def normalize_f(f):
    """Normalize f so that its integral is 1."""
    integral = np.sum(f) * dx
    if integral > 0:
        return f / integral
    else:
        return f

def objective(f):
    """Minimize the maximum of the autoconvolution, with f normalized to have integral 1."""
    f_norm = normalize_f(f)
    if np.any(f_norm < 0):
        return 1e10
    return compute_autoconvolution_max(f_norm)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10 seconds margin
    
    # Initialize with a smooth function
    x = np.linspace(-0.25, 0.25, N, endpoint=False)
    # Start with a Gaussian-like function
    f0 = np.exp(-10 * x**2)
    f0 = f0 / np.sum(f0) * (1.0 / dx)  # Normalize to integral 1
    
    best_f = f0.copy()
    best_val = objective(f0)
    
    # Use gradient-based optimization
    # We'll use L-BFGS-B with bounds to enforce non-negativity
    
    def grad(f):
        # Numerical gradient
        eps = 1e-8
        g = np.zeros_like(f)
        for i in range(N):
            f_plus = f.copy()
            f_plus[i] += eps
            f_minus = f.copy()
            f_minus[i] = max(0, f[i] - eps)
            g[i] = (objective(f_plus) - objective(f_minus)) / (2 * eps)
        return g
    
    # This numerical gradient is too slow. Let's use a different approach.
    # We'll use the fact that the objective is smooth except at the maximum point.
    # We can use a soft-max approximation.
    
    def soft_max_obj(f, temperature=0.01):
        f_norm = normalize_f(f)
        if np.any(f_norm < 0):
            return 1e10
        n_fft = 4 * N
        F = np.fft.fft(f_norm, n=n_fft)
        conv = np.fft.ifft(F * F, n=n_fft).real
        autoconv = conv[:2*N-1] * dx
        # Soft max
        max_val = np.max(autoconv)
        if max_val <= 0:
            return 0.0
        soft_max = max_val + temperature * np.log(np.sum(np.exp((autoconv - max_val) / temperature)))
        return soft_max
    
    def soft_max_grad(f, temperature=0.01):
        f_norm = normalize_f(f)
        if np.any(f_norm < 0):
            return np.zeros_like(f)
        n_fft = 4 * N
        F = np.fft.fft(f_norm, n=n_fft)
        conv = np.fft.ifft(F * F, n=n_fft).real
        autoconv = conv[:2*N-1] * dx
        max_val = np.max(autoconv)
        if max_val <= 0:
            return np.zeros_like(f)
        # Gradient of soft max
        weights = np.exp((autoconv - max_val) / temperature)
        weights /= np.sum(weights)
        # Gradient of autoconv with respect to f
        # autoconv = dx * IFFT(FFT(f) ** 2)
        # d(autoconv)/df = 2 * dx * IFFT(FFT(f) * conj(FFT(f))) but this is not correct.
        # Actually, d(FFT(f)^2)/df = 2 * FFT(f) * d(FFT(f))/df = 2 * FFT(f) * FFT(I) where I is the impulse.
        # This is getting complicated. Let's use numerical gradient for the soft max.
        
        eps = 1e-6
        g = np.zeros_like(f)
        for i in range(0, N, 2):  # Compute every other component to save time
            f_plus = f.copy()
            f_plus[i] += eps
            f_minus = f.copy()
            f_minus[i] = max(0, f[i] - eps)
            g[i] = (soft_max_obj(f_plus, temperature) - soft_max_obj(f_minus, temperature)) / (2 * eps)
            if i + 1 < N:
                f_plus[i+1] = f[i+1] + eps
                f_minus[i+1] = max(0, f[i+1] - eps)
                g[i+1] = (soft_max_obj(f_plus, temperature) - soft_max_obj(f_minus, temperature)) / (2 * eps)
        return g
    
    # This is still too slow. Let's use a simpler approach: coordinate descent.
    
    # Coordinate descent: update one component at a time.
    f = f0.copy()
    current_val = objective(f)
    
    # Precompute the FFT-related quantities
    n_fft = 4 * N
    
    def update_component(f, i, step_size=0.1):
        """Update the i-th component of f to minimize the objective."""
        f_norm = normalize_f(f)
        n_fft = 4 * N
        F = np.fft.fft(f_norm, n=n_fft)
        conv = np.fft.ifft(F * F, n=n_fft).real
        autoconv = conv[:2*N-1] * dx
        max_val = np.max(autoconv)
        
        # Try to decrease f[i] or increase it
        # We'll do a line search
        best_fi = f[i]
        best_obj = max_val
        
        # Try decreasing f[i]
        for factor in [0.9, 0.8, 0.7, 0.5, 0.3, 0.1]:
            f_new = f.copy()
            f_new[i] = max(0, f[i] * factor)
            f_new_norm = normalize_f(f_new)
            F_new = np.fft.fft(f_new_norm, n=n_fft)
            conv_new = np.fft.ifft(F_new * F_new, n=n_fft).real
            autoconv_new = conv_new[:2*N-1] * dx
            max_new = np.max(autoconv_new)
            if max_new < best_obj:
                best_obj = max_new
                best_fi = f_new[i]
        
        # Try increasing f[i]
        for factor in [1.1, 1.2, 1.5, 2.0]:
            f_new = f.copy()
            f_new[i] = f[i] * factor
            f_new_norm = normalize_f(f_new)
            F_new = np.fft.fft(f_new_norm, n=n_fft)
            conv_new = np.fft.ifft(F_new * F_new, n=n_fft).real
            autoconv_new = conv_new[:2*N-1] * dx
            max_new = np.max(autoconv_new)
            if max_new < best_obj:
                best_obj = max_new
                best_fi = f_new[i]
        
        return best_fi, best_obj
    
    # Coordinate descent loop
    n_iterations = 0
    while time.time() < deadline:
        n_iterations += 1
        improved = False
        for i in range(N):
            if time.time() > deadline - 5:
                break
            f_new_i, obj_new = update_component(f, i)
            if obj_new < current_val - 1e-10:
                f[i] = f_new_i
                current_val = obj_new
                improved = True
        if not improved:
            break
    
    # Final normalization
    f = normalize_f(f)
    final_val = objective(f)
    
    if final_val < best_val:
        best_f = f
        best_val = final_val
    
    return list(best_f)

# Test the function
if __name__ == "__main__":
    f = construct()
    print(f"Length: {len(f)}")
    print(f"Score: {objective(np.array(f))}")
