import numpy as np
import time
import sys

N = 600

def _autoconv_max(heights):
    """
    Compute the maximum of the normalized autoconvolution of the step function.
    
    The step function f(x) = h_i on interval i, where intervals are equal
    on [-1/4, 1/4]. The autoconvolution is (f * f)(t) = integral f(x)f(t-x)dx.
    
    For a step function with N equal intervals of width w = (1/4 - (-1/4))/N = 1/(2N),
    the autoconvolution at discrete points can be computed via FFT.
    
    The normalized maximum is max_t (f*f)(t) / (integral f)^2 * (width of domain)?
    Actually, let me think about this more carefully.
    
    The autoconvolution (f*f)(t) = ∫ f(x) f(t-x) dx
    
    For a step function on [-1/4, 1/4] with N intervals of width w = 1/(2N):
    f(x) = h_i for x in [x_i, x_{i+1}) where x_i = -1/4 + i*w
    
    The convolution is a piecewise linear function. The maximum occurs at some point.
    
    Using FFT: The discrete convolution of the heights with themselves, scaled appropriately.
    
    Let me use the standard approach:
    - The autoconvolution sampled at the interval boundaries can be computed via FFT
    - The maximum of the continuous autoconvolution is the maximum of the piecewise linear interpolation
    
    Actually, for step functions, the autoconvolution is a piecewise quadratic function? No, it's piecewise linear.
    
    Let me think again. f is a step function. f*f at time t:
    (f*f)(t) = sum over i,j of h_i * h_j * measure({x: x in interval i AND t-x in interval j})
    
    This is a piecewise linear function of t. The breakpoints occur at t = x_i - x_j for various i,j.
    
    A simpler approach: use FFT to compute the convolution at the grid points, then find the max.
    
    The key insight: for the purpose of bounding C1, we want to minimize the peak of the normalized autoconvolution.
    
    Normalization: typically C1 relates to ||f*f||_∞ / (||f||_2^2) or similar.
    Let me assume the score is max_t (f*f)(t) where f is the step function, and we want to minimize this.
    
    Actually, re-reading the task: "minimize the normalized maximum of its autoconvolution"
    
    Let me assume the score is: max_t (f*f)(t) / (∫f)^2 * (1/width) or just max_t (f*f)(t)
    
    For the seed program [1.0]*N, let's compute what the score would be.
    
    If f = 1 on [-1/4, 1/4], then (f*f)(t) = length of intersection of [-1/4, 1/4] and [t-1/4, t+1/4]
    = min(1/2, 1/2 - |t|) for |t| <= 1/2, 0 otherwise.
    Max is 1/2 at t=0.
    
    So the seed score is 1/2 = 0.5? But the score is 1.0 for the seed...
    
    Wait, the score is "the upper bound on C1" and "lower is better". The seed has score 1.0.
    So the raw value for the seed must be 1.0.
    
    Let me reconsider. Maybe the normalization is different.
    
    Perhaps: score = max_t (f*f)(t) / (∫f dx)^2 * (1/2) or something.
    
    If f=1 on [-1/4,1/4], ∫f = 1/2, (f*f)(0) = 1/2.
    If score = (f*f)(0) / (∫f)^2 = (1/2)/(1/4) = 2. That's not 1.
    
    If score = (f*f)(0) / (∫f) = (1/2)/(1/2) = 1. Yes! That gives 1.
    
    So score = max_t (f*f)(t) / (∫f dx)
    
    Let me verify: for f=1 on [-1/4,1/4], max (f*f) = 1/2, ∫f = 1/2, ratio = 1. ✓
    
    So I need to compute: max_t (f*f)(t) / (sum(h_i) * w) where w = 1/(2N)
    
    Actually ∫f = sum(h_i * w) = w * sum(h_i)
    
    And (f*f)(t) needs to be computed. Let me use FFT.
    """
    w = 1.0 / (2 * N)
    # Autoconvolution via FFT
    # The continuous autoconvolution of a step function can be computed by:
    # 1. Up-sample to get a finer grid
    # 2. Use FFT convolution
    
    # Simpler: use the fact that for step functions, the autoconvolution at the 
    # interval midpoints or boundaries can be computed.
    
    # Let's compute the autoconvolution at all relevant points using FFT.
    # The autoconvolution (f*f)(t) for step function f with heights h on intervals of width w:
    # (f*f)(t) = sum_{i,j} h_i h_j * max(0, min(x_{i+1}, t+x_j) - max(x_i, t+x_{j+1}))
    
    # This is complex. Let's use a numerical approach with FFT.
    
    # Create a fine grid for the convolution
    # The support of f is [-1/4, 1/4], so f*f has support [-1/2, 1/2]
    # Use a grid with 2N intervals for the convolution
    
    # Actually, let's just use scipy or numpy FFT directly.
    # The discrete approximation: treat h as a signal, compute its autocorrelation.
    
    # The autoconvolution of the step function, sampled at the N+1 interval boundaries,
    # can be computed as follows:
    
    # Let's use a finer discretization for accuracy
    M = 4 * N  # oversample factor
    # Create the fine-grained step function
    # Each original interval becomes M/N = 4 fine intervals
    
    # Actually, for the maximum of the piecewise linear autoconvolution, it suffices
    # to evaluate at the breakpoints, which are at t = (i-j)*w for i,j in 0..N-1
    # That's at most 2N-1 distinct values.
    
    # Let me compute the autoconvolution at t = k*w for k = -(N-1), ..., N-1
    # (f*f)(k*w) = sum_{i,j} h_i h_j * max(0, min(1, k + i - j + 1) - max(0, k + i - j))
    # Wait, let me be more careful.
    
    # Interval i: [i*w, (i+1)*w] shifted to start at -1/4
    # Let's work in units of w. Interval i is [i, i+1] in w-units.
    # f(x) = h_i for x in [i, i+1)
    # (f*f)(t) = integral f(x) f(t-x) dx
    # In w-units, if t = k*w, then:
    # (f*f)(k*w) = w * sum_{i,j} h_i h_j * max(0, min(i+1, k+j+1) - max(i, k+j))
    # Hmm, this is getting complicated.
    
    # Let's just use FFT with a fine grid.
    n_fft = 4 * N
    # Create the fine step function
    fine_h = np.repeat(h := np.array(heights), M // N)
    # Pad for linear convolution
    n_conv = len(fine_h) * 2 - 1
    n_fft = 1 << (n_conv - 1).bit_length()
    
    F = np.fft.rfft(fine_h, n=n_fft)
    conv = np.fft.irfft(F * F.conj(), n=n_fft)[:n_conv]
    
    # The autoconvolution in physical units
    # Each fine interval has width w_fine = w / (M//N) = 1/(2N) / 4 = 1/(8N)
    # Actually, let me just compute the max of conv and scale appropriately.
    
    # The sum of heights times width gives the integral
    integral_f = np.sum(heights) * w
    
    # The max of the autoconvolution
    # conv is the discrete autoconvolution of the fine step function
    # Each sample represents an interval of width w_fine = w / 4
    # The actual autoconvolution value at each point is conv[k] * w_fine
    # Wait, no. The FFT convolution gives sum of products, not the integral.
    
    # Let me think differently. 
    # f(x) = h_i on [x_i, x_{i+1})
    # (f*f)(t) = ∫ f(x) f(t-x) dx
    # If we discretize with fine grid of width dw, then:
    # (f*f)(t_k) ≈ dw * sum_m f(x_m) f(t_k - x_m)
    # This is dw * (f * f)(t_k) in discrete sense
    
    # So if I compute the discrete autocorrelation of the fine step function,
    # multiply by dw, I get the autoconvolution values.
    
    dw = w / (M // N)  # width of each fine interval
    autoconv = conv * dw
    
    max_autoconv = np.max(autoconv)
    
    return max_autoconv / integral_f

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin
    
    # Strategy: Use cosine series parameterization + iterative optimization
    # f(x) = a0 + sum_{k=1}^{K} a_k * cos(2*pi*k*x / L) where L = 1/2
    # On the interval [-1/4, 1/4], this becomes a0 + sum a_k cos(4*pi*k*x)
    
    # The step function heights are samples of this cosine series at interval centers.
    # We want to minimize max(f*f) / integral(f)
    
    # Key insight: to minimize the peak of autoconvolution while keeping integral fixed,
    # we want to spread the mass as evenly as possible, but with oscillations that
    # reduce the peak.
    
    # Let's parameterize: h_i = a0 + sum_{k=1}^{K} a_k cos(2*pi*k*i/N)
    # (using discrete cosine transform basis)
    
    # We'll optimize the coefficients a_k to minimize the score.
    
    best_heights = [1.0] * N
    best_score = _autoconv_max(best_heights)
    
    K = 20  # number of cosine modes
    
    # Initial guess: uniform + small random perturbations
    np.random.seed(42)
    
    def score_from_coeffs(a0, coeffs):
        """Compute score from cosine coefficients."""
        if time.time() > deadline:
            return None
        # Generate heights
        i = np.arange(N)
        h = a0 + np.zeros(N)
        for k in range(1, K + 1):
            h += coeffs[k-1] * np.cos(2 * np.pi * k * i / N)
        # Ensure nonnegativity
        h = np.maximum(h, 0)
        # Normalize so that integral is fixed (or just compute the ratio)
        return _autoconv_max(h)
    
    # Use scipy's minimize
    from scipy.optimize import minimize
    
    # Initial coefficients: small random values
    x0 = np.zeros(K + 1)
    x0[0] = 1.0  # a0
    x0[1:] = np.random.randn(K) * 0.1
    
    def objective(x):
        if time.time() > deadline:
            return best_score
        a0 = x[0]
        coeffs = x[1:]
        return score_from_coeffs(a0, coeffs)
    
    # Try multiple initializations
    n_restarts = 5
    for restart in range(n_restarts):
        if time.time() > deadline - 20:
            break
        
        if restart == 0:
            x_init = x0
        else:
            x_init = np.zeros(K + 1)
            x_init[0] = 1.0 + np.random.randn() * 0.3
            x_init[1:] = np.random.randn(K) * 0.5
        
        try:
            result = minimize(
                objective, x_init,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-12}
            )
            if result.fun < best_score:
                best_score = result.fun
                a0 = result.x[0]
                coeffs = result.x[1:]
                i = np.arange(N)
                h = a0 + np.zeros(N)
                for k in range(1, K + 1):
                    h += coeffs[k-1] * np.cos(2 * np.pi * k * i / N)
                h = np.maximum(h, 0)
                best_heights = list(h)
        except Exception as e:
            pass
    
    # Try a different approach: direct optimization with more modes
    if time.time() < deadline - 30:
        K2 = 40
        i = np.arange(N)
        cos_basis = np.array([np.cos(2 * np.pi * k * i / N) for k in range(1, K2 + 1)]).T
        
        def objective2(x):
            if time.time() > deadline:
                return best_score
            a0 = x[0]
            coeffs = x[1:]
            h = a0 + cos_basis @ coeffs
            h = np.maximum(h, 0)
            return _autoconv_max(h)
        
        for restart in range(3):
            if time.time() > deadline - 15:
                break
            
            x_init = np.zeros(K2 + 1)
            x_init[0] = 1.0
            x_init[1:] = np.random.randn(K2) * (0.3 if restart == 0 else 0.8)
            
            try:
                result = minimize(
                    objective2, x_init,
                    method='L-BFGS-B',
                    options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-12}
                )
                if result.fun < best_score:
                    best_score = result.fun
                    a0 = result.x[0]
                    coeffs = result.x[1:]
                    h = a0 + cos_basis @ coeffs
                    h = np.maximum(h, 0)
                    best_heights = list(h)
            except Exception:
                pass
    
    # Simulated annealing refinement
    if time.time() < deadline - 10:
        h = np.array(best_heights, dtype=float)
        current_score = best_score
        T = 0.01
        T_min = 0.001
        alpha = 0.999
        
        for step in range(5000):
            if time.time() > deadline - 5:
                break
            
            # Random perturbation
            idx = np.random.randint(0, N)
            delta = np.random.randn() * 0.05
            new_h = h.copy()
            new_h[idx] = max(0, h[idx] + delta)
            
            new_score = _autoconv_max(new_h)
            
            if new_score < current_score or np.random.rand() < np.exp((current_score - new_score) / T):
                h = new_h
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_heights = list(h)
            
            T *= alpha
            if T < T_min:
                break
    
    return best_heights

if __name__ == "__main__":
    result = construct()
    print(f"Score: {_autoconv_max(result)}")
