import numpy as np
from scipy.optimize import minimize

N = 400

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    # Initialize with random signs or random continuous values
    # We want to flatten the spectrum, which minimizes sidelobes.
    # We use gradient descent on x in R^N.
    
    x = np.random.randn(N)
    
    def objective(params):
        x = params
        # Compute autocorrelation
        # Using FFT for speed
        X = np.fft.fft(x)
        power = np.abs(X)**2
        # Inverse FFT of power gives circular autocorrelation
        c = np.fft.ifft(power).real
        # c[0] is energy. c[k] for k!=0 are sidelobes.
        # We want to minimize the max absolute value of sidelobes.
        # Or sum of squares. Max is non-smooth. Sum of squares is smooth.
        # Let's minimize sum of squared sidelobes.
        # Exclude c[0]
        sidelobes = np.concatenate([c[1:], c[:0]]) # c[0] is at index 0. 
        # Note: ifft returns c[0], c[1], ..., c[N-1].
        # c[N-1] is the wrap-around correlation (same as c[-1]).
        # So we exclude c[0].
        return np.sum(sidelobes**2)
        
    # Simple gradient descent
    # Since N=400, FFT is fast.
    # We can use scipy minimize for robustness, or manual GD.
    # Manual GD with adaptive step size.
    
    lr = 0.01
    for i in range(1000):
        # Compute gradient
        # grad of sum(c_k^2) wrt x.
        # c = ifft(|fft(x)|^2).
        # This is equivalent to convolving x with reversed x.
        # Let's use numerical gradient or analytical.
        # Analytical:
        # c_k = sum x_i x_{i+k}.
        # d/dc_k (c_k^2) = 2 c_k.
        # d/dx_j (c_k^2) = 2 c_k (x_{j+k} + x_{j-k}).
        # Sum over k != 0.
        
        X = np.fft.fft(x)
        power = np.abs(X)**2
        c = np.fft.ifft(power).real
        c[0] = 0 # Ignore main lobe
        
        # Gradient calculation
        # grad[j] = sum_{k != 0} 2 c_k (x_{j+k} + x_{j-k})
        # This is a convolution of c with x?
        # Let's just use a simple finite difference or scipy.
        # Given 120s limit, 1000 iterations of FFT (N=400) is very fast.
        
        # To save time, let's just do a few iterations of GD.
        # Or use scipy.optimize.minimize with method='BFGS' or 'L-BFGS-B'.
        # L-BFGS-B is good for large N.
        
        pass # Placeholder

    # Actually, let's just use a random search or a specific known sequence?
    # No, directive says gradient based.
    
    # Let's implement a simple GD loop.
    x = np.random.randn(N)
    for step in range(500):
        X = np.fft.fft(x)
        power = np.abs(X)**2
        c = np.fft.ifft(power).real
        c[0] = 0
        
        # Compute gradient analytically
        # grad = 2 * (c * x) in circular convolution sense?
        # c_k = sum_i x_i x_{i+k}
        # d/dx_j sum_k c_k^2 = 2 sum_k c_k (x_{j+k} + x_{j-k})
        # Let h_m = c_{-m} = c_m (real, even).
        # Term is sum_k c_k x_{j+k} + sum_k c_k x_{j-k}
        # Let l = j+k -> k = l-j. Sum_l c_{l-j} x_l. This is conv(c, x).
        # Second term: let m = j-k -> k = j-m. Sum_m c_{j-m} x_m. This is conv(c, x) as well?
        # Yes, circular convolution of c and x.
        # So grad = 2 * ifft(fft(c) * fft(x)).real
        
        grad = 2.0 * np.fft.ifft(np.fft.fft(c) * np.fft.fft(x)).real
        
        # Update
        x -= 0.01 * grad
        
    # Normalize to unit energy or something? 
    # The score is scale invariant. 
    # But we should ensure it's not all zeros.
    if np.linalg.norm(x) < 1e-6:
        x = np.random.randn(N)
        
    return x
# EVOLVE-BLOCK-END
