import numpy as np
from scipy.fft import fft, ifft

N = 600
# Domain is [-1/4, 1/4], so step size h = (0.5) / 600
h = 0.5 / N
x = np.linspace(-0.25, 0.25, N, endpoint=False) + h/2 # Centers of the intervals

def compute_score(f):
    # f is a list or array of nonnegative values
    f = np.array(f, dtype=float)
    if np.max(f) == 0:
        return float('inf')
    # Normalize so max is 1 to match the "normalized maximum" definition if needed, 
    # but the score definition is max(f*f) / (max(f) * int(f)).
    # Actually, the score is independent of scaling if we define it as 
    # max(f*f) / (max(f) * sum(f)*h).
    # Let's stick to the definition: 
    # S = max(f * f) / (max(f) * integral(f))
    # Where f*f is autoconvolution.
    
    # Autoconvolution via FFT
    # Zero pad to avoid circular convolution artifacts
    size = N * 2
    F = fft(f, n=size)
    conv = np.real(ifft(F * F, n=size))
    
    max_conv = np.max(conv)
    max_f = np.max(f)
    integral_f = np.sum(f) * h
    
    if max_f == 0 or integral_f == 0:
        return float('inf')
        
    score = max_conv / (max_f * integral_f)
    return score

# EVOLVE-BLOCK-START
def construct():
    # We will search for the best shape among a few analytic families
    # We want to minimize the score.
    
    best_score = float('inf')
    best_f = [1.0] * N # Default to seed behavior (boxcar)
    
    # 1. Boxcar of varying widths
    # The boxcar score is always 1.0 if it's a perfect rectangle.
    # Let's verify:
    # f = 1 on [0, W]. conv peak = W. max_f = 1. int = W. Score = 1.
    # So any boxcar gives 1.0.
    
    # 2. Triangular functions
    # f(x) = max(0, 1 - |x|/a)
    # Let's try different widths 'a'
    
    # 3. Gaussian functions
    # f(x) = exp(-(x/sigma)^2)
    
    # 4. Raised Cosine
    # f(x) = cos^2(pi x / (2a)) for |x|<a, 0 else
    
    shapes = []
    
    # Generate Triangles
    for a in np.linspace(0.05, 0.25, 50):
        f = np.maximum(0, 1 - np.abs(x)/a)
        shapes.append(f)
        
    # Generate Gaussians
    for sigma in np.linspace(0.02, 0.15, 50):
        f = np.exp(-(x/sigma)**2)
        shapes.append(f)
        
    # Generate Raised Cosines
    for a in np.linspace(0.05, 0.25, 50):
        f = np.zeros(N)
        mask = np.abs(x) < a
        f[mask] = np.cos(np.pi * x[mask] / (2*a))**2
        shapes.append(f)
        
    # Generate "Bump" functions (higher order polynomial, e.g., (1-x^2)^k)
    for k in [2, 3, 4, 5]:
        for a in np.linspace(0.1, 0.25, 20):
            f = np.zeros(N)
            mask = np.abs(x) < a
            xm = x[mask] / a
            f[mask] = np.maximum(0, 1 - xm**2)**k
            shapes.append(f)
            
    # Evaluate all
    for f in shapes:
        # Ensure non-negative
        f = np.maximum(0, f)
        # Discretize: f is defined at x centers. 
        # The problem asks for "step heights on equal intervals".
        # The value of the step function on interval i is f_i.
        # The integral is sum(f_i) * h.
        # The autoconvolution of a step function is a piecewise constant function.
        # Using FFT on the values approximates the convolution of the impulse train?
        # No, it approximates the convolution of the continuous function formed by these steps.
        # Since N=600 is small, and the function is smooth, this is a good approximation.
        # However, strictly, a step function f(t) = c_i on [x_i, x_{i+1}].
        # The convolution (f*f)(t) = \int f(u)f(t-u) du.
        # Discretized: sum_j c_j c_k h^2 ...
        # The FFT approach on the array `f` treats them as samples of a function.
        # Let's refine the score calculation to be more robust to the "step" nature.
        
        score = compute_score(f)
        if score < best_score:
            best_score = score
            best_f = f.copy()
            
    # Return the best found as a list of floats
    return [float(v) for v in best_f]
# EVOLVE-BLOCK-END
