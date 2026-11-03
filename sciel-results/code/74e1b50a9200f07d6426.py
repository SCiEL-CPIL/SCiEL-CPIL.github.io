import numpy as np
import time
from scipy.optimize import minimize
import sys

N = 600
# Grid on [-1/4, 1/4]
x = np.linspace(-0.25, 0.25, N)
dx = x[1] - x[0]

def compute_score(f):
    # f is a numpy array of length N
    # Compute autoconvolution using FFT
    # Zero-pad to avoid circular convolution effects if necessary, 
    # but standard linear convolution of length N with N has length 2N-1.
    # The problem asks for autoconvolution. Usually implies linear.
    # However, if we use FFT of size 2N, we get linear convolution.
    F = np.fft.fft(f, n=2*N)
    H = np.fft.ifft(F * F)
    # H is complex due to numerical errors, take real part
    H = H.real
    # The linear convolution is centered. 
    # For even/odd lengths, the peak is at index N-1 or N.
    # We want the maximum value of the convolution.
    max_val = np.max(H)
    
    # Normalization: sum of squares of f
    norm = np.sum(f**2)
    if norm < 1e-12:
        return 1e10
    return max_val / norm

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    best_score = 1.0 # Seed score
    best_f = np.ones(N)
    
    # Strategy: f(x) = |g(x)|^2 where g(x) = sum c_k exp(i w_k x)
    # We optimize c_k (real, imag) and w_k.
    # Let's try K=3 or K=4 exponentials.
    
    def objective(params):
        # params: [c1_r, c1_i, w1, c2_r, c2_i, w2, ...]
        K = 3
        g = np.zeros(N, dtype=complex)
        for k in range(K):
            cr = params[3*k]
            ci = params[3*k+1]
            w = params[3*k+2]
            c = cr + 1j*ci
            g += c * np.exp(1j * w * x)
        f = np.abs(g)**2
        return compute_score(f)
        
    def objective_and_f(params):
        K = 3
        g = np.zeros(N, dtype=complex)
        for k in range(K):
            cr = params[3*k]
            ci = params[3*k+1]
            w = params[3*k+2]
            c = cr + 1j*ci
            g += c * np.exp(1j * w * x)
        f = np.abs(g)**2
        return compute_score(f), f

    # Initial guesses
    # We want to spread the energy. 
    # Uniform f=1 corresponds to g having only DC component? No, |c0|^2 = 1.
    # If we add other frequencies, we create ripples.
    # Maybe we want a "window" shape.
    
    # Let's try random restarts with different K and initial conditions.
    
    max_restarts = 20
    restarts_done = 0
    
    while time.time() < deadline and restarts_done < max_restarts:
        restarts_done += 1
        
        # Choose K
        K = np.random.choice([2, 3, 4, 5])
        
        # Initialize parameters
        # Frequencies should be small to keep the function smooth on the grid
        # x range is 0.5. w * x should not oscillate too fast.
        # Let's pick w from a small range, e.g., [-10, 10]
        
        params0 = []
        for k in range(K):
            cr = np.random.uniform(-1, 1)
            ci = np.random.uniform(-1, 1)
            w = np.random.uniform(-5, 5)
            params0.extend([cr, ci, w])
            
        # Ensure we don't start with zero energy
        if np.random.rand() < 0.5:
            params0[0] = 1.0 # Set first real coeff to 1
            
        res = minimize(objective, params0, method='Nelder-Mead', 
                       options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-4})
        
        if res.success:
            score, f = objective_and_f(res.x)
            if score < best_score:
                best_score = score
                best_f = f
        
        # Also try a simpler heuristic: Gaussian-like windows
        # f(x) = exp(-x^2 / sigma^2)
        # This is not strictly |sum c_k e^{i w_k x}|^2 with few terms, 
        # but we can fit it or just use it directly if the representation isn't strict.
        # The prompt says "Represent the step function as the squared magnitude...".
        # So we should stick to the representation.
        
    # Final check
    if time.time() < deadline - 5:
         # One more try with fixed frequencies and optimized coeffs?
         pass

    return list(best_f)

if __name__ == "__main__":
    # Quick self-test
    f = construct()
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")
    score = compute_score(np.array(f))
    print(f"Score: {score}")
