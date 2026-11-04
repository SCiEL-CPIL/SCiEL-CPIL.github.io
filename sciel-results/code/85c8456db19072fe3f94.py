"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    # Initialize with a random signal
    rng = np.random.default_rng(42)
    f = rng.normal(0, 1, N)
    
    # Normalize to unit L2 norm
    f = f / np.linalg.norm(f)
    
    # Gradient descent to minimize L3 norm of autocorrelation
    # Objective: sum(|R|^3) where R is circular autocorrelation
    
    def compute_autocorr(f):
        F = np.fft.fft(f)
        power = F * np.conj(F)
        R = np.real(np.fft.ifft(power))
        return R

    # We want to minimize E = sum(R^3) assuming R > 0? No, R can be negative.
    # The task says "signed step function".
    # "Upper bound on C3". Usually C3 involves the integral of the cube of the autocorrelation.
    # If we minimize the L3 norm of the autocorrelation, we reduce the bound.
    
    # Gradient of E = sum(|R|^3) w.r.t f
    # dE/df(t) = sum_tau 3 |R(tau)|^2 sign(R(tau)) (f(t+tau) + f(t-tau))
    # This is equivalent to 3 * convolve( W, f ) where W(tau) = |R(tau)|^2 sign(R(tau))
    # Note: The sum over tau of f(t+tau) is a convolution.
    
    lr = 0.1
    steps = 200
    
    for i in range(steps):
        R = compute_autocorr(f)
        # Weight function for gradient
        W = 3 * np.abs(R)**2 * np.sign(R)
        
        # Gradient is convolution of W and f
        # Circular convolution
        F = np.fft.fft(f)
        W_hat = np.fft.fft(W)
        grad_hat = W_hat * F
        grad = np.real(np.fft.ifft(grad_hat))
        
        # Update
        f = f - lr * grad
        
        # Renormalize to keep L2 norm constant (scale invariance)
        norm = np.linalg.norm(f)
        if norm > 1e-10:
            f = f / norm
            
        # Optional: Threshold or project? No, keep continuous for now.
        # If "step function" implies discrete levels, we might need to project.
        # The task says "signed step heights". It doesn't restrict to binary.
        
    return f
# EVOLVE-BLOCK-END
