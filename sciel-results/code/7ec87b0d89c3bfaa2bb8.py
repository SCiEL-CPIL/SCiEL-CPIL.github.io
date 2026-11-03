"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 600

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a nonnegative step function with 600 equal intervals on [-1/4, 1/4]
    to minimize the normalized maximum of its autoconvolution.
    
    Strategy: Use spectral parameterization via DFT coefficients.
    We work in the frequency domain where the autoconvolution peak can be
    directly optimized by adjusting DFT coefficients while maintaining
    nonnegativity in the time domain.
    
    The autoconvolution of f with itself has maximum related to the L-infinity
    norm of the convolution. In frequency domain, convolution becomes multiplication,
    so |f*f| in time domain relates to |F|^2 in frequency domain.
    
    We use coordinate descent on the real and imaginary parts of the DFT coefficients
    to minimize the peak of the autoconvolution, while projecting back to ensure
    nonnegativity.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Initialize with a reasonable starting point
    # Start with a smooth bump-like function
    t = np.linspace(-0.25, 0.25, N, endpoint=False)
    # Initial guess: a raised cosine or similar smooth nonnegative function
    f = 1.0 + 0.5 * np.cos(2 * np.pi * t / 0.5)  # Period 0.5, centered
    f = np.maximum(f, 0.0)
    
    best_f = f.copy()
    best_score = np.inf
    
    def compute_score(f):
        """Compute the normalized maximum of autoconvolution."""
        # Autoconvolution: f * f
        # Using FFT for efficiency
        F = np.fft.fft(f)
        conv = np.fft.ifft(F * F).real
        # Normalize: divide by integral of f squared (or sum of f^2 * dt)
        # The task says "normalized maximum" - we interpret as max(conv) / (sum(f^2) * dt)
        # Actually, let's think about normalization more carefully.
        # The autoconvolution (f*f)(x) = integral f(y) f(x-y) dy
        # For step functions on equal intervals, this is a discrete convolution.
        # Normalization: likely max(f*f) / (integral f)^2 or similar.
        # Let's use: max(conv) / (sum(f) * dt)^2 as a reasonable normalization
        dt = 0.5 / N
        integral_f = np.sum(f) * dt
        if integral_f < 1e-12:
            return np.inf
        return np.max(conv) / (integral_f ** 2)
    
    best_score = compute_score(best_f)
    
    # Spectral coordinate descent
    # Work with DFT coefficients
    F = np.fft.fft(best_f).copy()
    
    # We'll do coordinate descent on F (complex coefficients)
    # For each coefficient, adjust to reduce the peak of autoconvolution
    # while maintaining nonnegativity of ifft(F)
    
    # Number of iterations based on time budget
    max_iters = 5000
    iters_per_check = 100
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Pick a random coefficient to update
        # Skip DC component (index 0) to preserve mean somewhat
        # Actually, let's allow all but handle carefully
        k = np.random.randint(0, N)
        
        old_Fk = F[k].copy()
        
        # Try small perturbations in real and imaginary parts
        best_Fk = old_Fk
        best_local_score = compute_score(np.fft.ifft(F).real)
        
        # We need to be careful: changing F[k] changes the time-domain signal
        # The autoconvolution peak is a non-smooth function, so we use
        # a stochastic approach: try random perturbations
        
        # Perturb real part
        for _ in range(3):
            delta_r = np.random.randn() * 0.1 * np.abs(old_Fk) if np.abs(old_Fk) > 1e-10 else np.random.randn() * 0.1
            F[k] = old_Fk + delta_r + 0j
            f_test = np.fft.ifft(F).real
            if np.min(f_test) < -1e-10:
                # Project to nonnegative
                f_test = np.maximum(f_test, 0.0)
                F[k] = np.fft.fft(f_test)[k]
            score = compute_score(f_test)
            if score < best_local_score:
                best_local_score = score
                best_Fk = F[k].copy()
        
        # Perturb imaginary part
        for _ in range(3):
            delta_i = np.random.randn() * 0.1 * np.abs(old_Fk) if np.abs(old_Fk) > 1e-10 else np.random.randn() * 0.1
            F[k] = old_Fk + 1j * delta_i
            f_test = np.fft.ifft(F).real
            if np.min(f_test) < -1e-10:
                f_test = np.maximum(f_test, 0.0)
                F[k] = np.fft.fft(f_test)[k]
            score = compute_score(f_test)
            if score < best_local_score:
                best_local_score = score
                best_Fk = F[k].copy()
        
        F[k] = best_Fk
        
        # Update best if improved
        f_current = np.fft.ifft(F).real
        f_current = np.maximum(f_current, 0.0)
        current_score = compute_score(f_current)
        if current_score < best_score:
            best_score = current_score
            best_f = f_current.copy()
        
        # Check time
        if (it + 1) % iters_per_check == 0:
            if time.time() > deadline:
                break
    
    # Final projection to ensure nonnegativity
    best_f = np.maximum(best_f, 0.0)
    
    return list(best_f)
# EVOLVE-BLOCK-END
