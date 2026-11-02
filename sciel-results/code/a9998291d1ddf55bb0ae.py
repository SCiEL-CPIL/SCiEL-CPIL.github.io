"""Multi-start simulated annealing to minimize the normalized autoconvolution peak."""
import numpy as np
import time

N = 600

def compute_score(heights):
    """Compute the normalized maximum of the autoconvolution of the step function."""
    h = np.array(heights, dtype=np.float64)
    # Autoconvolution via FFT
    n_fft = 2 * N
    H = np.fft.rfft(h, n=n_fft)
    conv = np.fft.irfft(H * np.conj(H), n=n_fft)
    # The autoconvolution of a step function: the peak is at the center
    # conv[k] for k in 0..N-1 gives the discrete autoconvolution
    # The continuous autoconvolution peak for step functions is at the center
    # For a step function, the autoconvolution at the center is sum of h_i^2 weighted
    # Actually, let's just use the discrete autoconvolution peak
    peak = np.max(conv[:N])
    # Normalize: for uniform h=1, conv[0] = sum(h^2) = N, and the "normalized" value
    # The task says uniform gives 1.0, so normalization must make uniform = 1.0
    # For uniform h=1: conv[0] = N = 600, and we need score = 1.0
    # So normalization is peak / N? Let's check: 600/600 = 1.0. Yes!
    return peak / N

def construct():
    deadline = time.time() + 130  # Safety margin before 150s limit
    
    best_heights = [1.0] * N
    best_score = compute_score(best_heights)
    
    rng = np.random.default_rng(42)
    
    # Multi-start simulated annealing
    n_starts = 8
    t_start = time.time()
    
    for start in range(n_starts):
        if time.time() > deadline:
            break
        
        # Initialize with different starting points
        if start == 0:
            x = np.ones(N)
        elif start == 1:
            # Gaussian bump
            x = 1.0 + 0.3 * np.exp(-((np.arange(N) - N/2) / (N/6))**2)
            x = np.maximum(x, 0)
        elif start == 2:
            # Inverted bump
            x = 1.0 - 0.3 * np.exp(-((np.arange(N) - N/2) / (N/6))**2)
            x = np.maximum(x, 0)
        elif start == 3:
            # Random
            x = np.abs(rng.normal(1.0, 0.5, N))
        elif start == 4:
            # Cosine
            x = 0.5 + 0.5 * np.cos(np.pi * np.arange(N) / N)
            x = np.maximum(x, 0.01)
        elif start == 5:
            # Two bumps
            x = 1.0 + 0.4 * (np.exp(-((np.arange(N) - N/4) / (N/10))**2) + 
                              np.exp(-((np.arange(N) - 3*N/4) / (N/10))**2))
            x = np.maximum(x, 0)
        elif start == 6:
            # Ramp
            x = np.linspace(0.5, 1.5, N)
            x = np.maximum(x, 0)
        else:
            # Random with smoothing
            x = np.abs(rng.normal(1.0, 0.3, N))
            # Smooth
            kernel = np.ones(5) / 5
            x = np.convolve(x, kernel, mode='same')
            x = np.maximum(x, 0)
        
        current_score = compute_score(x)
        if current_score < best_score:
            best_score = current_score
            best_heights = x.tolist()
        
        # Simulated annealing
        T = 0.01
        T_min = 1e-6
        cooling = 0.995
        n_steps = 5000
        
        for step in range(n_steps):
            if time.time() > deadline:
                break
            
            # Perturbation: Gaussian noise on a random subset
            n_perturb = rng.integers(5, 50)
            indices = rng.choice(N, size=n_perturb, replace=False)
            noise = rng.normal(0, 0.1, size=n_perturb)
            
            x_new = x.copy()
            x_new[indices] += noise
            x_new = np.maximum(x_new, 0)
            
            new_score = compute_score(x_new)
            
            if new_score < current_score or rng.random() < np.exp(-(new_score - current_score) / T):
                x = x_new
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_heights = x.tolist()
            
            T = max(T * cooling, T_min)
    
    return best_heights
