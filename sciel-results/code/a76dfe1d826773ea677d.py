"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

N = 400

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution


def _autoconvolve(x):
    """Compute full autoconvolution (linear) of x."""
    # Use FFT for efficiency
    n = len(x)
    size = 2 * n - 1
    # Pad to next power of 2 for FFT efficiency
    fft_size = 1
    while fft_size < size:
        fft_size *= 2
    X = np.fft.rfft(x, n=fft_size)
    ac = np.fft.irfft(X * X, n=fft_size)[:size]
    return ac


def _score(x):
    """
    Compute the score: maximum absolute autoconvolution divided by squared integral.
    Lower is better.
    """
    ac = _autoconvolve(x)
    max_abs_ac = np.max(np.abs(ac))
    integral = np.sum(x)
    sq_integral = integral * integral
    if abs(sq_integral) < 1e-12:
        return float('inf')
    return max_abs_ac / sq_integral


def _construct_zadoff_chu_like(chirp_param):
    """
    Construct a signed step function from a Zadoff-Chu-like sequence with chirp parameter.
    The sequence is defined as the sign of the real part of a complex chirp.
    """
    n = N
    t = np.arange(n)
    # Zadoff-Chu-like: exp(i * chirp_param * t^2)
    # We take the real part and threshold to get signed values (+1 or -1)
    phase = chirp_param * t * t
    real_part = np.cos(phase)
    # Sign function, with 0 mapped to 1
    x = np.sign(real_part)
    x[x == 0] = 1.0
    return x


def construct():
    """
    Main entry point. Returns a sequence of signed step heights (array of +1/-1).
    """
    deadline = time.time() + 130  # Safety margin: 120s limit, keep 10s buffer
    
    best_x = np.ones(N)
    best_score = _score(best_x)
    
    # Strategy 1: Optimize chirp parameter for Zadoff-Chu-like sequence
    # The chirp parameter determines the quadratic phase, and we want to minimize
    # the max |autoconvolution| / (sum)^2
    
    # First, do a coarse grid search over chirp parameters
    # The chirp parameter alpha in exp(i * alpha * t^2)
    # For good properties, alpha should be around pi / N or similar
    # Let's search over a range
    
    t = np.arange(N)
    
    def chirp_score(alpha):
        if time.time() > deadline - 5:
            return float('inf')
        phase = alpha * t * t
        real_part = np.cos(phase)
        x = np.sign(real_part)
        x[x == 0] = 1.0
        return _score(x)
    
    # Coarse search
    alpha_range = np.linspace(0.01, 0.5, 50)
    best_alpha = None
    best_chirp_score = float('inf')
    
    for a in alpha_range:
        if time.time() > deadline - 10:
            break
        s = chirp_score(a)
        if s < best_chirp_score:
            best_chirp_score = s
            best_alpha = a
    
    if best_alpha is not None and best_chirp_score < best_score:
        x = _construct_zadoff_chu_like(best_alpha)
        best_x = x
        best_score = best_chirp_score
    
    # Refine with finer search around best alpha
    if best_alpha is not None and time.time() < deadline - 20:
        alpha_fine = np.linspace(best_alpha - 0.02, best_alpha + 0.02, 100)
        for a in alpha_fine:
            if time.time() > deadline - 10:
                break
            s = chirp_score(a)
            if s < best_chirp_score:
                best_chirp_score = s
                best_alpha = a
        
        if best_chirp_score < best_score:
            x = _construct_zadoff_chu_like(best_alpha)
            best_x = x
            best_score = best_chirp_score
    
    # Strategy 2: Local search / hill climbing on the binary sequence
    # Try flipping individual bits to improve the score
    if time.time() < deadline - 15:
        x = best_x.copy()
        current_score = best_score
        improved = True
        max_iterations = 200000
        iteration = 0
        
        while improved and time.time() < deadline - 5 and iteration < max_iterations:
            improved = False
            iteration += 1
            
            # Try random flips
            n_trials = 100
            for _ in range(n_trials):
                if time.time() > deadline - 5:
                    break
                idx = np.random.randint(0, N)
                old_val = x[idx]
                x[idx] = -old_val
                
                new_score = _score(x)
                
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                    if current_score < best_score:
                        best_score = current_score
                        best_x = x.copy()
                else:
                    x[idx] = old_val
        
        # Simulated annealing style: allow some worse moves
        if time.time() < deadline - 10:
            x = best_x.copy()
            current_score = best_score
            temperature = 0.1
            cooling = 0.999
            max_sa_iters = 50000
            sa_iter = 0
            
            while sa_iter < max_sa_iters and time.time() < deadline - 5:
                idx = np.random.randint(0, N)
                old_val = x[idx]
                x[idx] = -old_val
                
                new_score = _score(x)
                delta = new_score - current_score
                
                if delta < 0 or np.random.random() < np.exp(-delta / (temperature + 1e-12)):
                    current_score = new_score
                    if current_score < best_score:
                        best_score = current_score
                        best_x = x.copy()
                else:
                    x[idx] = old_val
                
                temperature *= cooling
                sa_iter += 1
    
    return best_x
# EVOLVE-BLOCK-END
