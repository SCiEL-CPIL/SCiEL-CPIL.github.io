"""Program for the suite task; designs a phase-coded signal with smooth envelope to minimize autoconvolution peak."""
import numpy as np
from scipy.optimize import minimize

N = 600

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a nonnegative step function as the squared magnitude of a phase-coded signal
    with a smooth envelope, optimized to minimize the normalized maximum of its autoconvolution.
    
    The approach:
    - Represent f(n) = |g(n)|^2 * envelope(n) where g(n) is a sum of complex exponentials
      (frequency hops) with optimized phases and amplitudes.
    - Use a Hann envelope to taper edges and reduce spectral leakage.
    - Optimize the discrete phase jumps and amplitudes via simulated annealing to exploit
      low auto-correlation properties of pseudo-random sequences.
    """
    
    def compute_f(params):
        """Compute the step function heights from parameters."""
        K = 5  # Number of frequency components
        amplitudes = params[:K]
        phases = params[K:2*K]
        
        # Ensure amplitudes are nonnegative
        amplitudes = np.maximum(amplitudes, 0)
        
        # Create the base signal as sum of complex exponentials
        n = np.arange(N)
        g = np.zeros(N, dtype=complex)
        
        for k in range(K):
            freq = (k + 1) * 0.01  # Small frequency hops
            g += amplitudes[k] * np.exp(1j * (phases[k] + 2 * np.pi * freq * n))
        
        # Step function heights are the squared magnitude
        f = np.abs(g) ** 2
        
        # Apply a smooth envelope to taper edges (Hann window)
        envelope = 0.5 * (1 + np.cos(2 * np.pi * n / (N - 1)))
        f = f * envelope
        
        return f
    
    def compute_score(params):
        """Compute the score (normalized maximum of autoconvolution)."""
        f = compute_f(params)
        
        # Normalize f so that sum(f) = 1
        total = np.sum(f)
        if total < 1e-10:
            return 1e10  # Penalize near-zero functions
        
        f_norm = f / total
        
        # Compute autoconvolution using FFT for efficiency
        n_fft = 2 * N
        f_fft = np.fft.fft(f_norm, n=n_fft)
        autoconv = np.fft.ifft(f_fft * np.conj(f_fft)).real
        autoconv = np.abs(autoconv)  # Take absolute value for safety
        
        # The peak is at lag 0 (index 0)
        peak = np.max(autoconv)
        
        return peak
    
    # Simulated annealing optimization
    rng = np.random.RandomState(42)
    
    # Initial parameters: K amplitudes (equal) and K phases (random)
    K = 5
    init_params = np.concatenate([
        np.ones(K) / K,  # Equal amplitudes
        rng.uniform(0, 2 * np.pi, K)  # Random phases
    ])
    
    best_params = init_params.copy()
    best_score = compute_score(init_params)
    
    # Simulated annealing parameters
    T = 1.0
    T_min = 1e-6
    alpha = 0.999
    iterations = 3000
    
    current_params = init_params.copy()
    current_score = best_score
    
    for i in range(iterations):
        # Generate perturbation
        perturbation = rng.normal(0, 0.1, size=2*K)
        perturbation[:K] *= 0.5  # Smaller perturbations for amplitudes
        new_params = current_params + perturbation
        
        # Ensure amplitudes are nonnegative
        new_params[:K] = np.maximum(new_params[:K], 0)
        
        new_score = compute_score(new_params)
        
        # Acceptance criterion
        delta = new_score - current_score
        if delta < 0 or rng.random() < np.exp(-delta / T):
            current_params = new_params
            current_score = new_score
            
            if current_score < best_score:
                best_score = current_score
                best_params = current_params.copy()
        
        # Cool down
        T *= alpha
        if T < T_min:
            T = T_min
    
    # Get optimal parameters
    optimal_params = best_params
    
    # Compute final step function
    f = compute_f(optimal_params)
    
    # Ensure nonnegative (should already be)
    f = np.maximum(f, 0)
    
    return list(f)
# EVOLVE-BLOCK-END
