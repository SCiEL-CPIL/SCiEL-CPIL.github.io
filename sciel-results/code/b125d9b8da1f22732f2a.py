"""
Strategy: Sparse Spectral Optimization via IDFT.

The previous strategy optimized the 400 heights directly (or via a 2-component sum),
which is a high-dimensional problem (400 DOF) prone to local minima when using
surrogate max functions.

This new strategy exploits the fact that the autoconvolution in the time domain
corresponds to the product of the spectra in the frequency domain.
By representing the step heights $h$ as the Inverse Discrete Fourier Transform (IDFT)
of a sparse spectrum $H_k$, we reduce the dimensionality of the optimization problem.

Let $h = \text{IDFT}(H)$.
We constrain $H$ to be real-valued (so $h$ is real) and symmetric ($H_k = H_{N-k}^*$).
We select a small set of frequency indices $K$ (e.g., 20-40 bins) and optimize
the complex amplitudes $A_k, B_k$ for these bins.

The objective is to minimize $V = |2 N \max(\text{conv}(h, h)) / (\sum h)^2|$.
Since $V$ is scale-invariant, we can normalize $h$ such that $\sum h = 1$.
Then $V = 2 N \max(\text{conv}(h, h))$.

We use a smooth surrogate for the max function (LogSumExp) and optimize the
sparse spectral coefficients using L-BFGS-B.
The sparsity ensures that the resulting $h$ has a smooth, structured profile
dominated by a few Fourier modes, which is known to produce low-peak
autoconvolutions (similar to Barker or Golay-like properties but continuous).
"""

import numpy as np
import time
from scipy.optimize import minimize

N = 400
_DEADLINE = 140.0  # Safety margin before 150s limit

def _evaluate(h):
    """Compute the exact task value."""
    s = np.sum(h)
    if s == 0 or not np.isfinite(s):
        return np.inf
    conv = np.convolve(h, h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def _surrogate_max(x, alpha):
    """Smooth approximation of max(x)."""
    x_max = np.max(x)
    if not np.isfinite(x_max):
        return x_max
    x_shifted = (x - x_max) / alpha
    x_shifted = np.clip(x_shifted, -100, 100)
    log_sum_exp = np.log(np.sum(np.exp(x_shifted)))
    return x_max + alpha * log_sum_exp

def _construct_from_spectrum(coeffs, freq_indices):
    """
    Reconstruct h from sparse spectral coefficients.
    coeffs: complex array of length len(freq_indices)
    freq_indices: array of frequency bin indices
    """
    H = np.zeros(N, dtype=np.complex128)
    # DC component (index 0) must be real
    # Other components come in conjugate pairs to ensure h is real
    # We define the spectrum by specifying positive frequency bins.
    # If we pick indices k, we must also set N-k.
    
    # To ensure h is real, we enforce H[k] = conj(H[N-k]).
    # We will optimize the complex values for a set of unique positive frequencies.
    # Let's say we pick indices from 1 to N//2.
    # The DC term (index 0) is separate.
    
    # The input 'coeffs' is structured as:
    # coeffs[0] is the real DC amplitude.
    # coeffs[1:] are complex amplitudes for the selected positive frequency bins.
    
    dc_val = coeffs[0]
    H[0] = dc_val
    
    for i, k in enumerate(freq_indices):
        val = coeffs[i + 1]
        H[k] = val
        H[N - k] = np.conj(val)
        
    h = np.real(np.fft.ifft(H))
    return h

def _objective(params, freq_indices, alpha):
    """
    Objective function.
    params: 
      params[0]: real DC component
      params[1:]: real and imag parts of complex coefficients for freq_indices
    """
    n_freq = len(freq_indices)
    # Unpack params
    dc = params[0]
    complex_coeffs = np.zeros(n_freq, dtype=np.complex128)
    for i in range(n_freq):
        re = params[1 + 2*i]
        im = params[2 + 2*i]
        complex_coeffs[i] = re + 1j*im
        
    full_coeffs = np.concatenate(([dc], complex_coeffs))
    
    h = _construct_from_spectrum(full_coeffs, freq_indices)
    
    s_sum = np.sum(h)
    if abs(s_sum) < 1e-6:
        return 1e10
        
    # Normalize h so sum(h) = 1 to fix scale
    h_norm = h / s_sum
    
    conv = np.convolve(h_norm, h_norm)
    
    # We want to minimize the maximum peak.
    # The lag 0 term is sum(h_norm^2).
    # If h is uniform, sum(h^2) = 1/N. max(conv) >= 1/N.
    # V = 2 * N * max(conv).
    
    obj = _surrogate_max(conv, alpha)
    
    # Regularization to prevent coefficients from exploding
    # and to encourage a "sparse" or "smooth" solution in the time domain?
    # Actually, we just want low peaks.
    # Add a small penalty on the magnitude of coefficients to keep them bounded
    # relative to the DC term, preventing numerical issues.
    reg = 1e-8 * np.sum(np.abs(params)**2)
    
    return obj + reg

def construct():
    """
    Construct step heights using Sparse Spectral Optimization.
    """
    t0 = time.time()
    deadline = t0 + _DEADLINE
    
    best_val = _evaluate(np.ones(N))
    best_h = np.ones(N).copy()
    
    # Select a set of frequency indices to optimize.
    # Low frequencies tend to produce smooth h, which has lower peak autoconvolution
    # compared to random noise.
    # We will try different sets of frequencies.
    
    # Strategy: Use the first K low-frequency bins (excluding DC).
    # K around 10-30 seems reasonable.
    
    n_restarts = 15
    rng = np.random.default_rng(42)
    
    # List of frequency sets to try
    # Set 1: First 10 bins
    # Set 2: First 20 bins
    # Set 3: Random selection of 15 bins from 1..N/2
    # Set 4: Random selection of 25 bins
    
    freq_sets = []
    freq_sets.append(np.arange(1, 11))
    freq_sets.append(np.arange(1, 21))
    
    # Add some random sparse sets
    for _ in range(3):
        k = rng.integers(10, 30)
        indices = rng.choice(np.arange(1, N//2), size=k, replace=False)
        indices.sort()
        freq_sets.append(indices)
    
    # Alpha annealing
    alpha_start = 0.2
    alpha_end = 0.01
    
    for i in range(n_restarts):
        if time.time() > deadline - 15:
            break
            
        # Pick a frequency set
        # Cycle through them, with some randomization in later restarts
        if i < len(freq_sets):
            freq_indices = freq_sets[i]
        else:
            # Random sparse set
            k = rng.integers(10, 40)
            indices = rng.choice(np.arange(1, N//2), size=k, replace=False)
            indices.sort()
            freq_indices = indices
            
        n_freq = len(freq_indices)
        # Total params: 1 (DC) + 2 * n_freq (Re, Im for each)
        param_size = 1 + 2 * n_freq
        
        # Initialize parameters
        # DC should be positive and large enough to keep sum(h) away from 0.
        # If DC is large, h is close to constant.
        # We want to find perturbations that reduce the peak.
        # Start with small perturbations around the constant solution (DC=1, others=0)
        # Or random small values.
        
        params = np.zeros(param_size)
        params[0] = 1.0  # DC
        
        # Add small random noise to break symmetry
        noise_scale = 0.1
        params[1:] = rng.normal(0, noise_scale, param_size - 1)
        
        # Anneal alpha
        alpha = alpha_start + (alpha_end - alpha_start) * (i / n_restarts)
        
        def obj_wrapper(p):
            return _objective(p, freq_indices, alpha)
        
        try:
            # Use L-BFGS-B
            # We might need to provide gradients for speed, but N=400 and small params
            # means finite difference is okay if param_size is small (< 50).
            # If param_size is large, we might be slow.
            # Let's stick to param_size < 60.
            
            res = minimize(
                obj_wrapper,
                params,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
            )
            
            if res.fun < 1e9:
                # Reconstruct h
                dc = res.x[0]
                complex_coeffs = np.zeros(n_freq, dtype=np.complex128)
                for j in range(n_freq):
                    re = res.x[1 + 2*j]
                    im = res.x[2 + 2*j]
                    complex_coeffs[j] = re + 1j*im
                    
                full_coeffs = np.concatenate(([dc], complex_coeffs))
                h_opt = _construct_from_spectrum(full_coeffs, freq_indices)
                
                s_sum = np.sum(h_opt)
                if abs(s_sum) > 1e-6:
                    h_norm = h_opt / s_sum
                    v_opt = _evaluate(h_norm)
                    if v_opt < best_val:
                        best_val = v_opt
                        best_h = h_norm.copy()
                        
        except Exception as e:
            pass

    # Final polish: Direct space refinement
    # This helps fix any issues and potentially improve further.
    if time.time() < deadline - 10:
        h_curr = best_h.copy()
        v_curr = best_val
        step_size = 0.05
        # Use a more aggressive local search
        for _ in range(2000):
            if time.time() > deadline - 2:
                break
            # Random coordinate descent with small steps
            idx = rng.integers(0, N)
            h_new = h_curr.copy()
            h_new[idx] += rng.normal(0, step_size)
            v_new = _evaluate(h_new)
            if v_new < v_curr:
                v_curr = v_new
                h_curr = h_new
                step_size *= 0.98
            else:
                step_size = min(step_size * 1.05, 0.2)
                
        if v_curr < best_val:
            best_val = v_curr
            best_h = h_curr

    if not np.all(np.isfinite(best_h)) or np.sum(best_h) == 0:
        return np.ones(N)
        
    return best_h

# EVOLVE-BLOCK-END
