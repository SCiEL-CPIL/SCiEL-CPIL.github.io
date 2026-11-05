"""
Strategy: Frequency-Domain Spectral Shaping via IDFT.

This approach fundamentally changes the representation from time-domain step
heights to frequency-domain spectral coefficients.

Key Insights:
1. The autoconvolution $c = h * h$ is related to the power spectrum $|H|^2$
   via the Discrete Fourier Transform (DFT). Specifically, the DFT of the
   circular autoconvolution is $|H|^2$. The linear autoconvolution is a
   truncated version of the circular one, but for large N, the peak behavior
   is dominated by the low-frequency components of the spectrum.
2. The objective is to minimize $\max(\text{conv}(h, h))$ subject to
   $\sum h \neq 0$. Note that $\text{conv}(h, h)[0]$ is the DC component
   of the power spectrum scaled by N (roughly). Actually, $\sum h$ is $H[0]$.
   The term $\max(\text{conv}(h,h))$ is the peak of the autocorrelation.
   For a white noise signal, the peak is at lag 0. We want to flatten the
   autocorrelation, which corresponds to making the power spectrum $|H|^2$
   as flat as possible (constant magnitude across frequencies).
3. However, we are constrained to real-valued $h$, so $H$ must satisfy
   Hermitian symmetry.
4. We can parameterize $h$ by its DFT coefficients $H[k]$. To ensure $h$ is
   real, we optimize the magnitudes $|H[k]|$ and phases $\phi[k]$ for
   $k=0, \dots, N/2$, enforcing symmetry.
5. A simpler and more effective parameterization for real signals is to
   optimize the DFT coefficients directly as complex numbers, but constrain
   the optimization to the subspace of real-valued time signals by
   projecting back to real time domain after each update, or by using a
   real-valued parameterization like a low-frequency cosine/sine basis.

   Let's use a specific low-dimensional spectral parameterization:
   $h[n] = \sum_{k=0}^{K} A_k \cos(2\pi k n / N + \phi_k)$.
   This is equivalent to restricting the spectrum to the first K+1 bins.
   By choosing K small (e.g., 10-20), we have a low-dimensional optimization
   problem. The "flatness" of the spectrum in the unoptimized high-frequency
   bins will be zero, which might not be ideal, but the low-frequency
   structure can be tuned to reduce the peak.
   
   Actually, a better approach is to use a "soft" spectrum where we optimize
   the magnitudes of the first K bins, and set the rest to zero. This forces
   the signal to be band-limited. Band-limited signals have specific
   autocorrelation properties.
   
   Alternative: Optimize the full spectrum but use a regularizer to keep it
   smooth/flat. But full 400-dim optimization is what we want to avoid if
   possible, or do it in frequency domain where the objective might be
   smoother.
   
   Let's try: 
   1. Initialize $H$ with a flat magnitude (white noise) and random phases.
   2. Convert to $h$ via IDFT.
   3. Objective: $V(h) = \frac{2 N \max(\text{conv}(h,h))}{(\sum h)^2}$.
   4. We want to minimize this.
   5. Gradient descent on the spectral coefficients (real and imaginary parts
      for $k > 0$, real for $k=0$ and $k=N/2$).
   
   To make this efficient and differentiable, we can use the fact that
   $\text{conv}(h,h)$ is the inverse DFT of $|H|^2$ (circular conv).
   The linear conv is slightly different, but for large N, the difference is
   small in the peak region if the signal is well-behaved.
   However, the evaluator uses `np.convolve` (linear).
   
   Let's stick to time-domain evaluation for accuracy but use frequency-domain
   parameterization for the search.
   
   Parameterization:
   $h = \text{IDFT}(H)$
   Where $H$ is constructed from parameters $p$.
   Let's use a "Gaussian-like" spectrum in frequency domain.
   $H[k] = A \exp(-\alpha (k - k_0)^2) \exp(i \phi_k)$.
   This allows us to shape the spectral envelope.
   
   Even simpler: Use a small number of "spectral peaks".
   $H[k] = \sum_{j=1}^{M} A_j \exp(i \phi_j) \exp(-\sigma_j^2 (k - f_j)^2)$.
   This gives us $3M$ parameters.
   
   Let's implement this:
   1. Define $H$ from M Gaussian spectral peaks.
   2. Compute $h = \text{real}(\text{IDFT}(H))$.
   3. Evaluate objective.
   4. Optimize parameters using L-BFGS-B.
"""

import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 400
_DEADLINE = 145.0  # Safety margin for 150s limit


def _evaluate(h):
    """Compute the exact task value."""
    s = np.sum(h)
    if s == 0 or not np.isfinite(s):
        return np.inf
    conv = np.convolve(h, h)
    return abs(2.0 * N * np.max(conv) / (s * s))


def _h_from_spectral_peaks(params, M):
    """
    Construct h from M Gaussian spectral peaks.
    params: [A1, f1, s1, phi1, A2, f2, s2, phi2, ...]
    A: amplitude, f: center frequency (bin index 0..N/2), s: width (sigma), phi: phase
    """
    k = np.arange(N)
    H = np.zeros(N, dtype=complex)
    
    for j in range(M):
        A = params[4*j]
        f = params[4*j + 1]
        s = params[4*j + 2]
        phi = params[4*j + 3]
        
        # Gaussian in frequency domain
        # We need to handle the symmetry for real h.
        # It's easier to define H for k=0..N-1 and take real part of IDFT.
        # A real signal requires H[k] = conj(H[N-k]).
        # If we define a function g(k) that is symmetric, we are good.
        # Let's just add the Gaussian at f and N-f.
        
        # Gaussian centered at f
        g_f = A * np.exp(-((k - f)**2) / (2 * s**2)) * np.exp(1j * phi)
        H += g_f
        
        # Gaussian centered at N-f (mirror)
        # For real signal, we need H[N-k] = conj(H[k]).
        # If we add a Gaussian at N-f with the same amplitude and phase?
        # Let's check.
        # H[k] = A exp(-((k-f)^2)/2s^2) exp(i phi)
        # H[N-k] = A exp(-((N-k-f)^2)/2s^2) exp(i phi)
        # This is not necessarily conj(H[k]).
        # To ensure real h, we can just take the real part of the IDFT at the end.
        # h = real(IDFT(H)). This is equivalent to using a Hermitian spectrum.
        
    h = np.real(np.fft.ifft(H) * N) # np.fft.ifft has 1/N factor, so multiply by N to match DFT definition? 
    # Standard: H[k] = sum h[n] exp(-i 2pi kn/N). h[n] = 1/N sum H[k] exp(i 2pi kn/N).
    # np.fft.ifft does 1/N sum. So h = np.real(np.fft.ifft(H)).
    # Let's verify scaling.
    # If H is all 1s, ifft gives 1 at n=0, 0 elsewhere. 
    # My H is sum of Gaussians.
    # Let's just use np.fft.ifft(H) and take real part.
    h = np.real(np.fft.ifft(H))
    
    # Normalize to avoid huge/small values
    s = np.sum(h)
    if abs(s) < 1e-10:
        return np.zeros(N)
    return h / s * np.sqrt(N) # Scaling doesn't matter for the ratio, but helps optimization stability
    
    # Actually, the objective is scale invariant. 
    # value = abs(2 * N * max(conv) / sum(h)^2)
    # If h -> c*h, conv -> c^2 conv, sum -> c sum.
    # value -> abs(2 * N * c^2 max(conv) / (c^2 sum^2)) = value.
    # So we can normalize h to have sum 1, or norm 1, etc.
    # Let's normalize sum to 1.
    s = np.sum(h)
    if abs(s) < 1e-10:
        return np.zeros(N)
    return h / s


def _spectral_objective(params, M):
    """Objective: minimize the task value for the spectral peak model."""
    h = _h_from_spectral_peaks(params, M)
    if not np.all(np.isfinite(h)) or np.sum(h) == 0:
        return 1e10
    val = _evaluate(h)
    return val


def construct():
    t0 = time.time()
    deadline = t0 + _DEADLINE
    
    best_h = np.ones(N)
    best_val = _evaluate(best_h)  # Should be 2.0
    
    # Phase 1: Optimize spectral peak parameters
    # Try different numbers of peaks M
    for M in [2, 3, 5]:
        if time.time() > deadline - 60:
            break
            
        # Parameters: A, f, s, phi for each peak
        # A: amplitude (0, 5)
        # f: frequency bin (0, N/2)
        # s: sigma (1, N/4)
        # phi: phase (0, 2pi)
        
        bounds = []
        for j in range(M):
            bounds.append((0.1, 5.0))   # A
            bounds.append((0.0, N/2.0)) # f
            bounds.append((1.0, N/4.0)) # s
            bounds.append((0.0, 2*np.pi)) # phi
        
        # Use Differential Evolution for global search
        try:
            if time.time() < deadline - 30:
                result = differential_evolution(
                    lambda p: _spectral_objective(p, M),
                    bounds=bounds,
                    seed=42 + M,
                    maxiter=50,
                    tol=1e-6,
                    popsize=15,
                    polish=True
                )
                
                if result.fun < best_val:
                    h_candidate = _h_from_spectral_peaks(result.x, M)
                    v_candidate = _evaluate(h_candidate)
                    if v_candidate < best_val:
                        best_val = v_candidate
                        best_h = h_candidate.copy()
        except Exception:
            pass
        
        # Phase 2: Local refinement using L-BFGS-B on the best params from DE
        # We need to re-run DE to get the best x, or store it.
        # Let's do a quick local refinement on the current best_h using a 
        # different approach: random restarts in frequency domain.
        
    # Phase 3: Local refinement in time domain (coordinate ascent)
    # This is a fallback to ensure we squeeze out more value.
    if time.time() < deadline - 5:
        h = best_h.copy()
        current_val = best_val
        step_size = 0.1
        rng = np.random.default_rng(123)
        
        while time.time() < deadline - 5:
            improved = False
            for _ in range(100): # 100 random coordinates per epoch
                if time.time() > deadline - 5:
                    break
                idx = rng.integers(0, N)
                h_new = h.copy()
                h_new[idx] += rng.normal(0, step_size)
                
                v_new = _evaluate(h_new)
                if v_new < current_val:
                    current_val = v_new
                    h = h_new
                    improved = True
                    step_size *= 0.9
                else:
                    step_size = min(step_size * 1.05, 0.5)
            
            if not improved:
                step_size *= 0.8
                if step_size < 0.01:
                    break
            
            if current_val < best_val:
                best_val = current_val
                best_h = h.copy()
    
    # Ensure validity
    if not np.all(np.isfinite(best_h)) or np.sum(best_h) == 0:
        return np.ones(N)
    
    return best_h
