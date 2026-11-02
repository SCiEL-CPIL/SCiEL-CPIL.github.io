"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def _max_overlap(h):
    """
    Compute the maximum overlap of h with shifted (1-h).
    The overlap for shift k is sum_i h[i] * (1 - h[(i+k) % N]).
    We want to minimize the maximum over all shifts k.
    """
    N = len(h)
    # overlap_k = sum_i h[i] * (1 - h[(i+k)%N])
    # = N/2 - sum_i h[i]*h[(i+k)%N]  (since sum h = N/2)
    # So minimizing max overlap = maximizing min autocorr[k]
    
    # Compute circular autocorrelation via FFT
    H = np.fft.fft(h)
    autocorr = np.fft.ifft(np.abs(H)**2).real
    # autocorr[k] = sum_i h[i] * h[(i+k) % N]
    
    # overlap_k = N/2 - autocorr[k]
    overlaps = N/2 - autocorr
    return np.max(overlaps)

def _score(h):
    """
    Score is the upper bound on C5, which is the max overlap.
    Lower is better.
    """
    return _max_overlap(h)

def _project_to_valid(h):
    """
    Project h to satisfy: 0 <= h_i <= 1 and sum(h) = N/2.
    Uses a simple iterative clipping and normalization.
    """
    h = np.clip(h, 0, 1)
    # Iteratively adjust to meet sum constraint
    for _ in range(50):
        s = np.sum(h)
        if abs(s - N/2) < 1e-10:
            break
        # Scale to adjust sum, but keep within [0,1]
        if s > N/2:
            # Need to reduce sum; subtract from elements that are > 0
            excess = s - N/2
            # Find elements that can be reduced
            mask = h > 1e-10
            if np.sum(mask) == 0:
                break
            # Proportionally reduce
            h[mask] -= excess * (h[mask] / s)
            h = np.clip(h, 0, 1)
        else:
            # Need to increase sum; add to elements that are < 1
            deficit = N/2 - s
            mask = h < 1 - 1e-10
            if np.sum(mask) == 0:
                break
            # Proportionally increase
            h[mask] += deficit * ((1 - h[mask]) / np.sum(1 - h[mask]))
            h = np.clip(h, 0, 1)
    # Final exact adjustment
    diff = N/2 - np.sum(h)
    if abs(diff) > 1e-10:
        # Add/subtract diff to the element closest to 0.5
        idx = np.argmin(np.abs(h - 0.5))
        h[idx] += diff
        h = np.clip(h, 0, 1)
    return h

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin
    
    # Generate diverse initial candidates
    candidates = []
    
    # Uniform
    h0 = np.full(N, 0.5)
    candidates.append(h0)
    
    # Alternating
    h1 = np.zeros(N)
    h1[0::2] = 1.0
    h1[1::2] = 0.0
    candidates.append(h1)
    
    # Sinusoidal patterns with various frequencies
    x = np.linspace(0, 2*np.pi, N, endpoint=False)
    for freq in [1, 2, 3, 5, 7, 10]:
        h = 0.5 + 0.5 * np.cos(freq * x)
        h = _project_to_valid(h)
        candidates.append(h)
    
    # Random candidates
    rng = np.random.default_rng(42)
    for _ in range(20):
        h_rand = rng.random(N)
        h_rand = _project_to_valid(h_rand)
        candidates.append(h_rand)
    
    # Evaluate all candidates
    best_h = None
    best_score = float('inf')
    for h in candidates:
        s = _score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()
    
    # Spectral-domain optimization
    # Work with Fourier coefficients: H_k = FFT(h)_k
    # The autocorrelation is ifft(|H|^2), so overlap_k = N/2 - ifft(|H|^2)[k]
    # We want to minimize max_k (N/2 - ifft(|H|^2)[k]) = N/2 - min_k ifft(|H|^2)[k]
    # So we want to maximize min_k ifft(|H|^2)[k]
    
    # Parameterize H in terms of real amplitudes and phases
    # H[0] must be real (sum = N/2, so H[0] = N/2)
    # For k and N-k, H is conjugate symmetric
    
    def _h_to_H(h):
        return np.fft.fft(h)
    
    def _H_to_h(H):
        return np.fft.ifft(H).real
    
    # Objective in Fourier space: minimize smooth max of overlaps
    def _fourier_objective(params):
        """
        params: [amplitude_1, phase_1, amplitude_2, phase_2, ..., amplitude_{N/2-1}, phase_{N/2-1}]
        H[0] = N/2 (fixed)
        H[k] = amp_k * exp(i * phase_k) for k = 1, ..., N/2-1
        H[N-k] = conj(H[k])
        """
        if time.time() > deadline - 2:
            return float('inf')
        
        amp = params[0::2]
        phase = params[1::2]
        
        # Build H
        H = np.zeros(N, dtype=complex)
        H[0] = N/2
        for k in range(1, N//2):
            H[k] = amp[k-1] * np.exp(1j * phase[k-1])
            H[N-k] = np.conj(H[k])
        if N % 2 == 0:
            # H[N/2] must be real
            H[N//2] = amp[N//2 - 1]  # real
        
        h = _H_to_h(H)
        h = _project_to_valid(h)
        
        # Compute score
        s = _score(h)
        return s
    
    # Use L-BFGS-B in Fourier space
    # Initial params from best_h
    H_best = _h_to_H(best_h)
    amp_init = np.zeros(N//2)
    phase_init = np.zeros(N//2)
    for k in range(1, N//2):
        amp_init[k-1] = np.abs(H_best[k])
        phase_init[k-1] = np.angle(H_best[k])
    if N % 2 == 0:
        amp_init[N//2 - 1] = np.real(H_best[N//2])
    
    # Flatten
    params_init = np.zeros(2 * (N//2))
    params_init[0::2] = amp_init
    params_init[1::2] = phase_init
    
    # Bounds
    amp_bounds = [(0, 10)] * (N//2)
    phase_bounds = [(-np.pi, np.pi)] * (N//2)
    bounds = []
    for i in range(N//2):
        bounds.append(amp_bounds[i])
        bounds.append(phase_bounds[i])
    
    # Try optimization from a few different starting points
    for trial in range(5):
        if time.time() > deadline - 10:
            break
        
        if trial == 0:
            p_init = params_init.copy()
        else:
            # Perturb
            p_init = params_init + 0.1 * rng.standard_normal(2 * (N//2))
        
        try:
            res = minimize(_fourier_objective, p_init, method='L-BFGS-B', 
                          bounds=bounds, options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8})
            
            if res.fun < best_score:
                # Extract h from result
                amp = res.x[0::2]
                phase = res.x[1::2]
                H = np.zeros(N, dtype=complex)
                H[0] = N/2
                for k in range(1, N//2):
                    H[k] = amp[k-1] * np.exp(1j * phase[k-1])
                    H[N-k] = np.conj(H[k])
                if N % 2 == 0:
                    H[N//2] = amp[N//2 - 1]
                
                h_opt = _H_to_h(H)
                h_opt = _project_to_valid(h_opt)
                s = _score(h_opt)
                if s < best_score:
                    best_score = s
                    best_h = h_opt.copy()
        except Exception:
            pass
    
    # Also try direct gradient descent in h-space with subgradient
    if time.time() < deadline - 30:
        # Subgradient descent on the max overlap
        h_curr = best_h.copy()
        lr = 0.01
        for step in range(2000):
            if time.time() > deadline - 5:
                break
            
            # Compute all overlaps
            H = np.fft.fft(h_curr)
            autocorr = np.fft.ifft(np.abs(H)**2).real
            overlaps = N/2 - autocorr
            
            # Find argmax
            k_max = np.argmax(overlaps)
            
            # Subgradient of overlap_k_max w.r.t. h
            # overlap_k = sum_i h[i] * (1 - h[(i+k)%N])
            # d(overlap_k)/dh[j] = (1 - h[(j+k)%N]) - h[(j-k)%N]
            grad = np.zeros(N)
            for j in range(N):
                grad[j] = (1 - h_curr[(j + k_max) % N]) - h_curr[(j - k_max) % N]
            
            # Project gradient to satisfy sum constraint (zero mean component)
            grad = grad - np.mean(grad)
            
            # Update
            h_curr = h_curr - lr * grad
            h_curr = _project_to_valid(h_curr)
            
            s = _score(h_curr)
            if s < best_score:
                best_score = s
                best_h = h_curr.copy()
            
            # Adaptive learning rate
            if step % 100 == 0 and step > 0:
                lr *= 0.95
    
    # Final check
    h_final = _project_to_valid(best_h)
    
    return h_final
# EVOLVE-BLOCK-END
