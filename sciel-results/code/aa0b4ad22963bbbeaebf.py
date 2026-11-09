import numpy as np
import time
import random

N = 50

def compute_ratio(h):
    """
    Compute the ratio (L2^2) / (L1 * SupConv).
    Uses FFT for speed. Assumes normalization to sum 1.
    """
    h = np.maximum(h, 0)
    s = np.sum(h)
    if s == 0:
        return 0.0, h
    h_norm = h / s
    l2_sq = np.sum(h_norm**2)
    
    # FFT convolution
    size = 2 * len(h_norm)
    H = np.zeros(size)
    H[:len(h_norm)] = h_norm
    H_fft = np.fft.fft(H)
    Conv_fft = H_fft * H_fft
    Conv = np.real(np.fft.ifft(Conv_fft))
    sup_conv = np.max(Conv)
    
    if sup_conv == 0:
        return 0.0, h_norm
    ratio = l2_sq / sup_conv
    return ratio, h_norm

def random_initialization():
    # Create a random non-negative vector
    h = np.random.rand(N)
    return h

def optimize(h0, steps=200):
    """
    Coordinate ascent with adaptive step size, using FFT for conv.
    """
    h = h0.copy()
    s = np.sum(h)
    if s == 0:
        return 0.0, h
    h = h / s
    
    best_ratio, _ = compute_ratio(h)
    best_h = h.copy()
    
    # Adaptive step size
    step_size = 0.05
    
    for _ in range(steps):
        improved = False
        for i in range(N):
            original = h[i]
            
            # Try increasing h[i]
            h[i] += step_size
            h = np.maximum(h, 0)
            s = np.sum(h)
            if s == 0: break
            h = h / s
            
            r, h_new = compute_ratio(h)
            if r > best_ratio:
                best_ratio = r
                best_h = h_new.copy()
                improved = True
                step_size *= 1.1
            else:
                # Revert and try decreasing
                h[i] = original
                h[i] -= step_size
                if h[i] < 0:
                    h[i] = 0
                s = np.sum(h)
                if s == 0: break
                h = h / s
                
                r, h_new = compute_ratio(h)
                if r > best_ratio:
                    best_ratio = r
                    best_h = h_new.copy()
                    improved = True
                    step_size *= 1.1
                else:
                    # Revert
                    h[i] = original
                    s = np.sum(h)
                    h = h / s
                    step_size *= 0.8
                    if step_size < 1e-6:
                        break
            
            if step_size < 1e-6:
                break
                
        if step_size < 1e-6:
            break
            
    return best_ratio, best_h

def spectral_refine(h0, steps=50):
    """
    Refine the profile in the frequency domain to suppress the autoconvolution peak.
    The autoconvolution peak is related to the DC component of the power spectrum.
    We project the profile onto the subspace orthogonal to the all-ones vector (zero mean)
    and try to minimize the L2 norm of the resulting autoconvolution, which indirectly
    helps reduce the peak for zero-mean components, while keeping the DC component fixed.
    """
    h = h0.copy()
    s = np.sum(h)
    if s == 0:
        return 0.0, h
    h = h / s
    
    best_ratio, _ = compute_ratio(h)
    best_h = h.copy()
    
    # Compute FFT of h
    H = np.fft.fft(h)
    
    # Keep DC component (index 0) and conjugate symmetry
    # We will perturb the non-DC components
    for _ in range(steps):
        # Generate a random perturbation in frequency domain
        # Zero mean constraint means H[0] is fixed.
        # We perturb H[1:N-1]
        pert = np.zeros(N, dtype=complex)
        # Random phase and small amplitude
        for k in range(1, N):
            if k < N - k: # Independent frequencies
                pert[k] = (np.random.rand() - 0.5) * 0.1 + 1j * (np.random.rand() - 0.5) * 0.1
                pert[N-k] = np.conj(pert[k])
            elif k == N-k: # Nyquist frequency (if N is even)
                pert[k] = (np.random.rand() - 0.5) * 0.1
        
        H_new = H + pert
        h_new = np.real(np.fft.ifft(H_new))
        
        # Ensure non-negativity
        h_new = np.maximum(h_new, 0)
        
        # Re-normalize
        s_new = np.sum(h_new)
        if s_new == 0:
            continue
        h_new = h_new / s_new
        
        r_new, _ = compute_ratio(h_new)
        
        if r_new > best_ratio:
            best_ratio = r_new
            best_h = h_new.copy()
            H = np.fft.fft(h_new) # Update H to the new best
            
    return best_ratio, best_h

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    best_ratio = 0.0
    best_h = np.ones(N)
    
    # Initial best is seed
    r_seed, h_seed = compute_ratio(best_h)
    best_ratio = r_seed
    best_h = h_seed
    
    # Try several random starts
    num_starts = 200 # More starts because FFT is fast
    
    for i in range(num_starts):
        if time.time() > deadline:
            break
        
        # Different initializations
        if i % 5 == 0:
            h0 = np.ones(N) # Uniform
        elif i % 5 == 1:
            h0 = np.random.rand(N)
        elif i % 5 == 2:
            # Gaussian bump
            x = np.linspace(-1, 1, N)
            h0 = np.exp(-x**2 * 5)
        elif i % 5 == 3:
            # Two bumps
            x = np.linspace(-1, 1, N)
            h0 = np.exp(-(x-0.5)**2 * 10) + np.exp(-(x+0.5)**2 * 10)
        else:
            # Random sparse
            h0 = np.zeros(N)
            idx = np.random.choice(N, size=5, replace=False)
            h0[idx] = 1.0
            
        # Optimize with coordinate ascent
        r, h_opt = optimize(h0, steps=100)
        
        if r > best_ratio:
            best_ratio = r
            best_h = h_opt
        
        # Then refine with spectral method
        if time.time() < deadline - 10: # Leave some time
            r_spec, h_spec = spectral_refine(h_opt, steps=50)
            if r_spec > best_ratio:
                best_ratio = r_spec
                best_h = h_spec
    
    # Return the list of heights
    return list(best_h)
