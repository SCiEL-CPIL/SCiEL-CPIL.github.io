import numpy as np
import time

N = 400
DEADLINE = 150.0  # Safety margin from 150s kill, leave 10s for overhead

def _compute_c3(signed_step):
    """
    Compute the C3 upper bound (max absolute value of autoconvolution).
    """
    # Use a fine grid for accurate convolution
    M = 4 * N
    dx = 0.5 / M  # Domain length is 0.5
    
    # Build f on fine grid
    f_fine = np.zeros(M)
    # Vectorized assignment
    indices = np.repeat(np.arange(N), M // N)
    f_fine[indices] = signed_step
    
    # Autoconvolution via FFT
    # Note: Circular convolution of length 2*M is equivalent to linear if M is large enough
    # f is supported on [0, M) in the discrete sense, length M.
    # Linear convolution length is 2M-1.
    # We use FFT of size 2*M.
    fft_f = np.fft.fft(f_fine, n=2*M)
    autoconv = np.fft.ifft(fft_f * np.conj(fft_f)).real * dx
    
    # The autoconvolution is defined on [-0.5, 0.5] shifted.
    # We just need the max absolute value.
    return np.max(np.abs(autoconv))

def construct():
    start_time = time.time()
    
    # 1. Spectral Minimization over continuous values in [-1, 1]
    # We want to minimize ||f * f||_2^2.
    # In Fourier space, ||f * f||_2^2 = ||F^2||_2^2.
    # We can iteratively update f to reduce the energy of F^2.
    # A simple gradient step or projection method can be used.
    # Let's use a few iterations of "spectral filtering" where we dampen the frequencies
    # that contribute most to the autoconvolution energy.
    
    # Initialize with random values or all ones?
    # All ones gives a peak at 0. Random gives a flatter spectrum but higher variance.
    # Let's start with a random initialization to break symmetry, then project.
    # Actually, the problem is symmetric under sign flip and permutation of intervals (if we ignore the spatial structure of convolution).
    # Wait, convolution is NOT invariant under arbitrary permutation. It respects the order.
    
    # Let's try starting from a "white noise" pattern or a specific structured pattern.
    # A better start: Random continuous values in [-1, 1].
    np.random.seed(123)
    f = np.random.uniform(-1, 1, N)
    
    # Iterate: 
    # 1. Compute FFT
    # 2. Compute "gradient" of energy of F^2.
    # Energy = sum |F_k|^4.
    # d/dF_k (|F_k|^4) = 4 |F_k|^2 F_k.
    # So we want to subtract a multiple of |F_k|^2 F_k from F_k.
    # This is equivalent to applying a filter in frequency domain.
    
    # Let's perform a few iterations of this spectral descent.
    # We must ensure f stays in [-1, 1] or at least we clip at the end.
    # Clipping changes the spectrum, so we need to iterate:
    # f <- clip(f - eta * ifft(|FFT(f)|^2 * FFT(f)))
    
    steps = 20
    eta = 0.1
    
    for i in range(steps):
        if time.time() - start_time > DEADLINE - 5.0:
            break
            
        F = np.fft.fft(f)
        # Gradient of sum |F|^4 w.r.t f
        # grad_f = ifft(4 * |F|^2 * F)
        # We use a smaller step size to avoid oscillation
        grad = np.fft.ifft(4 * np.abs(F)**2 * F).real
        f = f - eta * grad
        f = np.clip(f, -1, 1)
        
    # 2. Threshold to {-1, 1}
    # Simple threshold at 0
    f_signed = np.sign(f)
    # Handle zeros (unlikely with continuous init, but just in case)
    f_signed[f_signed == 0] = 1
    
    # 3. Local Search Refinement
    # The spectral method might get stuck in a local minimum or the thresholding
    # might not be optimal. We do a quick local search to polish.
    
    best_step = f_signed.copy()
    best_c3 = _compute_c3(best_step)
    
    # Precompute FFT for fast evaluation? 
    # _compute_c3 is fast enough for a few hundred iterations.
    
    np.random.seed(456)
    max_iter = 500
    no_improve = 0
    patience = 50
    
    for i in range(max_iter):
        if time.time() - start_time > DEADLINE - 2.0:
            break
            
        idx = np.random.randint(N)
        old_val = best_step[idx]
        best_step[idx] = -old_val
        
        new_c3 = _compute_c3(best_step)
        
        if new_c3 < best_c3:
            best_c3 = new_c3
            no_improve = 0
        else:
            best_step[idx] = old_val
            no_improve += 1
            
        if no_improve >= patience:
            break
            
    return best_step
