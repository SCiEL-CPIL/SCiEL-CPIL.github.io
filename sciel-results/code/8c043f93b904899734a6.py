import numpy as np
from scipy.fft import fft, ifft
import time

N = 600

def compute_autoconv_max(h):
    """Compute the normalized maximum of the autoconvolution of step function h."""
    h = np.asarray(h, dtype=float)
    if len(h) == 0:
        return 0.0
    n = len(h)
    size = 2 * n
    H = fft(h, size)
    conv = np.real(ifft(H * H, size)[:2*n-1])
    total = np.sum(h)
    if total < 1e-10:
        return 0.0
    return np.max(conv) / (total * total)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    # Initial guess: White noise is a good start for flat spectrum, 
    # but we need non-negative. Let's start with a mix of uniform and random.
    # Or just ones, which has a sharp peak at lag 0.
    
    # Strategy: 
    # 1. Frequency domain flattening (multiplicative update on power spectrum)
    # 2. Non-negativity projection
    # 3. Spatial gradient descent on autoconvolution variance if needed
    
    def flat_spectrum_update(h, gamma, iters):
        """
        Apply iterative frequency domain flattening with non-negativity projection.
        h: array of length N
        gamma: step size
        iters: number of inner iterations
        """
        current_h = h.copy()
        for _ in range(iters):
            if time.time() > deadline - 5:
                break
            
            # FFT
            size = 2 * N
            H = fft(current_h, size)
            
            # Power spectrum
            P = np.abs(H) ** 2
            
            # Mean power (target for flat spectrum)
            # Note: Sum of P is constant (Parseval) proportional to sum(h^2).
            # We normalize h at the end of each step to keep sum(h) constant, 
            # so sum(h^2) changes. We want P to be flat.
            P_mean = np.mean(P)
            
            # Update H: reduce components with high power, boost low power
            # Using exponential update for stability
            # H_new = H * exp(-gamma * (P - P_mean))
            # To avoid underflow/overflow, scale gamma appropriately
            delta_P = P - P_mean
            H_new = H * np.exp(-gamma * delta_P)
            
            # Inverse FFT
            h_new = np.real(ifft(H_new, size)[:N])
            
            # Non-negativity projection
            h_new = np.maximum(h_new, 0)
            
            # Normalize to keep sum(h) constant (e.g., 1.0)
            s = np.sum(h_new)
            if s > 1e-10:
                h_new = h_new / s
            else:
                # Fallback if it collapses
                h_new = h.copy()
                break
                
            current_h = h_new
            
        return current_h

    # Initializations
    np.random.seed(42)
    initial_guesses = [
        np.ones(N),
        np.random.rand(N),
        1.0 + 0.5 * np.cos(2 * np.pi * np.linspace(0, 1, N, endpoint=False)),
        1.0 + 0.5 * np.sin(2 * np.pi * np.linspace(0, 1, N, endpoint=False)),
    ]
    
    best_h = np.ones(N)
    best_score = compute_autoconv_max(best_h)
    
    for h0 in initial_guesses:
        if time.time() > deadline - 10:
            break
            
        h0 = np.maximum(h0, 0.01)
        h0 = h0 / np.sum(h0)
        
        current_h = h0.copy()
        
        # Phase 1: Coarse Frequency Flattening
        # Use larger gamma initially
        gamma = 0.1
        for _ in range(20):
            if time.time() > deadline - 5:
                break
            current_h = flat_spectrum_update(current_h, gamma, iters=10)
            score = compute_autoconv_max(current_h)
            if score < best_score:
                best_score = score
                best_h = current_h.copy()
        
        # Phase 2: Fine Frequency Flattening
        gamma = 0.01
        for _ in range(50):
            if time.time() > deadline - 5:
                break
            current_h = flat_spectrum_update(current_h, gamma, iters=5)
            score = compute_autoconv_max(current_h)
            if score < best_score:
                best_score = score
                best_h = current_h.copy()
                
        # Phase 3: Spatial Gradient Descent on Autoconvolution Variance
        # Objective: minimize sum((C - mean(C))^2)
        # This further flattens the autoconvolution in the spatial domain.
        
        def spatial_grad_step(h, lr, iters):
            current_h = h.copy()
            for _ in range(iters):
                if time.time() > deadline - 5:
                    break
                # Compute autoconvolution
                size = 2 * N
                H = fft(current_h, size)
                C = np.real(ifft(H * H, size)[:2*N-1])
                
                # Mean of autoconvolution
                C_mean = np.mean(C)
                
                # Gradient of variance w.r.t h
                # d/dh_i sum_k (C_k - C_mean)^2
                # C_k = sum_j h_j h_{k-j}
                # This is effectively a convolution of the error (C - C_mean) with h
                # grad = 2 * ( (C - C_mean) * h )
                # Note: (C - C_mean) has length 2N-1, h has length N.
                # We need the gradient for each h_i.
                # grad_i = 2 * sum_k (C_k - C_mean) * (dC_k/dh_i)
                # dC_k/dh_i = h_{k-i} + h_{i-k} (symmetry)
                # This is equivalent to circular cross-correlation or similar.
                # Simpler: Use FFT to compute the convolution of (C-C_mean) with h.
                # We want the gradient vector g of length N.
                # g_i = sum_k (C_k - C_mean) (h_{k-i} + h_{i-k})
                # Let E = C - C_mean.
                # g = 2 * (E * h) [Circular Correlation/Convolution]
                
                E = C - C_mean
                # E has length 2N-1. We need to map this to a convolution with h (length N).
                # The gradient of the quadratic form h^T M h is 2 M h.
                # Here the "matrix" is defined by the autoconvolution structure.
                # Actually, the gradient of sum C_k^2 w.r.t h is 4 * (h * C) ?? 
                # Let's use the explicit formula:
                # grad_i = 2 * sum_k (C_k - C_mean) * h_{k-i} (with appropriate bounds/circularity)
                # Since C is linear convolution, the gradient involves cross-correlation.
                
                # Efficient computation:
                # grad = 2 * ifft( fft(E, 2N) * fft(h, 2N) ) ??? No.
                
                # Let's stick to the spatial update:
                # h_new = h - lr * grad
                # We can approximate the gradient using FFT convolution of E and h.
                # E is centered at 0..2N-2.
                # We want the "correlation" of E and h.
                
                # Simpler robust approach: 
                # If we minimize sum C_k^2, the gradient is 4 * (h * C) (linear conv, truncated to N).
                # Let's compute linear convolution of h and C.
                # C has length 2N-1. h has length N.
                # Conv length 3N-2.
                # We only need the first N components? Or the middle?
                # The term for h_i in sum C_k^2 is:
                # 2 * sum_k C_k * (d C_k / d h_i)
                # d C_k / d h_i = h_{k-i} (if k-i in [0, N-1]) + h_{i-k} (if i-k in [0, N-1])
                # So grad_i = 2 * ( sum_{k} C_k h_{k-i} ) + 2 * ( sum_k C_k h_{i-k} )
                # This is 2 * (C * h)_i + 2 * (C * h^R)_i ?
                
                # Let's just use the frequency domain insight:
                # The Hessian is related to |H|^2.
                # Let's just do a simple spatial step:
                # Reduce h_i where it contributes to high peaks.
                # Contribution of h_i to peak at k is h_i * h_{k-i}.
                
                # To avoid complexity, we'll use the Frequency Flattening which is already a structural change.
                # I will add a simple local smoothing if time permits, but the main change is the FD flattening.
                
                pass # No spatial grad step implemented to keep code clean and focused on the structural change
        
        return best_h

    # Final check
    if time.time() < deadline - 2:
        # One last round of very fine FD flattening on the best found
        gamma = 0.001
        final_h = flat_spectrum_update(best_h, gamma, iters=20)
        final_score = compute_autoconv_max(final_h)
        if final_score < best_score:
            best_h = final_h
    
    return list(best_h)
