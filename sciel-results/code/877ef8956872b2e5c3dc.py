import numpy as np
import time
import sys

N = 600

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Initialize with uniform distribution
    h = np.ones(N) / N
    
    # FFT size
    M = 2048
    h_padded = np.zeros(M)
    h_padded[:N] = h
    
    best_score = np.inf
    best_h = h.copy()
    
    # Learning rate
    alpha = 0.01
    
    while time.time() < deadline:
        # Update padded array
        h_padded[:N] = h
        
        # Compute autoconvolution via FFT
        F = np.fft.fft(h_padded)
        C = np.fft.ifft(F * np.conj(F)).real
        
        # The autoconvolution is symmetric, peak should be near M/2 or 0
        # In FFT, the circular convolution is computed.
        # We need the linear autoconvolution.
        # The linear autoconvolution of two length N signals is length 2N-1.
        # The FFT of size M >= 2N gives the linear convolution if we handle the wrap-around.
        # Actually, `ifft(fft(f)*conj(fft(f)))` gives the circular autoconvolution.
        # If M >= 2N-1, the circular autoconvolution matches the linear autoconvolution
        # in the first 2N-1 bins, and the rest are zero (or wrap-around if not padded enough).
        # Since M=2048 and 2N-1 = 1199, we are safe.
        
        # Find the peak
        # The peak of the autoconvolution of a non-negative function is at lag 0.
        # In the FFT output, lag 0 is at index 0 (and M/2 due to symmetry?).
        # For real signals, the autoconvolution is even.
        # The value at lag 0 is C[0].
        # However, due to numerical errors, it might be slightly off.
        # Also, we want the global max.
        peak_val = np.max(C)
        peak_idx = np.argmax(C)
        
        # Calculate score (approximate)
        # Norm is sum(h) = 1
        score = peak_val
        if score < best_score:
            best_score = score
            best_h = h.copy()
            
        # Compute gradient
        # grad[i] = d/dh_i (C[peak_idx])
        # C[k] = sum_j h[j] h[(k-j) % M]
        # dC[k]/dh[i] = h[(k-i) % M] + h[(k+i) % M]
        
        k = peak_idx
        grad = np.zeros(N)
        # We only care about i in 0..N-1
        # h is zero for i >= N
        # So h[(k-i) % M] is non-zero only if (k-i) % M < N
        # Similarly for (k+i) % M
        
        # Vectorized approach
        i = np.arange(N)
        # Term 1: h[(k - i) % M]
        idx1 = (k - i) % M
        # We need h at these indices, but h is only defined for 0..N-1
        # If idx1 >= N, the value is 0
        val1 = np.zeros(N)
        mask1 = idx1 < N
        val1[mask1] = h[idx1[mask1]]
        
        # Term 2: h[(k + i) % M]
        idx2 = (k + i) % M
        val2 = np.zeros(N)
        mask2 = idx2 < N
        val2[mask2] = h[idx2[mask2]]
        
        grad = val1 + val2
        
        # Update
        h_new = h - alpha * grad
        
        # Project to simplex: h >= 0, sum(h) = 1
        # Simple projection
        h_new = np.maximum(h_new, 0)
        s = np.sum(h_new)
        if s > 0:
            h_new /= s
        else:
            h_new = np.ones(N) / N
            
        h = h_new
        
        # Adaptive learning rate?
        # If score increases, reduce alpha
        # For now, keep it simple
        
    # Final verification with exact continuous autoconvolution?
    # Given time constraints, we trust the FFT approximation.
    # The FFT approximation is quite accurate for smooth functions.
    
    return best_h.tolist()

# EVOLVE-BLOCK-END
