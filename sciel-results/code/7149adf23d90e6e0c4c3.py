import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    # Structural change: Spectral shaping.
    # We generate a signal in the frequency domain with random phases and unit amplitude,
    # then inverse FFT. This tends to produce low-peak signals (flat spectrum).
    # We test several candidates and pick the one with the lowest max autoconvolution
    # normalized by squared sum.
    
    N = 4096
    
    # Constant vector baseline
    a_const = np.ones(N, dtype=np.float64)
    a_const /= N
    score_const = np.max(np.abs(np.convolve(a_const, a_const, mode='full'))) / (np.sum(a_const)**2)
    
    best_a = a_const
    best_score = score_const
    
    # Try random phase masks
    # Note: To ensure real signal, spectrum must be conjugate symmetric.
    # We generate random phases for positive frequencies and mirror them.
    
    for i in range(20):
        if time.time() > deadline - 10:
            break
        
        # Random phases
        phases = np.random.uniform(0, 2 * np.pi, N)
        # Ensure conjugate symmetry for real output
        # F[k] = A[k] * exp(i * phi[k])
        # F[N-k] = conj(F[k])
        # So phi[N-k] = -phi[k]
        
        # Simplify: Generate random real and imag parts, normalize magnitude?
        # Or just use random real signal?
        # Random real signal is not "spectral shaping" in the same way.
        
        # Let's stick to random real values first as a "different family"
        # but the inspiration specifically mentions "random, band-limited phase mask".
        
        # Construct complex spectrum
        spec = np.zeros(N, dtype=np.complex128)
        # Fill first half
        spec[:N//2] = np.exp(1j * phases[:N//2])
        spec[N//2] = 1.0 # DC
        # Mirror
        spec[N//2+1:] = np.conj(spec[1:N//2][::-1])
        
        a_rand = np.real(np.fft.ifft(spec))
        
        # Normalize sum to 1 to be comparable?
        # The metric is scale invariant, so we don't need to normalize,
        # but it helps with numerical stability.
        s = np.sum(a_rand)
        if np.abs(s) < 1e-6:
            continue
        a_rand = a_rand / s
        
        # Calculate score
        # Use FFT for convolution speed
        conv = np.fft.ifft(np.fft.fft(a_rand) * np.fft.fft(a_rand))
        max_conv = np.max(np.abs(conv))
        score = max_conv / (1.0**2) # Since sum is 1
        
        if score < best_score:
            best_score = score
            best_a = a_rand
            
    return best_a
