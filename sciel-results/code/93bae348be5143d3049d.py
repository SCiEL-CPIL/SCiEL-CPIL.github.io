import numpy as np
from scipy.signal import fftconvolve

N = 400
w = 1.0 / 800.0  # Correct width for [-1/4, 1/4] with 400 intervals

def construct():
    # Start with random signs for better exploration
    np.random.seed(42)
    s = np.random.choice([-1, 1], N)
    
    def compute_autoconv_max_nonzero(s):
        # Compute discrete autocorrelation
        n_fft = 2 * N
        S = np.fft.rfft(s, n_fft)
        r = np.fft.irfft(S * np.conj(S), n_fft)[:N]
        
        # We want to minimize the maximum absolute value at non-zero lags
        # r[0] is constant N, so we ignore it
        if N > 1:
            max_val = np.max(np.abs(r[1:])) * w
        else:
            max_val = 0.0
        return max_val

    best_s = s.copy()
    best_val = compute_autoconv_max_nonzero(s)
    
    # Subgradient descent on relaxed signs
    for iteration in range(10000):
        # Compute autocorrelation
        n_fft = 2 * N
        S = np.fft.rfft(s, n_fft)
        r = np.fft.irfft(S * np.conj(S), n_fft)[:N]
        
        # Find the k >= 1 that achieves the maximum absolute value
        if N > 1:
            abs_r_nonzero = np.abs(r[1:])
            k_max_rel = np.argmax(abs_r_nonzero)
            k_max = k_max_rel + 1  # Actual lag index
            r_max = r[k_max]
            
            # Subgradient of |r[k_max]| with respect to s
            # r[k_max] = sum_{i=0}^{N-1-k_max} s[i] * s[i+k_max]
            # d|r[k_max]|/ds[i] = sign(r[k_max]) * (s[i+k_max] if i < N-k_max else 0)
            #                    + sign(r[k_max]) * (s[i-k_max] if i >= k_max else 0)
            
            sg = np.zeros(N)
            sign_r = np.sign(r_max)
            
            for i in range(N):
                grad = 0.0
                if i < N - k_max:
                    grad += s[i + k_max]
                if i >= k_max:
                    grad += s[i - k_max]
                sg[i] = sign_r * grad
            
            # Update: s = s - lr * sg, then project to [-1, 1]
            # Use a decaying learning rate
            lr = 1.0 / (10.0 + iteration)
            s = s - lr * sg
            s = np.clip(s, -1, 1)
            
            # Check current value
            current_val = compute_autoconv_max_nonzero(s)
            if current_val < best_val:
                best_val = current_val
                best_s = s.copy()
        else:
            break

    # Round to ±1
    s_final = np.sign(best_s)
    # Handle zeros
    s_final[s_final == 0] = 1
    
    return s_final
