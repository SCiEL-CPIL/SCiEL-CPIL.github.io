import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    def calc_max_sidelobe_fft(x):
        # Zero pad to next power of 2 or 2N to avoid circular convolution artifacts
        # For aperiodic autocorrelation, we need length >= 2N-1
        n_fft = 1024 # 2^10, which is > 2*400-1 = 799
        X = np.fft.fft(x, n=n_fft)
        C = np.fft.ifft(X * np.conj(X)).real
        # C is circular autocorrelation of the zero-padded signal
        # The aperiodic autocorrelation is in the first N elements?
        # Let's verify.
        # If x is zero padded to n_fft, then C[k] = sum x[i] x[i+k]
        # For k < N, this is the aperiodic autocorrelation.
        # For k >= N, it wraps around, but since x is 0 there, it's 0?
        # Actually, if x is padded with zeros, the circular correlation matches aperiodic for lags 0..N-1.
        # We need lags 1..N-1 and -1..-(N-1).
        # In FFT output, C[0] is lag 0.
        # C[1..N-1] are positive lags.
        # C[n_fft - (N-1) .. n_fft - 1] are negative lags.
        
        # Max absolute value of sidelobes
        # Lags 1 to N-1
        pos_lags = C[1:N]
        # Lags -(N-1) to -1 correspond to indices n_fft-(N-1) to n_fft-1
        neg_lags = C[n_fft-(N-1):n_fft]
        
        return np.max(np.abs(np.concatenate([pos_lags, neg_lags])))

    best_x = None
    best_val = float('inf')
    
    while time.time() < deadline:
        x = np.random.choice([-1, 1], N)
        curr_max = calc_max_sidelobe_fft(x)
        
        no_improve_count = 0
        while time.time() < deadline and no_improve_count < 5000:
            i = np.random.randint(N)
            x[i] *= -1
            new_max = calc_max_sidelobe_fft(x)
            
            if new_max < curr_max:
                curr_max = new_max
                no_improve_count = 0
                if curr_max < best_val:
                    best_val = curr_max
                    best_x = x.copy()
            else:
                x[i] *= -1
                no_improve_count += 1
                
    if best_x is None:
        return np.ones(N)
    return best_x
