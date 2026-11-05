import numpy as np
import time

N = 400

def construct():
    # Deadline guard
    deadline = time.time() + 110
    
    # 1. Initialization: Thue-Morse sequence
    # Thue-Morse: t_n = popcount(n) % 2
    # Map 0 -> 1, 1 -> -1
    def thue_morse(n):
        return np.array([1 if bin(i).count('1') % 2 == 0 else -1 for i in range(n)])
    
    s = thue_morse(N)
    
    # Helper to compute peak sidelobe
    def peak_sidelobe(seq):
        # Aperiodic autocorrelation
        # R(k) = sum_{i=0}^{N-1-k} s[i] s[i+k]
        # Using FFT:
        # Zero-pad to 2*N
        # r = ifft(fft(s) * conj(fft(s)))
        # The first N elements of r give R(0), R(1), ... R(N-1) (circular)
        # Wait, circular autocorrelation is different.
        # For aperiodic, we can use:
        # R(k) = sum s[i] s[i+k]
        # This is the cross-correlation of s with a shifted s.
        # Efficient way:
        # s_padded = np.zeros(2*N)
        # s_padded[:N] = s
        # fft_s = np.fft.fft(s_padded)
        # acf = np.fft.ifft(fft_s * np.conj(fft_s)).real
        # acf[:N] contains R(0), R(1), ..., R(N-1)
        
        s_pad = np.zeros(2*N)
        s_pad[:N] = seq
        f = np.fft.fft(s_pad)
        acf = np.fft.ifft(f * np.conj(f)).real
        # acf[0] is R(0) = N
        # acf[1] is R(1)
        # ...
        # acf[N-1] is R(N-1)
        # We want max |acf[k]| for k in 1..N-1
        return np.max(np.abs(acf[1:N]))
    
    cur_peak = peak_sidelobe(s)
    best_s = s.copy()
    best_peak = cur_peak
    
    # Simulated Annealing
    T = 10.0
    T_min = 0.1
    alpha = 0.999
    
    while time.time() < deadline:
        i = np.random.randint(N)
        
        # Flip
        s[i] *= -1
        new_peak = peak_sidelobe(s)
        
        delta = new_peak - cur_peak
        
        if delta < 0:
            cur_peak = new_peak
            if cur_peak < best_peak:
                best_peak = cur_peak
                best_s = s.copy()
        else:
            # Accept with probability exp(-delta/T)
            if np.random.rand() < np.exp(-delta / T):
                cur_peak = new_peak
            else:
                # Revert
                s[i] *= -1
        
        T *= alpha
        if T < T_min:
            T = T_min
            # Optional: Restart if stuck?
            # For now, just keep going at low T (greedy)
            
    return best_s
