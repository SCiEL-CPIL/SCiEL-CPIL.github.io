import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s margin

    def get_sidelobe_max(f):
        c = np.convolve(f, f, mode='full')
        # c has length 2N-1
        # Peak at N-1
        # Sidelobes: 0..N-2 and N..2N-2
        # We can just take max of abs of all except the peak
        # But peak is N. Sidelobes are smaller.
        # To be safe, exclude index N-1
        # c[:N-1] and c[N:]
        # Note: c[0] is f[0]*f[0] = 1. c[2N-2] is f[N-1]*f[N-1] = 1.
        # Max sidelobe is max(max(abs(c[:N-1])), max(abs(c[N:])))
        # Optimization: c is symmetric. c[j] = c[2N-2-j].
        # So we only need to check c[0:N-1].
        # Wait, c[0] corresponds to lag N-1? No.
        # np.convolve(a, a, 'full')
        # a=[a0, a1]
        # conv = [a0^2, a0a1+a1a0, a1^2]
        # Index 0: lag 1? No, standard convolution index 0 is a0*b0.
        # For autoconvolution of length N:
        # Index 0: sum f[i]f[i-N]? No.
        # Let's check np.convolve documentation.
        # "The length of the result is M + N - 1, where M is the length of a and N is the length of b."
        # "The result at index k is sum_{i} a[i] b[k-i]".
        # For a=b=f, length N.
        # Index 0: f[0]f[0].
        # Index 1: f[0]f[1] + f[1]f[0].
        # Index N-1: sum_{i=0}^{N-1} f[i]f[i]. This is the peak (sum of squares = N).
        # Index N: f[1]f[0] + ... ?
        # Actually, autoconvolution is symmetric.
        # c[N-1] is the peak.
        # c[j] for j < N-1 corresponds to positive lags?
        # Let's verify with N=3, f=[1,1,1].
        # conv = [1, 2, 3, 2, 1].
        # Peak at index 2 (N-1). Value 3.
        # Sidelobes: 1, 2, 2, 1.
        # Max sidelobe is 2.
        # c[:N-1] = [1, 2]. Max 2.
        # c[N:] = [2, 1]. Max 2.
        # So max(abs(c[:N-1])) is sufficient due to symmetry.
        return np.max(np.abs(c[:N-1]))

    best_f = None
    best_val = float('inf')
    
    # Multiple restarts
    while time.time() < deadline - 10:
        # Initialize
        f = np.random.choice([1, -1], size=N)
        current_val = get_sidelobe_max(f)
        
        if current_val < best_val:
            best_val = current_val
            best_f = f.copy()
            
        T = 1.0
        cooling = 0.9995
        # Run SA for a fixed number of steps or time
        # Let's run for ~10 seconds or 100k steps
        steps = 0
        max_steps = 200000
        sa_start = time.time()
        
        while steps < max_steps and time.time() < sa_start + 8.0:
            i = np.random.randint(0, N)
            f[i] = -f[i]
            
            new_val = get_sidelobe_max(f)
            
            delta = new_val - current_val
            if delta < 0 or np.random.rand() < np.exp(-delta / T):
                current_val = new_val
                if new_val < best_val:
                    best_val = new_val
                    best_f = f.copy()
            else:
                f[i] = -f[i]
                
            T *= cooling
            if T < 0.01:
                T = 0.01
            steps += 1
            
    return best_f
