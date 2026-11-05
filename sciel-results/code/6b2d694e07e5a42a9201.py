"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def construct():
    """
    Construct a signed step function with 400 intervals to minimize the
    autoconvolution upper bound (max |autoconv|).
    """
    start_time = time.time()
    deadline = start_time + 115.0  # 5-second safety margin before 120s limit

    def thue_morse(n):
        """Generate Thue-Morse sequence of length n as +/-1."""
        x = np.zeros(n, dtype=np.int8)
        x[0] = 1
        for i in range(1, n):
            # Count set bits in i, even -> 1, odd -> -1
            x[i] = 1 if bin(i).count('1') % 2 == 0 else -1
        return x

    def max_autocorr(x):
        """Compute max |linear autocorrelation| of a +/-1 sequence of length N."""
        # Use FFT for speed
        n_fft = 1 << (2 * N - 1).bit_length()
        X = np.fft.rfft(x, n=n_fft)
        A = np.fft.irfft(X * np.conj(X), n=n_fft)[:N]
        return np.max(np.abs(A))

    def try_flips(x, time_limit):
        """Greedy local search: flip signs to reduce max autocorr."""
        x = x.copy()
        current_max = max_autocorr(x)
        best_max = current_max
        n_fft = 1 << (2 * N - 1).bit_length()
        
        while time.time() < time_limit:
            # Compute autocorr
            X = np.fft.rfft(x, n=n_fft)
            A = np.fft.irfft(X * np.conj(X), n=n_fft)[:N]
            current_max = np.max(np.abs(A))
            if current_max < best_max:
                best_max = current_max
            
            # Find index of max absolute autocorr
            idx = np.argmax(np.abs(A))
            val = A[idx]
            sign = 1 if val > 0 else -1
            
            # Try flipping each position and see if max decreases
            # Only try a subset to save time
            best_flip = -1
            best_val = current_max
            
            # Sample some positions to flip
            if N <= 100:
                candidates = range(N)
            else:
                # Try positions near the peak of autocorr, and random
                candidates = list(range(max(0, idx - 20), min(N, idx + 20)))
                # Add some random positions
                candidates += random.sample(range(N), min(50, N))
                candidates = list(set(candidates))
                random.shuffle(candidates)
            
            for i in candidates:
                if i >= N:
                    continue
                old_x = x[i]
                x[i] = -old_x
                
                # Quick check: recompute autocorr
                X2 = np.fft.rfft(x, n=n_fft)
                A2 = np.fft.irfft(X2 * np.conj(X2), n=n_fft)[:N]
                new_max = np.max(np.abs(A2))
                
                x[i] = old_x  # revert
                
                if new_max < best_val:
                    best_val = new_max
                    best_flip = i
            
            if best_flip >= 0:
                x[best_flip] = -x[best_flip]
            else:
                # No improvement, break
                break
        
        return x, best_max

    # Initialize with Thue-Morse
    x0 = thue_morse(N)
    best_x = x0.copy()
    best_val = max_autocorr(x0)
    
    # Do greedy local search from Thue-Morse init
    x_refined, val_refined = try_flips(x0, deadline - 30.0)
    if val_refined < best_val:
        best_x = x_refined
        best_val = val_refined
    
    # Random restarts with random initializations
    n_restarts = 0
    while time.time() < deadline - 15.0:
        n_restarts += 1
        # Random +/-1 initialization
        x_rand = np.random.choice([-1, 1], size=N).astype(np.int8)
        x_try, val_try = try_flips(x_rand, min(deadline - 10.0, time.time() + 5.0))
        if val_try < best_val:
            best_x = x_try
            best_val = val_try
    
    # Final polish: more careful local search on best solution
    if time.time() < deadline - 5.0:
        x_final, val_final = try_flips(best_x, deadline - 5.0)
        if val_final < best_val:
            best_x = x_final
            best_val = val_final
    
    return best_x
# EVOLVE-BLOCK-END
