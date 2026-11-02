import numpy as np
import time

N = 400

def _calc_score(x):
    S = np.sum(x)
    if S == 0:
        return 0.0 # Avoid division by zero, though S should be large
    # Compute autoconvolution
    # Zero pad to avoid circular convolution artifacts
    n_fft = 2 * N
    X = np.fft.fft(x, n=n_fft)
    C = np.fft.ifft(X * np.conj(X)).real
    # C[0] is c_0 = N
    # C[1:N] are positive lags
    # C[N+1:] are negative lags (wrapped)
    # We want max absolute value for k != 0
    # The lags are 0, 1, ..., N-1, -(N-1), ..., -1
    # In FFT output:
    # C[0] -> lag 0
    # C[1] -> lag 1
    # ...
    # C[N-1] -> lag N-1
    # C[N] -> lag N (which is 0 for length N sequence? No, lag N is 0)
    # Actually, for linear convolution of length N with itself:
    # The result has length 2N-1.
    # FFT size 2N gives linear convolution if padded correctly?
    # np.fft.fft(x, n=2*N) computes circular conv of length 2N.
    # Since x has length N, x*x has length 2N-1.
    # Circular conv of length 2N is the same as linear conv if 2N >= 2N-1. Yes.
    # So C contains the linear autoconvolution.
    # C[0] = c_0
    # C[1:N] = c_1 ... c_{N-1}
    # C[N] = c_N = 0
    # C[N+1:2N] corresponds to negative lags?
    # Let's check indices.
    # Linear conv y[k] = sum x[i] x[i+k].
    # FFT method gives y[0..2N-2].
    # y[0] is lag 0.
    # y[1] is lag 1.
    # ...
    # y[N-1] is lag N-1.
    # y[N] is lag N (0).
    # y[N+1] is lag N+1? No.
    # The autoconvolution is symmetric.
    # c_k = c_{-k}.
    # So we just need to look at lags 1 to N-1.
    # In the FFT output C of size 2N:
    # C[0] is c_0.
    # C[1] is c_1.
    # ...
    # C[N-1] is c_{N-1}.
    # C[N] is c_N = 0.
    # C[N+1] ... C[2N-1] are the wrapped negative lags?
    # Actually, np.fft.ifft returns the linear convolution result in order 0..2N-1 if padded to 2N?
    # No, ifft of product of FFTs gives circular convolution.
    # If we pad x to 2N, the circular convolution is the linear convolution.
    # The result is indexed 0 to 2N-1.
    # c_0 is at 0.
    # c_1 is at 1.
    # ...
    # c_{N-1} is at N-1.
    # c_N is at N (value 0).
    # c_{N+1} is at N+1? No, linear conv length is 2N-1.
    # Indices 0 to 2N-2.
    # So C[0:N] covers lags 0 to N-1.
    # C[N:2N] covers the "negative" lags wrapped around?
    # For autoconvolution, c_k = c_{-k}.
    # So max_{k != 0} |c_k| is max(|C[1:N]|, |C[N+1:2N]|)?
    # Actually, due to symmetry, C[k] = C[2N-k] (approximately, with float errors).
    # So we can just take max of C[1:N].
    
    max_off_diag = np.max(np.abs(C[1:N]))
    val = max_off_diag / (S * S)
    return val

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    x = np.ones(N, dtype=np.float64)
    best_score = _calc_score(x)
    best_x = x.copy()
    
    # Local search
    # We flip one sign at a time.
    # To speed up, we can precompute contributions?
    # Or just recompute. N=400, FFT is fast.
    # 110 seconds / time_per_iter.
    # Let's estimate time_per_iter.
    
    # Simple local search with random restarts
    for _ in range(10000): # Max iterations
        if time.time() > deadline:
            break
            
        # Try flipping each index
        best_delta = 0
        best_flip_idx = -1
        current_score = _calc_score(x)
        
        # Optimization: Instead of full recompute for every flip,
        # we can just pick a random index or a few random indices to flip.
        # Full local search is O(N) evaluations per step.
        # O(N) * N log N = O(N^2 log N) per step.
        # 160k * 9 = 1.4M ops per step.
        # 110s / 1.4M ops/s? Python is slow.
        # Let's do random flips (First-Improvement or Best-Improvement on a subset).
        
        # Try 10 random flips per iteration
        for _ in range(10):
            idx = np.random.randint(N)
            x[idx] *= -1
            new_score = _calc_score(x)
            if new_score < current_score:
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_x = x.copy()
            else:
                x[idx] *= -1 # Revert
            
        # If no improvement in last 10, maybe restart?
        # For now, just keep going.
        
    return best_x
