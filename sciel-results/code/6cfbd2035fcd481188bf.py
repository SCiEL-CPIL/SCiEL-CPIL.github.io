"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
import math

_start_time = time.time()
_DEADLINE = 145  # seconds, safety margin before 150s kill

def _calc_score(h):
    """
    Calculate the score (upper bound on C3) for a given step function h.
    Score = max_{k != 0} |(h*h)_k| / ||h||_2^2
    For binary +/- 1, ||h||_2^2 = N.
    """
    # Compute autocorrelation using FFT
    # Zero-pad to 2*N to avoid circular convolution effects (though for full linear correlation 
    # we need size >= 2N-1, but for max sidelobe, 2N is usually sufficient if we check properly)
    # Actually, linear autocorrelation length is 2N-1.
    # FFT size must be at least 2N-1. Let's use next power of 2 for speed, or just 2*N.
    size = 2 * N
    H = np.fft.fft(h, size)
    power_spectrum = H * np.conj(H)
    corr = np.fft.ifft(power_spectrum).real
    
    # The linear autocorrelation is in the first N elements of the IFFT result 
    # if we consider the standard definition C[k] = sum h[i]h[i+k].
    # Wait, np.correlate(h, h, 'full') gives lags -(N-1) to N-1.
    # FFT method: 
    # corr_full = ifft(fft(h, 2N) * conj(fft(h, 2N)))
    # The linear autocorrelation is split. 
    # Specifically, corr[k] for k=0..N-1 is in corr_full[0..N-1]
    # and corr[-k] is in corr_full[2N-k].
    # Since autocorrelation is symmetric, we just need max abs of non-zero lags.
    
    # Extract the linear autocorrelation values
    # lags 0 to N-1 are at indices 0 to N-1
    # lags -(N-1) to -1 are at indices 2N-(N-1) to 2N-1 i.e., N+1 to 2N-1
    # Index N is the wrap-around alias (should be 0 if padding is correct, but let's be safe)
    
    # Actually, a simpler way to get linear autocorrelation from FFT:
    # c = ifft(fft(h, n=2*N) * conj(fft(h, n=2*N))).real
    # c[0:N] contains C(0), C(1), ..., C(N-1)
    # c[N:2*N] contains C(-N+1), ..., C(-1) ?? 
    # Let's verify.
    # Standard result: ifft(|fft(x)|^2) gives circular autocorrelation.
    # If we zero pad to 2N, the first N values are the linear autocorrelation for lags 0..N-1.
    # The next N values are the linear autocorrelation for lags -(N-1)..-1.
    
    c = corr[:N]
    c_neg = corr[N:2*N]
    
    # We want max |C(k)| for k != 0.
    # C(0) is c[0].
    # C(k) for k>0 is c[k].
    # C(-k) is c_neg[N-k] ?? 
    # Let's just use np.correlate for correctness in the scoring function, 
    # but use FFT for the optimization loop for speed? 
    # np.correlate is O(N^2). FFT is O(N log N).
    # For N=400, N^2 = 160,000. N log N ~ 3000. FFT is ~50x faster.
    
    # Let's stick to FFT for the inner loop.
    # c_pos = c[1:N]  # lags 1 to N-1
    # c_neg = c_neg[:N-1] # lags -1 to -(N-1) mapped to indices N to 2N-1?
    # Actually, c[N:2*N] corresponds to lags -N+1 to -1.
    # So c_neg[0] is C(-N+1), c_neg[N-2] is C(-1).
    
    max_val = np.max(np.abs(np.concatenate((c[1:], c_neg[:N-1]))))
    
    norm = np.sum(h**2)
    if norm == 0:
        return 0.0
    return max_val / norm

def _calc_score_fft(h):
    """Faster version for optimization loop."""
    size = 2 * N
    H = np.fft.fft(h, size)
    power_spectrum = H * np.conj(H)
    corr = np.fft.ifft(power_spectrum).real
    
    # lags 1..N-1 are at indices 1..N-1
    # lags -1..-(N-1) are at indices N..2N-1 (specifically N+1..2N-1 for -1..-(N-1) if we ignore index N which is alias?)
    # Let's check indices carefully.
    # For zero-padded FFT of length 2N:
    # ifft(|F|^2) gives circular autocorrelation of the padded sequence.
    # Since the sequence is zero-padded to 2N, the circular autocorrelation equals the linear autocorrelation
    # for lags 0..N-1.
    # The values for negative lags are at the end of the array.
    # Specifically, C(-k) is at index 2N-k.
    # So C(-1) is at 2N-1. C(-(N-1)) is at 2N-(N-1) = N+1.
    
    # So non-zero lags are:
    # Positive: indices 1 to N-1
    # Negative: indices N+1 to 2N-1
    
    max_val = np.max(np.abs(np.concatenate((corr[1:N], corr[N+1:2*N]))))
    return max_val / N # Norm is always N for +/- 1

def construct():
    global _start_time
    
    # Initial random sequence
    h = np.random.choice([-1.0, 1.0], size=N)
    current_score = _calc_score_fft(h)
    best_h = h.copy()
    best_score = current_score
    
    # Simulated Annealing Parameters
    T = 1.0  # Initial temperature
    T_min = 1e-6
    cooling = 0.999 # Geometric cooling rate
    
    # We will run until time limit
    while time.time() - _start_time < _DEADLINE:
        if T < T_min:
            break
            
        # Choose a move: flip one bit or two bits
        if np.random.random() < 0.8:
            # Single flip
            idx = np.random.randint(N)
            h[idx] *= -1
        else:
            # Double flip
            idx1, idx2 = np.random.choice(N, 2, replace=False)
            h[idx1] *= -1
            h[idx2] *= -1
        
        new_score = _calc_score_fft(h)
        delta = new_score - current_score
        
        if delta < 0 or np.random.random() < math.exp(-delta / T):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_h = h.copy()
        else:
            # Revert move
            if np.random.random() < 0.8:
                h[idx] *= -1
            else:
                h[idx1] *= -1
                h[idx2] *= -1
        
        T *= cooling
        
        # Optional: Local search refinement every few thousand steps or if T is low
        if T < 0.01 and time.time() - _start_time < _DEADLINE - 10:
            # Greedy local search
            improved = True
            while improved and time.time() - _start_time < _DEADLINE - 5:
                improved = False
                # Try flipping each bit
                for i in range(N):
                    if time.time() - _start_time > _DEADLINE - 5:
                        break
                    h[i] *= -1
                    val = _calc_score_fft(h)
                    if val < current_score:
                        current_score = val
                        if val < best_score:
                            best_score = val
                            best_h = h.copy()
                        improved = True
                    else:
                        h[i] *= -1
    
    # Final check
    return best_h
# EVOLVE-BLOCK-END
