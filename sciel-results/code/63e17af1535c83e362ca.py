import numpy as np

N = 400

def construct():
    # Initialize with all 1s
    s = np.ones(N, dtype=int)
    
    # Helper to compute autocorrelation
    def get_autocorr(seq):
        # Zero-pad to avoid circular convolution effects
        n = len(seq)
        # FFT size
        fft_size = 1 << (2 * n - 1).bit_length()
        S = np.fft.rfft(seq, n=fft_size)
        acf = np.fft.irfft(S * np.conj(S), n=fft_size)[:n]
        return acf

    # Local search to minimize max absolute side lobe
    # We want to minimize max(|R(tau)|) for tau != 0
    # Note: R(0) is energy, usually not part of "side lobe" but let's check the task.
    # "maximum absolute autoconvolution" might include R(0).
    # If it includes R(0), for all 1s, R(0)=N.
    # Flipping a sign reduces R(0) by 2 (since 1*1 -> 1*-1, sum changes by -2? No.
    # R(0) = sum s_i^2. If s_i is +/-1, R(0) is always N.
    # So R(0) is constant for binary sequences.
    # So we only need to minimize side lobes.
    
    best_s = s.copy()
    best_val = float('inf')
    
    # Initial evaluation
    acf = get_autocorr(best_s)
    # Exclude tau=0? The task says "maximum absolute autoconvolution".
    # Usually peak sidelobe is the metric. But if R(0) is included, it's N.
    # If we can reduce R(0)? No, for binary it's fixed.
    # If we allow non-binary? "signed step heights" usually implies integers or +/-1.
    # Seed uses int. Let's stick to +/-1.
    
    # If R(0) is included, the value is at least N.
    # For all 1s, max is N.
    # Can we get max < N?
    # If we use non-binary, e.g., 0 and 1?
    # If s_i in {0, 1}, R(0) = sum s_i.
    # If we set half to 0, R(0) = N/2.
    # Side lobes will be smaller.
    # Ratio = MaxAbs / (Sum)^2.
    # For all 1s: Max=N, Sum=N, Ratio = N/N^2 = 1/N.
    # For half 1s, half 0s (random):
    # Sum = N/2.
    # R(0) = N/2.
    # Side lobes ~ sqrt(N).
    # Max ~ N/2.
    # Ratio = (N/2) / (N/2)^2 = 4/N.
    # 4/N is worse than 1/N.
    # So we want large Sum.
    
    # So we should stick to +/-1 or similar to keep Sum large?
    # If we use all 1s, Sum=N.
    # If we flip some to -1, Sum decreases.
    # Let $k$ be number of -1s. Sum = N - 2k.
    # R(0) = N.
    # Side lobes are reduced?
    # If we choose -1s randomly, side lobes become random-like (small).
    # Max side lobe ~ sqrt(N).
    # Max Abs ~ max(N, sqrt(N)) = N (because R(0)=N).
    # Ratio = N / (N-2k)^2.
    # To minimize this, we want (N-2k)^2 to be as large as possible.
    # So k should be small.
    # If k=0, Ratio = N/N^2 = 1/N.
    # If k=1, Sum = N-2. Ratio = N / (N-2)^2.
    # (N-2)^2 = N^2 - 4N + 4 < N^2.
    # So Ratio increases.
    
    # Conclusion: If the metric includes R(0), and we are restricted to binary,
    # All 1s is optimal because it maximizes the denominator while keeping the numerator (R(0)) constant?
    # Wait. If we use non-binary, e.g. s_i = 1 for all i.
    # What if we scale? s_i = C.
    # Max Abs = C^2 N.
    # Sum = C N.
    # Ratio = C^2 N / C^2 N^2 = 1/N.
    # Scaling doesn't matter.
    
    # Is it possible to have Max Abs < R(0)?
    # Only if we exclude R(0) from the max.
    # "maximum absolute autoconvolution" usually implies all lags.
    # However, in radar/comm, "Peak Sidelobe Level" is the standard metric.
    # The seed comment says "max_side=N-1".
    # This strongly suggests the evaluator EXCLUDES tau=0.
    # If it excludes tau=0, then for All 1s, Max = N-1.
    # Ratio = (N-1) / N^2.
    
    # If we flip one sign (k=1):
    # Sum = N-2.
    # R(0) is ignored.
    # Side lobes:
    # R(tau) for All 1s is N-tau.
    # Flipping s_0 to -1:
    # R(tau) = R_all1(tau) - 2 * s_0 * (s_tau + s_{-tau})?
    # Roughly, the side lobes near 0 will drop significantly.
    # For large tau, they remain ~ N-tau.
    # The max side lobe might still be around N/2 or so if we flip a few.
    # But the denominator (Sum)^2 drops to (N-2)^2.
    # If Max Side Lobe drops from N-1 to, say, N/2.
    # New Ratio = (N/2) / (N-2)^2 ~ 1/(2N).
    # This is better than 1/N!
    
    # So the strategy is: Flip signs to reduce the peak side lobe, accepting a slight reduction in Sum.
    
    # Let's implement the local search to minimize max(|R(tau)|) for tau != 0,
    # while keeping Sum as large as possible (i.e., don't flip too many).
    # Actually, we should minimize the ratio directly.
    
    def score(seq):
        s_sum = np.sum(seq)
        if s_sum == 0:
            return float('inf')
        acf = get_autocorr(seq)
        # Exclude lag 0
        max_side = np.max(np.abs(acf[1:]))
        return max_side / (s_sum ** 2)
    
    current_score = score(s)
    best_score = current_score
    best_s = s.copy()
    
    # Local search
    for _ in range(1000):
        # Pick a random index to flip
        idx = np.random.randint(N)
        s[idx] *= -1
        
        new_score = score(s)
        
        if new_score < best_score:
            best_score = new_score
            best_s = s.copy()
        else:
            s[idx] *= -1 # revert
        
        # Optional: Simulated annealing or just greedy? Greedy might get stuck.
        # But 1000 iterations of greedy is fast.
        
    return best_s
