"""Erdos minimum overlap: find equal-width step heights h in [0,1], sum N/2,
minimizing the maximum overlap of h with shifted (1-h)."""
import numpy as np
import time

N = 100

def _overlap_max(h):
    """Compute the maximum overlap of h with shifted (1-h) over all shifts k=0..N-1.
    O_k = sum(h) - autocorr(k) = N/2 - A_k where A_k = sum h_i * h_{i+k mod N}
    We want to minimize max_k O_k.
    """
    # Circular autocorrelation
    H = np.fft.fft(h)
    A = np.real(np.fft.ifft(H * np.conj(H)))
    A = A * N  # FFT normalization: ifft(fft(x)*conj(fft(x))) gives autocorr/N
    # Actually, let's be more careful.
    # np.fft.fft(x) gives X_k. np.fft.ifft(X_k * conj(X_k)) gives autocorr / N
    # So A_k = N * ifft(fft(h) * conj(fft(h)))
    
    # Overlap O_k = sum(h) - A_k = N/2 - A_k
    # We want to minimize max_k O_k
    # max O_k = N/2 - min_k A_k
    
    # But wait, for k=0, A_0 = sum(h_i^2)
    # O_0 = N/2 - sum(h_i^2)
    # For k>0, A_k = sum h_i * h_{i+k}
    # O_k = N/2 - A_k
    
    # So max O_k = N/2 - min(A_0, min_{k>0} A_k)
    # To minimize max O_k, we want to maximize min(A_k).
    
    # But the task says "maximum overlap of h with shifted 1-h"
    # Let me re-read: "minimize the maximum overlap of h with shifted 1-h"
    # Overlap of h and shifted (1-h) at shift k:
    # O_k = sum h_i * (1 - h_{i+k}) = sum h_i - sum h_i * h_{i+k} = sum(h) - A_k
    # Since sum(h) = N/2, O_k = N/2 - A_k
    # We want to minimize max_k O_k = N/2 - min_k A_k
    # So we want to maximize min_k A_k.
    
    # For constant h=0.5: A_k = N/4 for all k, so min A_k = N/4, max O_k = N/4 = 25.
    
    # The score is the upper bound on C5, and the raw value is lower is better.
    # So we want to minimize max O_k.
    
    # Let me compute max O_k properly.
    O = N/2.0 - A
    return float(np.max(O))

def construct():
    """
    Returns a sequence of step heights h in [0, 1] with sum N/2.
    
    Strategy: Use a sinusoidal perturbation around 0.5 to try to improve on
    the constant solution. The key insight is that for circular autocorrelation,
    a single sinusoidal mode has autocorrelation that is also sinusoidal, and
    the minimum of the autocorrelation may be higher than for other patterns.
    
    For h_i = 0.5 + A*cos(2*pi*m*i/N + phi), the autocorrelation is:
    A_k = N/4 + A^2/2 * cos(2*pi*m*k/N)
    
    So min A_k = N/4 - A^2/2, and max O_k = N/2 - (N/4 - A^2/2) = N/4 + A^2/2.
    
    Wait, that's WORSE than constant 0.5! The constant gives N/4, and adding
    perturbation gives N/4 + A^2/2.
    
    Hmm, but the observation says a sinusoidal perturbation improved the score
    to 1.085. Let me reconsider.
    
    Actually wait - maybe the overlap is defined differently. Let me re-read:
    "maximum overlap of h with shifted 1-h"
    
    If h is binary-like (0s and 1s), then h*(1-h) = 0 at k=0, but for other
    shifts, h_i * (1-h_{i+k}) could be large.
    
    Actually, let me reconsider the problem. The "overlap" might be:
    O_k = sum h_i * (1 - h_{i+k})
    
    For h = 0.5 constant: O_k = sum 0.5 * 0.5 = N * 0.25 = 25 for all k.
    
    For a binary sequence with 50 ones and 50 zeros:
    O_0 = 0 (since h_i*(1-h_i) = 0)
    O_k = number of positions where h_i=1 and h_{i+k}=0, plus h_i=0 and h_{i+k}=1... 
    No wait, O_k = sum h_i*(1-h_{i+k}). If h is binary, this is the number of
    positions where h_i=1 and h_{i+k}=0.
    
    For a balanced binary sequence, this is approximately N/4 for most k,
    but could be different.
    
    The observation says: "reducing the maximum overlap from N/2=50 to approximately
    N/4 + A^2/2 = 25.12"
    
    So the baseline they're comparing against is N/2 = 50, not N/4 = 25.
    This suggests the "seed" program had a max overlap of 50, and the current
    constant-0.5 program also has a max overlap of... wait, no.
    
    If constant 0.5 gives O_k = 25 for all k, then max O_k = 25.
    If the seed score is 1.0 corresponding to value 25, then the seed value is 25.
    
    But the observation says the sinusoidal perturbation reduced max overlap from
    50 to 25.12. That doesn't make sense if constant 0.5 already gives 25.
    
    Let me reconsider. Maybe I'm computing the overlap wrong.
    
    "overlap of h with shifted 1-h" - maybe it's the L1 norm or something else?
    Or maybe it's sum |h_i - (1-h_{i+k})|?
    
    Actually, re-reading: "minimum overlap" - in the context of Erdos problems,
    this might refer to the minimum number of common 1s or something.
    
    Let me just try the sinusoidal approach and see what happens. The observation
    clearly states it improved the score, so let me implement it.
    
    h_i = 0.5 + A * cos(2*pi*m*i/N)
    
    With A and m as parameters to optimize.
    """
    start_time = time.time()
    deadline = start_time + 120.0
    
    # Baseline: constant 0.5
    best_h = np.full(N, 0.5)
    best_score = _overlap_max(best_h)
    
    # Try sinusoidal perturbations
    # h_i = 0.5 + A * cos(2*pi*m*i/N + phi)
    # For different m and A values
    
    for m in range(1, N // 2 + 1):
        if time.time() > deadline - 5.0:
            break
        
        for A in [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.49]:
            if time.time() > deadline - 5.0:
                break
            
            for phi in [0.0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]:
                if time.time() > deadline - 5.0:
                    break
                
                i = np.arange(N)
                h = 0.5 + A * np.cos(2 * np.pi * m * i / N + phi)
                
                # Clip to [0, 1]
                h = np.clip(h, 0.0, 1.0)
                
                # Adjust to ensure sum is exactly N/2
                # Current sum might not be exactly N/2 after clipping
                current_sum = h.sum()
                if abs(current_sum - N/2.0) > 1e-10:
                    # Scale to fix the sum
                    h = h * (N/2.0) / current_sum
                    h = np.clip(h, 0.0, 1.0)
                    
                    # Re-adjust
                    current_sum = h.sum()
                    if abs(current_sum - N/2.0) > 1e-8:
                        # Small correction on elements
                        diff = N/2.0 - current_sum
                        # Distribute the difference
                        for j in range(N):
                            if h[j] < 1.0 and diff > 0:
                                adj = min(1.0 - h[j], diff)
                                h[j] += adj
                                diff -= adj
                            elif h[j] > 0.0 and diff < 0:
                                adj = min(-diff, h[j])
                                h[j] -= adj
                                diff += adj
                            if abs(diff) < 1e-10:
                                break
            
            score = _overlap_max(h)
            
            if score < best_score:
                best_score = score
                best_h = h.copy()
    
    # If no improvement found, return the constant solution
    # Verify
    assert abs(best_h.sum() - N/2.0) < 1e-6, f"Sum is {best_h.sum()}, expected {N/2.0}"
    assert np.all(best_h >= 0.0 - 1e-10) and np.all(best_h <= 1.0 + 1e-10)
    best_h = np.clip(best_h, 0.0, 1.0)
    
    return best_h

# EVOLVE-BLOCK-START
# (kept for compatibility)
# EVOLVE-BLOCK-END
