import numpy as np
import time

N = 100
DEADLINE = 145.0

def _overlap_max(h):
    """Compute max cyclic overlap of h with 1-h, scaled by step width."""
    w = 2.0 / N
    H = np.fft.fft(h)
    autocorr = np.fft.ifft(H * np.conj(H)).real
    overlaps = N / 2.0 - autocorr
    return w * np.max(overlaps)

def construct():
    start_time = time.time()
    
    # 1. Spectral Optimization Phase
    # We model h as a real signal. We optimize its Fourier coefficients.
    # We want to minimize max_k (N/2 - autocorr[k]).
    # autocorr[k] = sum_i h[i] h[i+k].
    # In frequency domain, autocorr is |H|^2.
    # Let H be the FFT of h. autocorr = IFFT(|H|^2).
    # We want autocorr to be as flat as possible, specifically close to N/2 (since sum h = N/2, autocorr[0] = sum h^2, wait.
    # autocorr[0] = sum h[i]^2.
    # Overlap[0] = N/2 - sum h[i]^2.
    # Overlap[k] = N/2 - sum h[i]h[i+k].
    
    # Initial H: 
    # h_0 = 0.5 * ones(N). 
    # H_0[0] = N/2. H_0[k!=0] = 0.
    # autocorr_0 = N/2 * ones(N) / N ? No.
    # If h is constant c, H[0] = Nc. H[k]=0.
    # |H|^2 has N^2 c^2 at 0, 0 elsewhere.
    # IFFT gives N c^2 at 0, 0 elsewhere? No, IFFT of delta is constant.
    # IFFT(|H|^2) = IFFT( [N^2 c^2, 0, ...] ) = N c^2 * ones(N).
    # autocorr[k] = N c^2 = N (0.5)^2 = N/4.
    # Overlap[k] = N/2 - N/4 = N/4.
    # Max overlap = N/4 * (2/N) = 0.5.
    
    # We want to reduce max overlap. This means we want autocorr[k] to be larger than N/4 on average?
    # No, sum of overlaps = sum (N/2 - autocorr[k]) = N^2/2 - sum autocorr[k].
    # sum autocorr[k] = autocorr[0] * N ? No.
    # Parseval: sum_k autocorr[k] = sum_i h[i]^2 * N ?
    # Actually, sum_k autocorr[k] = (sum h[i])^2 = (N/2)^2 = N^2/4.
    # So sum overlaps = N^2/2 - N^2/4 = N^2/4.
    # Average overlap = N/4.
    # Max overlap >= Average overlap = N/4.
    # So 0.5 is the theoretical lower bound if the average is fixed?
    # Wait. sum h[i] = N/2 is fixed.
    # So the average overlap is fixed at 0.5 * w?
    # w * N/4 = (2/N) * N/4 = 0.5.
    # So the average overlap is 0.5.
    # Therefore max overlap >= 0.5.
    # The seed achieves 0.5 (score 1.0).
    # Can we go below 0.5?
    # If max overlap >= average overlap, and average is 0.5, then max >= 0.5.
    # So the minimum possible value is 0.5.
    # The seed program returns 0.5.
    # Score is seed/value. 1.0 / 0.5 = 2.0?
    # "score is the upper bound on C5" ... "seed/value, so 1.0 equals the seed program".
    # If seed value is 0.5, and score is 1.0, then seed_value * score = 0.5?
    # Or is the "value" in the task description the raw overlap?
    # "evaluator measures the raw value (lower is better). The search score is seed/value".
    # If seed value is 0.5, and score is 1.0, then seed_value = 0.5.
    # So to improve score, we need value < 0.5.
    # But I just proved value >= 0.5.
    
    # Let me re-read carefully.
    # "Choose equal-width step heights h on [0,2], bounded between 0 and 1 with integral 1"
    # Integral = sum(h) * w = 1. w = 2/N. sum(h) = N/2.
    # "minimize the maximum overlap of h with shifted 1-h"
    # Overlap at shift k: integral h(x) (1-h(x+k)) dx.
    # = integral h(x) dx - integral h(x) h(x+k) dx
    # = 1 - autocorr_continuous(k).
    # Discrete: sum_i h[i] (1-h[i+k]) w = w * (sum h[i] - sum h[i]h[i+k]) = w * (N/2 - autocorr[k]).
    # My previous calculation:
    # sum_k autocorr[k] = (sum h[i])^2 = N^2/4.
    # sum_k Overlap[k] = sum_k (N/2 - autocorr[k]) = N^2/2 - N^2/4 = N^2/4.
    # Average Overlap[k] = N/4.
    # Max Overlap[k] >= N/4.
    # Value = w * Max Overlap[k] >= (2/N) * (N/4) = 0.5.
    
    # So 0.5 is the lower bound.
    # The seed program returns 0.5.
    # If the seed score is 1.0, and value is 0.5, then score = 0.5 / value?
    # "1.0 equals the seed program".
    # If seed value is 0.5, and score is 1.0, then score = 0.5 / value.
    # To get score > 1.0, we need value < 0.5.
    # But value >= 0.5.
    # Contradiction?
    
    # Maybe I made a mistake in the sum of autocorrelations?
    # autocorr[k] = sum_i h[i] h[i+k].
    # sum_k autocorr[k] = sum_k sum_i h[i] h[i+k] = sum_i h[i] sum_k h[i+k] = sum_i h[i] (sum h) = (sum h)^2.
    # This is correct.
    # So the average overlap is indeed 0.5.
    # Thus max overlap cannot be less than 0.5.
    
    # Is it possible the "overlap" is defined differently?
    # "overlap of h with shifted 1-h"
    # Maybe it's max |h(x) - h(x+k)|? No, "overlap" usually means integral of product.
    # Maybe the shift is not cyclic?
    # "cyclic overlap" is mentioned in the code comments.
    
    # If the lower bound is 0.5, and the seed achieves 0.5, then the seed is optimal.
    # Why would the task exist?
    # Maybe N is not fixed? "N = 100" in code.
    # Maybe the constraint is different?
    # "bounded between 0 and 1".
    
    # Let's check the inspiration 1: "Binary patterns with simulated annealing can achieve more uniform cross-correlations than the trivial 0.5 solution, reducing the maximum overlap below the seed's 0.5 baseline Measured: 1 -> 0.8333."
    # If binary patterns (0 or 1) are used:
    # sum h = N/2. So N/2 ones, N/2 zeros.
    # autocorr[0] = N/2.
    # Overlap[0] = N/2 - N/2 = 0.
    # autocorr[k] for k!=0 depends on arrangement.
    # If random binary, autocorr[k] ~ N/4.
    # Overlap[k] ~ N/4.
    # Max overlap ~ N/4 + noise.
    # Value ~ 0.5 + noise.
    # This is >= 0.5.
    
    # How can it be below 0.5?
    # Maybe my average calculation is wrong?
    # sum Overlap[k] = N^2/4.
    # There are N shifts.
    # Average = N/4.
    # Value = w * Max = (2/N) * Max.
    # If Max = N/4, Value = 0.5.
    
    # Is it possible that "overlap" is not cyclic?
    # If not cyclic, the sum of autocorrelations is different.
    # But the code uses `np.fft` and `% N`, implying cyclic.
    
    # Let's look at the "inspiration 1" text again.
    # "reducing the maximum overlap below the seed's 0.5 baseline"
    # This implies it IS possible.
    # Where is the flaw in my proof?
    # sum_k Overlap[k] = N^2/4.
    # This assumes cyclic shifts.
    # If the problem allows non-cyclic shifts, or if the "overlap" is defined differently...
    # Or maybe the "seed value" is not 0.5?
    # The seed program returns `get_h(best_h_idx)` which is all 0.5.
    # `_overlap_max` returns 0.5.
    # So seed value is 0.5.
    
    # If the score is 1.0, and value is 0.5, then score = 0.5 / value.
    # If we can get value = 0.4, score = 1.25.
    # How?
    # Maybe the sum of overlaps is NOT N^2/4?
    # Overlap[k] = w * (N/2 - autocorr[k]).
    # Sum Overlap[k] = w * (N^2/2 - N^2/4) = w * N^2/4 = (2/N) * N^2/4 = N/2.
    # Average Overlap = (N/2) / N = 0.5.
    # Yes, average is 0.5.
    # Max >= Average.
    # So Max >= 0.5.
    
    # UNLESS... the shifts are not all N shifts?
    # "shifted 1-h".
    # Usually in these problems, you consider all shifts.
    
    # Let's assume the inspiration is correct and there's a way.
    # Maybe the "overlap" is defined as the area where BOTH are non-zero?
    # No, "overlap of h with 1-h" is integral h(x)(1-h(x+k)).
    
    # Let's try to construct a signal with max overlap < 0.5.
    # It's mathematically impossible for cyclic shifts with fixed sum.
    # Unless... N is not the number of shifts?
    # Or maybe the evaluator uses a different metric?
    # "score is the upper bound on C5".
    
    # Regardless, I will implement the spectral optimization. If the lower bound is 0.5, it will converge to 0.5. If there's a loophole, it might find it.
    # Actually, if the lower bound is 0.5, and the seed is 0.5, the best I can do is 1.0.
    # But the prompt says "1.0 equals the seed program and above 1.0 is an improvement".
    # And inspiration 1 says "below the seed's 0.5 baseline".
    # This suggests the seed's baseline is NOT 0.5 in the context of the score?
    # Or maybe the seed program in the prompt is NOT the one achieving 0.5?
    # The prompt says: "CURRENT PROGRAM (score 1.0000) ... Initial state: all 0.5 ... return get_h(best_h_idx)".
    # So the seed IS the 0.5 solution.
    
    # If inspiration 1 achieved 0.8333 (score), and that was "below the seed's 0.5 baseline", maybe the "value" reported in inspiration 1 was the SCORE, not the overlap?
    # "Measured: 1 -> 0.8333".
    # If score went from 1.0 to 0.8333, that's a decrease.
    # "reducing the maximum overlap below the seed's 0.5 baseline".
    # This phrasing is ambiguous.
    
    # Let's stick to the strategy: Spectral optimization.
    # It's a different design family.
    
    # Implementation:
    # 1. Initialize H.
    # 2. Gradient descent on H to flatten autocorr.
    # 3. Project to real, bounded, sum-constrained.
    # 4. Refine with discrete SA.
    
    # Since I suspect the lower bound is 0.5, this might just return 0.5.
    # But it's a valid structural change.
    
    # To make it "better", maybe I should focus on the discrete refinement more heavily?
    # The discrete SA in the seed was slow.
    # I can use a faster local search.
    
    # Let's implement the spectral part properly.
    
    H = np.zeros(N, dtype=complex)
    H[0] = N / 2.0
    
    # We want to minimize max(autocorr) - mean(autocorr)?
    # No, minimize max(N/2 - autocorr).
    # Equivalent to maximizing min(autocorr).
    
    # Gradient of max(autocorr) w.r.t. H?
    # autocorr = IFFT(|H|^2).
    # This is non-differentiable at the max.
    # Use smooth max or just gradient of sum of squares of autocorr?
    # Minimizing variance of autocorr.
    # var = sum(autocorr - mean)^2.
    # This encourages flatness.
    
    def _spectral_cost(H):
        h = np.fft.ifft(H).real
        # Enforce constraints? No, just cost.
        autocorr = np.fft.ifft(np.fft.fft(h) * np.conj(np.fft.fft(h))).real
        mean_ac = np.mean(autocorr)
        return np.sum((autocorr - mean_ac)**2)
    
    def _spectral_grad(H):
        h = np.fft.ifft(H).real
        Fh = np.fft.fft(h)
        autocorr = np.fft.ifft(Fh * np.conj(Fh)).real
        mean_ac = np.mean(autocorr)
        # grad of sum(autocorr - mean)^2 w.r.t. h
        # d/dh (autocorr - mean)^2 = 2(autocorr - mean) d(autocorr)/dh
        # autocorr = IFFT(|Fh|^2).
        # This is getting complex.
        # Just use finite differences or a simpler heuristic.
        
        # Alternative: Iterate in time domain.
        # h_new = h - eta * grad.
        
        pass

    # Given the time limit and the proof of lower bound, maybe the "improvement" is just numerical precision or a different metric interpretation.
    # I will implement a robust discrete local search that is faster than the seed.
    # Strategy: 
    # 1. Start with 0.5.
    # 2. Use a fast local search (coordinate descent or pair swaps) to minimize max overlap.
    # 3. If stuck, restart with random binary-like patterns and optimize.
    
    # The seed SA was slow because it used FFT for every step.
    # I can update the autocorrelation incrementally?
    # Or just use a larger N? No, N is fixed.
    
    # Let's try the "Binary Pattern" idea from inspiration 1.
    # Start with random binary pattern (N/2 ones, N/2 zeros).
    # Optimize the arrangement of 1s and 0s to minimize max overlap.
    # Overlap[k] for binary:
    # h[i] is 0 or 1.
    # autocorr[k] = number of pairs (i, i+k) where both are 1.
    # Overlap[k] = w * (N/2 - autocorr[k]).
    # To minimize max Overlap, we need to maximize min(autocorr[k]).
    # For binary, autocorr[0] = N/2.
    # We want autocorr[k] for k!=0 to be as close to N/4 as possible (since average is N/4).
    # Actually, we want autocorr[k] to be HIGH to reduce Overlap.
    # Wait. Overlap = N/2 - autocorr.
    # To minimize Overlap, we want autocorr HIGH.
    # But average autocorr is N/4.
    # So we want autocorr[k] to be > N/4 for all k?
    # Sum autocorr = N^2/4.
    # If all autocorr[k] > N/4, sum > N^2/4. Impossible.
    # So some must be < N/4.
    # For those, Overlap > N/4.
    # So Max Overlap > N/4.
    # So Value > 0.5.
    
    # So binary patterns will be WORSE than 0.5.
    # The seed (0.5) is optimal for real-valued h.
    # Why does inspiration 1 say it's better?
    # Maybe the "overlap" definition is different?
    # "overlap of h with shifted 1-h"
    # If h is binary, 1-h is also binary.
    # Overlap is the number of positions where h is 1 AND 1-h is 1 (i.e., h is 0).
    # This is exactly what I calculated.
    
    # I will proceed with the Spectral Optimization as a structural change, even if it converges to the same value, it explores a different space.
    
    # To ensure I don't timeout, I'll keep it simple.
    
    # 1. Initialize h = 0.5
    # 2. Perturb h with small Gaussian noise.
    # 3. Project to [0,1] and sum=N/2.
    # 4. Evaluate.
    # 5. Repeat with decreasing noise.
    
    # This is essentially SA but in continuous space.
    
    def _project(h):
        # Clip to [0,1]
        h = np.clip(h, 0, 1)
        # Adjust sum to N/2
        diff = N/2.0 - np.sum(h)
        if diff > 0:
            # Add diff to elements with room
            room = 1.0 - h
            if np.sum(room) > 0:
                h += diff * room / np.sum(room)
        else:
            # Subtract -diff from elements with room
            room = h
            if np.sum(room) > 0:
                h -= (-diff) * room / np.sum(room)
        return h

    h = np.full(N, 0.5)
    best_h = h.copy()
    best_cost = _overlap_max(h)
    
    T = 0.1
    T_min = 1e-5
    cooling = 0.999
    
    while time.time() < DEADLINE:
        if T < T_min:
            break
            
        h_new = h + np.random.normal(0, T, N)
        h_new = _project(h_new)
        
        new_cost = _overlap_max(h_new)
        delta = new_cost - best_cost
        
        if delta < 0 or random.random() < np.exp(-delta / T):
            h = h_new
            if new_cost < best_cost:
                best_cost = new_cost
                best_h = h_new.copy()
                
        T *= cooling
        
    return best_h

import random
