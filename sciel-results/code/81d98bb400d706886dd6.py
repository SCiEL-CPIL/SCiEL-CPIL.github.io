"""Erdos minimum overlap: construct 100 step heights to minimize max correlation."""
import numpy as np

N = 100

def construct():
    # The objective is to minimize max(correlate(h, 1-h)) / N * 2.
    # Let u_i = h_i - 0.5. Then sum(u) = 0.
    # correlate(h, 1-h)_k = sum h_i (1 - h_{i+k}) = sum h_i - sum h_i h_{i+k}
    # = N*0.5 - sum (0.5+u_i)(0.5+u_{i+k})
    # = N/2 - (N/4 + 0.5 sum u_i + 0.5 sum u_{i+k} + sum u_i u_{i+k})
    # = N/4 - sum u_i u_{i+k}
    # So value = (N/4 - max_k sum u_i u_{i+k}) / N * 2 = 0.5 - 2/N * max_k C_u(k)
    # To minimize value, we want to maximize max_k C_u(k).
    # C_u(k) = sum u_i u_{i+k} is the autocorrelation of u.
    # We need max_k C_u(k) to be as large as possible.
    # This is equivalent to finding a zero-mean sequence u with max |u_i| <= 0.5
    # such that its maximum autocorrelation peak is maximized.
    
    # A simple way to get a high positive autocorrelation peak at k=0 (which is always positive)
    # is to make the sequence have a strong "bump" or "pulse".
    # The seed u=0 gives C_u(0)=0, so value = 0.5.
    # We want C_u(0) > 0. C_u(0) = sum u_i^2.
    # Maximize sum u_i^2 subject to sum u_i = 0 and |u_i| <= 0.5.
    # The maximum sum of squares is achieved by making u_i as close to +/-0.5 as possible.
    # Since sum u_i = 0, we need roughly half to be +0.5 and half to be -0.5.
    # Let 50 be +0.5 and 50 be -0.5.
    # Then sum u_i^2 = 50*(0.25) + 50*(0.25) = 12.5.
    # value = 0.5 - 2/100 * 12.5 = 0.5 - 0.25 = 0.25.
    
    # Can we do better? The max autocorrelation is max_k C_u(k).
    # For the block sequence (+0.5...+0.5, -0.5...-0.5), C_u(0)=12.5.
    # What about C_u(k) for k>0?
    # If the first 50 are +0.5 and last 50 are -0.5:
    # C_u(k) = sum_{i=0}^{N-k-1} u_i u_{i+k}
    # For k=1: sum_{i=0}^{48} (0.5)(0.5) + u_49 u_50 + sum_{i=50}^{98} (-0.5)(-0.5)
    # = 49*0.25 + (0.5)(-0.5) + 49*0.25 = 49 - 0.25 = 48.75.
    # Wait, 49*0.25 = 12.25. So 12.25 - 0.25 + 12.25 = 24.25.
    # This is larger than C_u(0)=12.5!
    # So max_k C_u(k) is at least 24.25.
    # value = 0.5 - 2/100 * 24.25 = 0.5 - 0.485 = 0.015.
    
    # Let's verify the block structure:
    # h = [0.5]*50 + [0.0]*50  -> u = [0.0]*50 + [-0.5]*50. Sum u = -25. Not zero mean.
    # We need sum u = 0.
    # Let u_i = +0.5 for i in [0, 49], u_i = -0.5 for i in [50, 99].
    # Sum u = 50*0.5 + 50*(-0.5) = 0. Correct.
    # h_i = 1.0 for i in [0, 49], h_i = 0.0 for i in [50, 99].
    # h is in [0, 1]. Correct.
    
    # Let's compute the value for this block sequence.
    # u = [0.5]*50 + [-0.5]*50
    # C_u(k) = sum u_i u_{i+k}
    # C_u(0) = 50*0.25 + 50*0.25 = 25.
    # C_u(1) = sum_{i=0}^{98} u_i u_{i+1}
    # i=0..48: u_i=0.5, u_{i+1}=0.5 -> 49 * 0.25 = 12.25
    # i=49: u_49=0.5, u_50=-0.5 -> -0.25
    # i=50..98: u_i=-0.5, u_{i+1}=-0.5 -> 49 * 0.25 = 12.25
    # C_u(1) = 12.25 - 0.25 + 12.25 = 24.25
    # C_u(49) = sum_{i=0}^{50} u_i u_{i+49}
    # i=0: u_0=0.5, u_49=0.5 -> 0.25
    # i=1: u_1=0.5, u_50=-0.5 -> -0.25
    # ...
    # i=49: u_49=0.5, u_98=-0.5 -> -0.25
    # i=50: u_50=-0.5, u_99=-0.5 -> 0.25
    # Let's count.
    # i=0: +0.25
    # i=1..49: 49 terms of -0.25
    # i=50: +0.25
    # C_u(49) = 0.25 - 49*0.25 + 0.25 = 0.5 - 12.25 = -11.75
    
    # C_u(50) = sum_{i=0}^{49} u_i u_{i+50}
    # u_i = 0.5, u_{i+50} = -0.5.
    # 50 * (0.5 * -0.5) = 50 * -0.25 = -12.5
    
    # The maximum seems to be at k=0 with value 25.
    # Let's check k=2.
    # i=0..47: 48 * 0.25 = 12
    # i=48: 0.5*0.5 = 0.25
    # i=49: 0.5*-0.5 = -0.25
    # i=50..97: 48 * 0.25 = 12
    # C_u(2) = 12 + 0.25 - 0.25 + 12 = 24.
    
    # So max C_u(k) = 25 at k=0.
    # value = 0.5 - 2/100 * 25 = 0.5 - 0.5 = 0.0.
    
    # Wait, is value 0.0 possible?
    # value = max_k (N/4 - C_u(k)) / N * 2
    # If max C_u(k) = N/4 = 25, then value = 0.
    # C_u(0) = sum u_i^2. Max sum u_i^2 with sum u_i=0 and |u_i|<=0.5 is 25.
    # So yes, value can be 0.
    
    # Let's double check the formula.
    # correlate(h, 1-h)_k = sum_{i=0}^{N-k-1} h_i (1-h_{i+k})
    # = sum h_i - sum h_i h_{i+k}
    # h_i = 0.5 + u_i
    # sum h_i over i=0..N-k-1 is NOT N/2 unless k=0.
    # Ah, the sum is only over N-k terms.
    # Let S_k = sum_{i=0}^{N-k-1} h_i.
    # correlate_k = S_k - sum h_i h_{i+k}
    # For the block sequence h=[1]*50+[0]*50:
    # k=0: S_0 = 50. sum h_i h_i = 50. correlate_0 = 50-50=0.
    # k=1: S_1 = 49*1 + 1*0 = 49. sum h_i h_{i+1} = 49*1*1 + 1*1*0 + ... wait.
    # h=[1,1,...,1,0,0,...,0]
    # k=1: h_i h_{i+1}:
    # i=0..48: 1*1=1 (49 times)
    # i=49: 1*0=0
    # i=50..98: 0*0=0
    # sum = 49.
    # correlate_1 = 49 - 49 = 0.
    # k=50: S_50 = 50*1 + 0 = 50.
    # h_i h_{i+50}:
    # i=0..49: 1*0=0.
    # sum = 0.
    # correlate_50 = 50 - 0 = 50.
    
    # So max correlate is 50.
    # value = 50 / 100 * 2 = 1.0.
    # This is the same as the seed!
    
    # My previous derivation assumed the sum was over all N terms.
    # The correlation is "full" mode, so for lag k, it sums N-k terms.
    # Let's re-derive.
    # v_k = sum_{i=0}^{N-k-1} h_i (1-h_{i+k})
    # We want to minimize max v_k.
    # Seed h=0.5: v_k = sum (0.5*0.5) = (N-k)*0.25.
    # Max is at k=0: N*0.25 = 25.
    # value = 25/100 * 2 = 0.5.
    # Wait, the seed value is 1.0 in the score, which is seed/value.
    # If seed value is 0.5, score is 1/0.5 = 2? No, "1.0 equals the seed program".
    # So the raw value of the seed must be 1.0?
    # "value = max(...) / len(h) * 2"
    # Seed: max = 25. value = 25/100 * 2 = 0.5.
    # Score = seed_value / child_value? No, "search score is seed/value".
    # If seed value is 0.5, and score is 1.0, then seed_value in the formula must be 0.5?
    # Or maybe the "seed value" in the score formula is the value of the seed program.
    # Yes. So if I get value 0.25, score is 0.5/0.25 = 2.0.
    
    # Back to the block sequence h=[1]*50+[0]*50.
    # v_50 = 50. value = 50/100*2 = 1.0.
    # So the block sequence gives the same value as the seed.
    
    # Can we do better than 0.5 (seed value)?
    # We need max v_k < 25.
    # v_k = S_k - sum h_i h_{i+k}.
    # To minimize v_k, we want S_k to be small and sum h_i h_{i+k} to be large.
    # Or just make v_k as flat as possible.
    
    # Consider h_i = 0.5 + A * cos(2*pi*i/N).
    # This has mean 0.5.
    # Let's try a small A.
    # This is a standard technique in signal processing to minimize peak autocorrelation.
    # However, we have a one-sided constraint h_i in [0,1].
    
    # Let's try to minimize the maximum of v_k directly.
    # This is a non-convex optimization problem.
    # But N=100 is small.
    # We can use a simple gradient-free optimization like Nelder-Mead or just random search with restarts.
    # Given the 120s limit, a simple local search might work.
    
    # Alternative: The problem is equivalent to finding a binary sequence (approximately)
    # that minimizes the maximum cross-correlation with its complement.
    # For binary sequences, this is related to Barker sequences or low autocorrelation sequences.
    # But here we have continuous values.
    
    # Let's try a simple randomized local search.
    # Start with seed.
    # Randomly perturb one element.
    # If value decreases, keep.
    
    import random
    random.seed(42)
    h = np.full(N, 0.5)
    
    def calc_value(h):
        corr = np.correlate(h, 1-h, mode="full")
        return np.max(corr) / N * 2
    
    current_val = calc_value(h)
    
    for step in range(10000):
        i = random.randint(0, N-1)
        old = h[i]
        # Perturb
        delta = random.uniform(-0.1, 0.1)
        new_val = old + delta
        if new_val < 0 or new_val > 1:
            continue
        h[i] = new_val
        new_score = calc_value(h)
        if new_score < current_val:
            current_val = new_score
        else:
            h[i] = old
    
    return h
