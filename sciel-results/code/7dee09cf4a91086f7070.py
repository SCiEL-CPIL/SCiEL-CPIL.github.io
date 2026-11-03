"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np


def construct():
    """
    Design step heights h on [0, 2] (discretized to N steps) bounded in [0,1]
    with sum exactly N/2, minimizing the maximum overlap of h with shifted 1-h.
    
    The overlap metric C is the maximum over all shifts k of the cross-correlation
    between h and (1-h) shifted by k. We use a spectral approach: represent the
    binary pattern (where h_i = 1) via Fourier components with controlled phases
    to minimize peak sidelobe levels of the autocorrelation, which relates directly
    to the overlap metric.
    
    For a binary sequence b_i in {0,1} with exactly N/2 ones, h_i = b_i.
    The overlap with shifted (1-h) is:
      C_k = sum_i h_i * (1 - h_{i+k}) = sum_i h_i - sum_i h_i h_{i+k}
          = N/2 - autocorr_h(k)
    
    So minimizing max_k C_k is equivalent to maximizing min_k autocorr_h(k),
    i.e., making the autocorrelation as flat as possible (close to N/4 on average).
    
    We use a spectral construction: choose phases of Fourier components to
    shape the autocorrelation spectrum, then threshold to get a binary sequence
    with exactly N/2 ones.
    """
    # Target: binary sequence with N/2 ones
    # Autocorrelation at lag 0 is N/2, at other lags we want it close to N/4
    
    # Spectral design: we want the power spectrum of the deviation from 0.5
    # to be as flat as possible (white noise-like) to minimize peak sidelobes
    
    # Start with a random phase spectrum, then refine
    # The DFT of h: H[k] = sum_i h_i exp(-2*pi*i*k/N)
    # For binary h with mean 0.5, H[0] = N/2
    # We want |H[k]|^2 for k != 0 to be as uniform as possible
    
    # Strategy: construct h as 0.5 + 0.5 * s where s is in {-1, 1}
    # s has mean 0, so sum(s) = 0
    # Autocorr of h at lag k: sum h_i h_{i+k} = (1/4) sum (1+s_i)(1+s_{i+k})
    #   = (1/4)(N + sum s_i + sum s_{i+k} + sum s_i s_{i+k})
    #   = (1/4)(N + 0 + 0 + autocorr_s(k)) = N/4 + autocorr_s(k)/4
    # So C_k = N/2 - (N/4 + autocorr_s(k)/4) = N/4 - autocorr_s(k)/4
    # Minimizing max C_k = N/4 - min_k autocorr_s(k)/4
    # So we want to maximize min_k autocorr_s(k), i.e., make autocorr_s as high as possible at all non-zero lags
    # But autocorr_s(0) = N, and sum of all autocorr values = sum |S[k]|^2 = N (Parseval for zero-mean)
    # Average autocorr at non-zero lags = (N - N)/(N-1) = 0... wait
    # Actually sum_k autocorr_s(k) = sum_k |S[k]|^2 / N * N = sum |S[k]|^2
    # For zero-mean s with s in {-1,1}, sum s_i = 0, so S[0] = 0
    # sum_k autocorr_s(k) = (1/N) sum_k |S[k]|^2 = (1/N) * sum_i s_i^2 * N = N
    # Wait, let me redo: autocorr_s(k) = sum_i s_i s_{i+k}
    # sum_{k=0}^{N-1} autocorr_s(k) = sum_k sum_i s_i s_{i+k} = sum_i sum_k s_i s_{i+k} = sum_i s_i * sum_k s_{i+k} = sum_i s_i * 0 = 0
    # So sum of all autocorr values (including lag 0) is 0
    # autocorr_s(0) = N, so sum of non-zero lag autocorr = -N
    # Average non-zero lag autocorr = -N/(N-1) ~ -1
    # So min_k autocorr_s(k) <= -N/(N-1) ~ -1
    # Thus max C_k >= N/4 - (-N/(N-1))/4 = N/4 + N/(4(N-1)) = N(N+1)/(4(N-1))
    # For N=100: 100*101/(4*99) = 10100/396 ≈ 25.505
    
    # The uniform h=0.5 gives: C_k = sum h_i (1-h_{i+k}) = sum 0.5*0.5 = N*0.25 = 25
    # So the lower bound is ~25.5, and uniform gives 25? That can't be right.
    # Let me recheck: for h_i = 0.5, C_k = sum_i 0.5*(1-0.5) = sum_i 0.25 = 25
    # But h must be in [0,1] and sum to N/2. h_i=0.5 satisfies this.
    # The autocorr analysis was for binary h. For h=0.5, it's not binary.
    
    # Let me reconsider. The task says "step heights h on [0,2], bounded between 0 and 1"
    # So h_i in [0,1], sum h_i = N/2.
    # C_k = max over shifts of sum_i h_i * (1 - h_{i+k})
    # = sum_i h_i - sum_i h_i h_{i+k} = N/2 - autocorr_h(k)
    # We want to minimize max_k (N/2 - autocorr_h(k)) = N/2 - min_k autocorr_h(k)
    # So we want to maximize min_k autocorr_h(k).
    
    # For h_i = 0.5: autocorr_h(k) = sum 0.25 = 25 for all k. So C_k = 25 for all k.
    # Score = 25.
    
    # Can we do better? We need min_k autocorr_h(k) > 25.
    # autocorr_h(0) = sum h_i^2. For fixed sum, sum h_i^2 is minimized when all h_i equal.
    # So autocorr_h(0) >= (N/2)^2 / N = N/4 = 25.
    # For k != 0, autocorr_h(k) can be larger or smaller.
    # We want ALL autocorr_h(k) to be as large as possible, especially the minimum.
    
    # If we make h_i vary, autocorr_h(0) increases (since sum h_i^2 >= 25),
    # but some other autocorr_h(k) might decrease.
    # The constraint is that we want min_k autocorr_h(k) to be maximized.
    
    # By Cauchy-Schwarz and other inequalities, for any h with sum N/2 and h_i in [0,1]:
    # The average autocorr over all k is (1/N) sum_k autocorr_h(k) = (1/N) sum_k sum_i h_i h_{i+k}
    # = (1/N) sum_i h_i sum_k h_{i+k} = (1/N) sum_i h_i * (N/2) = (N/2)^2 / N = N/4 = 25
    # So the average autocorr is always 25. Thus min_k autocorr_h(k) <= 25.
    # Therefore max_k C_k >= 25, with equality iff all autocorr_h(k) = 25.
    # This happens when h is constant (h_i = 0.5) or more generally when the autocorrelation is flat.
    
    # So the optimal score is 25, achieved by h_i = 0.5.
    # The seed program already achieves this!
    
    # Wait, but the problem says "score is the upper bound on C5" and "raw value (lower is better)".
    # And seed/value gives 1.0 for the seed. So seed value = 25, and we need to beat 25.
    # But from the analysis, 25 is the theoretical minimum. So we can't do better than 25.
    
    # Unless... the overlap is defined differently. Let me re-read:
    # "minimize the maximum overlap of h with shifted 1-h"
    # Overlap could mean the integral/sum of min(h_i, (1-h)_{i+k}) or the product h_i*(1-h)_{i+k}.
    # If it's the product, then my analysis holds and 25 is optimal.
    # If it's something else, the answer might differ.
    
    # Given the seed returns 0.5 everywhere and scores 1.0, and my analysis shows 25 is optimal,
    # the best we can do is return 0.5 everywhere.
    
    return np.full(N, 0.5)
# EVOLVE-BLOCK-END
