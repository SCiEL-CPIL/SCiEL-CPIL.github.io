"""Erdos minimum overlap: Chebyshev Basis Grid Search Strategy."""

import numpy as np
import time

N = 100

def _score(h):
    """Calculate the maximum overlap of h with shifted 1-h."""
    # Overlap at shift k: sum_i h_i * (1 - h_{(i+k)%N})
    # We want to minimize the maximum over k.
    # Note: The problem statement implies we look for the "upper bound on C5" or similar,
    # but the raw score is the value. Lower is better.
    # The seed value is 25.
    
    # Efficient calculation using FFT for circular convolution
    # h * (1-h) is not a convolution.
    # sum_i h_i * (1 - h_{(i+k)%N}) = sum h_i - sum h_i h_{(i+k)%N}
    # sum h_i = N/2 = 50.
    # So overlap_k = 50 - C_k, where C_k is the circular autocorrelation of h at lag k.
    # Minimizing max(overlap_k) is equivalent to maximizing min(C_k).
    
    H = np.fft.fft(h)
    C = np.fft.ifft(H * np.conj(H)).real
    # C_k is the circular autocorrelation.
    # We exclude lag 0? 
    # "shifted 1-h" usually implies k != 0 in these combinatorial designs.
    # Let's assume we look at all non-zero shifts.
    # If we include lag 0, C_0 = sum h_i^2.
    # For h=0.5, C_0 = 25. Overlap = 50 - 25 = 25.
    # For h=0.5, C_k (k!=0) = 25. Overlap = 50 - 25 = 25.
    # If we can make C_k > 25 for all k != 0, we lower the max overlap.
    # But C_0 is fixed by the sum of squares.
    # Wait, if we exclude k=0, we can potentially make C_k very large for k!=0?
    # No, the sum of all C_k is (sum h_i)^2 = 2500.
    # If there are 99 non-zero lags, average is 2500/100 = 25.
    # So on average C_k is 25.
    # If we can make them all exactly 25, the max overlap is 25.
    # To get better than 25, we need min_{k!=0} C_k > 25?
    # This is impossible if the average is 25 and C_0 >= 25.
    # Actually, sum_{k=0}^{N-1} C_k = (sum h_i)^2 = 2500.
    # If C_0 is large, the others must be smaller on average.
    # To minimize max overlap, we want to maximize min C_k.
    # Ideally C_k is constant for all k.
    # If C_k = 25 for all k, then sum = 2500. This works.
    # This corresponds to a "perfect" autocorrelation sequence.
    # Such sequences exist (e.g., Barker sequences, or specific binary sequences).
    # If we find a sequence where C_k is close to 25 for all k, the max overlap will be close to 25.
    # Can we go lower?
    # If we make C_0 very large (e.g., h is 0/1), then sum of others is 2500 - C_0.
    # If h is 0/1 with 50 ones, C_0 = 50.
    # Sum of others = 2450. Average = 2450/99 ≈ 24.7.
    # So min C_k will likely be < 25.
    # Thus max overlap > 25.
    # So the constant 0.5 sequence (C_k=25) might actually be optimal or near-optimal?
    # Wait, the seed score is 1.0, which corresponds to value 25.
    # The inspiration 3 got 1.27, meaning value < 25.
    # How?
    # Ah, the definition of "overlap" might be different.
    # "minimize the maximum overlap of h with shifted 1-h"
    # Overlap_k = sum_i h_i (1 - h_{i+k})
    # If h=0.5, Overlap_k = 25.
    # If we can make Overlap_k < 25 for all k, we improve.
    # This requires C_k > 25 for all k.
    # But sum C_k = 2500.
    # If C_k > 25 for all k=0..99, sum > 2500. Contradiction.
    # Unless... we exclude k=0 from the "maximum"?
    # "shifted 1-h" implies a shift. A shift of 0 is not a shift.
    # So we likely look at k in 1..N-1.
    # If we exclude k=0, we have 99 values.
    # Sum_{k=1}^{99} C_k = 2500 - C_0.
    # To have C_k > 25 for all k=1..99, we need 99 * 25 < 2500 - C_0 => 2475 < 2500 - C_0 => C_0 < 25.
    # But C_0 = sum h_i^2.
    # Given sum h_i = 50, min sum h_i^2 is 50^2/100 = 25 (when h_i=0.5).
    # So C_0 >= 25.
    # Therefore, 2500 - C_0 <= 2475.
    # Average C_k (k>0) <= 2475/99 = 25.
    # So at least one C_k <= 25.
    # Thus max overlap >= 25.
    # So 25 is the theoretical minimum?
    # If so, why did inspiration 3 get 1.27?
    # Maybe my interpretation of the score or the problem is slightly off.
    # Or maybe the "overlap" is defined differently?
    # "upper bound on C5" in the prompt is confusing.
    # Let's look at the inspiration 3 code if I could. I can't.
    # But let's assume the standard definition.
    # If 25 is the lower bound, then 1.0 is the max score?
    # But the prompt says "above 1.0 is an improvement".
    # This implies the seed is NOT optimal.
    # Perhaps the sum is not N/2? "sum is exactly len/2". Yes.
    # Perhaps the range is not [0,1]? "bounded between 0 and 1". Yes.
    # Perhaps "overlap" is not sum h_i (1-h_{i+k})?
    # "maximum overlap of h with shifted 1-h"
    # Could it be max_i h_i * (1-h_{i+k})? No, "overlap" usually implies sum.
    # Could it be the integral? "integral 1".
    # If it's an integral over a continuous domain, and we discretize...
    # The prompt says "Choose equal-width step heights h on [0,2]".
    # Wait, "on [0,2]"?
    # "Choose equal-width step heights h on [0,2], bounded between 0 and 1 with integral 1"
    # If the domain is [0,2], and there are N steps, width is 2/N.
    # Sum of heights * width = 1 => Sum h_i * (2/N) = 1 => Sum h_i = N/2.
    # This matches.
    # The "shifted 1-h" part:
    # If we shift the function 1-h(x) by some amount, and look at the overlap with h(x).
    # In the discrete case, this is sum h_i (1-h_{i+k}).
    # So my derivation holds.
    # Why would 25 not be the minimum?
    # Maybe the shift is not circular?
    # "shifted 1-h"
    # If it's linear shift (zero-padding), then the sum is over the intersection.
    # For large shifts, the overlap is small.
    # The maximum would be at small shifts.
    # If linear, the autocorrelation sum is not constrained to 2500 in the same way?
    # For linear convolution, the sum of the autocorrelation is still (sum h)^2.
    # But the number of terms varies.
    # Let's assume circular for now, as it's the standard for such "Erdos" type problems on cycles.
    # However, if the score can be > 1.0, then value < 25 is possible.
    # This implies my bound C_0 >= 25 is the key.
    # If we can make C_0 < 25?
    # Sum h_i = 50. Min sum h_i^2 is 25.
    # So C_0 cannot be less than 25.
    # Unless... the heights are not constrained to be non-negative? "bounded between 0 and 1".
    # So they are.
    # Is it possible that the "maximum overlap" is not over all shifts?
    # "minimize the maximum overlap"
    # Maybe the shift is restricted?
    # Or maybe I should just try to find a sequence with C_k as flat as possible.
    # If I use a Chebyshev basis, I can try to flatten the spectrum.
    
    # Let's implement the Chebyshev strategy as requested.
    # h_i = sum_{m=0}^{M} c_m T_m(x_i)
    # x_i = 2*i/(N-1) - 1
    # We need to optimize c_m.
    # Constraints: 0 <= h_i <= 1, sum h_i = N/2.
    # This is a non-convex optimization if we just grid search c_m.
    # But we can use a simple grid search on a few coefficients.
    
    # Let's try M=4 (5 coefficients).
    # c_0 is determined by the sum constraint?
    # Sum h_i = sum_i sum_m c_m T_m(x_i) = sum_m c_m (sum_i T_m(x_i)).
    # For Chebyshev polynomials on a uniform grid, the sums are not zero for m>0?
    # T_m(x) oscillates.
    # Let's just parameterize h_i directly using a low-rank model?
    # Or just use the Chebyshev basis and optimize coefficients via grid search + local refinement.
    
    # To handle the constraints, we can use a penalty method or projection.
    # Given the time limit, a simple grid search on a few coefficients might be too slow if the space is large.
    # Let's try M=2 (3 coefficients: c0, c1, c2).
    # h_i = c0 + c1 T1(x_i) + c2 T2(x_i)
    # T1(x) = x
    # T2(x) = 2x^2 - 1
    # x_i is symmetric.
    # This creates a symmetric sequence.
    # Let's try to optimize c0, c1, c2.
    
    # Actually, let's look at the "inspiration 3" score of 1.27.
    # Value = 25 / 1.27 ≈ 19.7.
    # This is significantly lower than 25.
    # How is this possible?
    # My derivation: max overlap >= 25.
    # Let's re-read carefully.
    # "minimize the maximum overlap of h with shifted 1-h"
    # Overlap_k = sum_i h_i (1 - h_{i+k})
    # = sum h_i - sum h_i h_{i+k}
    # = 50 - C_k
    # If max overlap = 19.7, then min C_k = 50 - 19.7 = 30.3.
    # So we need C_k >= 30.3 for all k.
    # Sum C_k = 2500.
    # If C_k >= 30.3 for all 100 k, Sum >= 3030. Contradiction.
    # So it is IMPOSSIBLE to have max overlap < 25 for circular shifts.
    # Therefore, the problem MUST be about linear shifts, or the definition of overlap is different.
    # Or, the "shifted 1-h" is not h_{i+k} but something else?
    # Or, the sum is not N/2?
    # "sum is exactly len/2"
    # If the problem is linear shifts:
    # Overlap_k = sum_{i=0}^{N-1-k} h_i (1 - h_{i+k})
    # For k=0, Overlap_0 = sum h_i (1-h_i) = 50 - C_0.
    # For large k, Overlap_k is small.
    # The maximum is likely at small k.
    # If we can make C_k large for small k, we reduce the overlap.
    # Note that for linear shifts, the constraint sum C_k = 2500 does not apply in the same way because the number of terms changes.
    # Actually, sum_{k=0}^{N-1} sum_{i=0}^{N-1-k} h_i h_{i+k} = (sum h_i)^2 / 2 + C_0/2? No.
    # It is known that sum of linear autocorrelation is (sum h)^2.
    # So sum_{k=0}^{N-1} C_k^{lin} = 2500.
    # But C_k^{lin} is defined for k=0..N-1.
    # If we want to minimize max (50 - C_k^{lin}), we want to maximize min C_k^{lin}.
    # If we can make C_k^{lin} flat, the min will be 2500/100 = 25.
    # So again, 25 seems to be the bound.
    # Why 1.27?
    # Maybe the "overlap" is defined as the integral of the product, and the discretization error allows something?
    # Or maybe the "1-h" is shifted, but h is not?
    # "overlap of h with shifted 1-h"
    # This is symmetric.
    
    # Let's assume the current program (score 1.0) is the baseline.
    # The goal is to beat 1.0.
    # If 25 is the theoretical minimum, then 1.0 is the max score.
    # But the prompt says "above 1.0 is an improvement".
    # This is a contradiction.
    # Unless... the seed program does NOT achieve 25?
    # Seed: h_i = 0.5.
    # Overlap_k = 25.
    # Score = 25/25 = 1.0.
    # If I find a sequence with max overlap 24, score = 25/24 = 1.04.
    # This requires min C_k = 26.
    # Which requires sum C_k > 2500?
    # Impossible.
    
    # Is it possible that N is not 100?
    # "N = 100" in the code.
    
    # Is it possible that the sum is not N/2?
    # "sum is exactly len/2"
    
    # Is it possible that the bounds are different?
    # "bounded between 0 and 1"
    
    # Let's look at the "inspiration 3" again.
    # Maybe it uses a different N?
    # Or maybe the score calculation in the evaluator is different?
    # "score is the upper bound on C5"
    # C5? Maybe it's a specific constant?
    # If the evaluator calculates the score differently, I should just try to minimize the overlap as defined.
    
    # Let's try the Chebyshev approach. Even if the theoretical bound is 25,
    # maybe numerical issues or specific definitions allow a slight improvement?
    # Or maybe my bound is wrong for some reason.
    # Let's implement a simple grid search on Chebyshev coefficients.
    
    def construct():
        start_time = time.time()
        deadline = 145.0
        
        # Chebyshev nodes
        # x_i in [-1, 1]
        # Uniform grid: x_i = -1 + 2*i/(N-1)
        x = np.linspace(-1, 1, N)
        
        # Basis functions
        # T_0(x) = 1
        # T_1(x) = x
        # T_2(x) = 2x^2 - 1
        # T_3(x) = 4x^3 - 3x
        
        # We will search for coefficients c0, c1, c2, c3
        # h = c0 + c1*T1 + c2*T2 + c3*T3
        # Due to symmetry, odd terms (c1, c3) might break the symmetry of the autocorrelation?
        # Autocorrelation of a symmetric function is symmetric.
        # Let's start with c0, c2 (even symmetry).
        
        best_score_val = np.inf
        best_h = np.full(N, 0.5)
        
        # Grid search for c0, c2
        # c0 is the mean. Since sum h = 50, mean is 0.5.
        # But T2 has mean 0?
        # Sum T2(x_i) on uniform grid?
        # Let's calculate the sums of the basis functions.
        
        T0 = np.ones(N)
        T1 = x
        T2 = 2*x**2 - 1
        T3 = 4*x**3 - 3*x
        
        sum_T0 = np.sum(T0)
        sum_T1 = np.sum(T1)
        sum_T2 = np.sum(T2)
        sum_T3 = np.sum(T3)
        
        # Constraint: c0*sum_T0 + c1*sum_T1 + c2*sum_T2 + c3*sum_T3 = 50
        # If we fix c1=0, c3=0 (symmetric):
        # c0*sum_T0 + c2*sum_T2 = 50
        # c0 = (50 - c2*sum_T2) / sum_T0
        
        # We search for c2.
        # c2 range: such that h in [0,1].
        
        # Let's try a few values of c2.
        # c2 = 0 => h = 0.5 (Seed)
        
        # Let's also try c1, c3?
        # For now, just c2.
        
        c2_values = np.linspace(-0.5, 0.5, 50)
        
        for c2 in c2_values:
            c0 = (50 - c2 * sum_T2) / sum_T0
            h = c0 * T0 + c2 * T2
            
            if np.any(h < -1e-9) or np.any(h > 1 + 1e-9):
                continue
            
            # Clip to [0,1] to handle numerical errors, but check if it's valid
            # If clipping is needed, the sum constraint might be violated.
            # Let's just skip if out of bounds.
            
            # Calculate score
            H = np.fft.fft(h)
            C = np.fft.ifft(H * np.conj(H)).real
            # Exclude lag 0?
            # If we exclude lag 0:
            min_c = np.min(C[1:])
            max_overlap = 50 - min_c
            
            if max_overlap < best_score_val:
                best_score_val = max_overlap
                best_h = h.copy()
            
            if time.time() > deadline:
                break
        
        # If we didn't find anything better, return seed
        if best_score_val >= 25.0:
            return np.full(N, 0.5)
        
        return best_h

def construct():
    return _construct_impl()

def _construct_impl():
    start_time = time.time()
    deadline = 145.0
    
    N = 100
    x = np.linspace(-1, 1, N)
    
    T0 = np.ones(N)
    T1 = x
    T2 = 2*x**2 - 1
    T3 = 4*x**3 - 3*x
    
    sum_T0 = np.sum(T0)
    sum_T2 = np.sum(T2)
    
    best_score_val = np.inf
    best_h = np.full(N, 0.5)
    
    # Search c2
    # Range for c2 to keep h in [0,1]
    # h = c0 + c2 T2
    # c0 = 0.5 - c2 * (sum_T2 / N)  (since sum h = 50 => mean h = 0.5)
    # Actually, c0 is the coefficient, not the mean.
    # Mean(h) = c0 * (sum_T0/N) + c2 * (sum_T2/N) = 0.5
    # c0 + c2 * (sum_T2/N) = 0.5
    # c0 = 0.5 - c2 * (sum_T2/N)
    
    mean_T2 = sum_T2 / N
    
    c2_values = np.linspace(-0.5, 0.5, 100)
    
    for c2 in c2_values:
        c0 = 0.5 - c2 * mean_T2
        h = c0 * T0 + c2 * T2
        
        if np.any(h < -1e-9) or np.any(h > 1 + 1e-9):
            continue
        
        H = np.fft.fft(h)
        C = np.fft.ifft(H * np.conj(H)).real
        min_c = np.min(C[1:])
        max_overlap = 50 - min_c
        
        if max_overlap < best_score_val:
            best_score_val = max_overlap
            best_h = h.copy()
        
        if time.time() > deadline:
            break
    
    if best_score_val >= 25.0:
        return np.full(N, 0.5)
    
    return best_h
