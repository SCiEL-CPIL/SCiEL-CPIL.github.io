import numpy as np
import time

N = 100

def _max_overlap_linear(h):
    """
    Compute the maximum linear overlap of h with shifted 1-h.
    Overlap(k) = sum_{i=0}^{N-1-k} h[i] * (1 - h[i+k])
    """
    max_ov = 0.0
    # We can vectorize or just loop. N=100 is small.
    # Using np.correlate might be easier but linear shifts with zero padding 
    # is effectively valid for 'valid' mode or we can just loop k.
    for k in range(N):
        if k == 0:
            ov = np.sum(h * (1 - h))
        else:
            ov = np.sum(h[:-k] * (1 - h[k:]))
        if ov > max_ov:
            max_ov = ov
    return max_ov

def _construct_ramp():
    """
    Construct a linear ramp h[i] = i / (N-1).
    Sum is exactly N/2.
    """
    h = np.arange(N, dtype=float) / (N - 1)
    return h

def _construct():
    """
    Entry point.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin

    # Initial guess: Ramp
    h = _construct_ramp()
    current_score = _max_overlap_linear(h)
    best_h = h.copy()
    best_score = current_score

    # Local Search
    # We can try swapping two elements, or shifting the ramp.
    # Or gradient descent.
    # Since N is small, let's try random swaps that maintain the sum.
    # But h is continuous.
    # Let's try adjusting h[i] and h[j] such that sum remains constant.
    # Or simply try a few different curves: Ramp, Sine, Cosine.
    
    candidates = []
    # Ramp
    candidates.append(_construct_ramp())
    
    # Sine ramp (smooth)
    h_sine = 0.5 * (1 - np.cos(np.linspace(0, np.pi, N)))
    # Normalize sum to N/2
    s = np.sum(h_sine)
    if s > 0:
        h_sine = h_sine * (N / 2) / s
        # Clip to [0, 1]
        h_sine = np.clip(h_sine, 0, 1)
        # Re-normalize after clip? Clip might change sum.
        # If sum is off, adjust linearly?
        # For simplicity, if sum is not exactly N/2, we might fail.
        # Let's stick to Ramp which has exact sum.
        
    # Try a "step" function: 0s then 1s.
    # We analyzed this: Max overlap is 1.
    # Ramp max overlap is ~ N/6.
    # N/6 = 16.6.
    # Step max overlap is 1.
    # Wait.
    # For Step: h=[1,1,...,1,0,0...].
    # k=0: 0.
    # k=1: Sum h[i](1-h[i+1]).
    # Only at the boundary 1->0 does h[i]=1, h[i+1]=0, term=1.
    # All other terms are 0.
    # So Sum = 1.
    # Max Overlap = 1.
    # 1 is MUCH better than 16.6.
    
    # So the Step function is better!
    # Let's verify the sum constraint.
    # h has N/2 ones and N/2 zeros. Sum = N/2.
    # Max overlap is 1.
    
    # Can we get 0?
    # If we interleave? h=[0,1,0,1...].
    # k=1: h[i](1-h[i+1]).
    # 0(0) + 1(1) + 0(0) + 1(1)...
    # Sum = N/2 * 1 = N/2 = 50.
    # Bad.
    
    # So Block Step is good.
    # What if we use a smoother transition?
    # If we have a gradient, we get many small terms.
    # Sum of small terms > 1 large term?
    # If we have a transition from 1 to 0 over M steps.
    # Terms are roughly 1/M * M = 1.
    # So any monotonic transition from 1 to 0 will have overlap ~1 at k=1.
    # Unless we can make the overlap 0.
    # Overlap is sum of positive terms.
    # To get 0, all terms must be 0.
    # h[i](1-h[i+k]) = 0 for all i,k.
    # This requires for every k, if h[i]>0 then h[i+k]=1.
    # This implies if h[i]=1, then h[i+k]=1.
    # So if any 1 exists, all subsequent elements must be 1.
    # So h must be [0, ..., 0, 1, ..., 1].
    # This is the Step function.
    # Max overlap is 1.
    
    # Is 1 the minimum?
    # What if N is even?
    # Step function max overlap is 1.
    # Seed is 25.
    # 1 is a huge improvement.
    
    # Let's return the Step function.
    # But wait, is "Step" stable?
    # If I return [1,1,...,1,0,0...], the score is 1.
    
    # Let's check if there's a trick to get < 1.
    # If h[i] is not 0 or 1?
    # If h[i] is small, h[i](1-h[i+k]) is small.
    # But sum must be N/2.
    # If we spread the mass, we increase the number of non-zero terms.
    # The "Step" concentrates mass to minimize the "interface" area.
    # This is likely optimal for linear shifts.
    
    # I will return the Step function.
    
    h_step = np.zeros(N)
    h_step[:N//2] = 1.0
    # Sum is N/2.
    
    # Check if any other arrangement is better.
    # If we put 1s at the end?
    # h=[0...0, 1...1].
    # k=1: 0...0, then 0->1 boundary?
    # h[i]=0, h[i+1]=1. Term 0(0)=0.
    # h[i]=1, h[i+1]=1. Term 1(0)=0.
    # So if 1s are at the END, the 1->0 boundary doesn't exist in the forward shift?
    # Shift is h[i] and h[i+k].
    # If h is 0...01...1.
    # k=1:
    # i where h[i]=0, h[i+1]=0 -> 0.
    # i where h[i]=0, h[i+1]=1 -> 0(0) = 0.
    # i where h[i]=1, h[i+1]=1 -> 1(0) = 0.
    # So Max Overlap is 0?
    # Let's check k>0.
    # If h is 0...01...1.
    # h[i]=1 implies h[j]=1 for all j>i.
    # So h[i+k] is also 1.
    # So h[i](1-h[i+k]) = 1(0) = 0.
    # If h[i]=0, term is 0.
    # So ALL overlaps are 0!
    
    # Wait.
    # If h = [0, 0, ..., 0, 1, 1, ..., 1].
    # Sum = N/2.
    # For any k:
    # If h[i]=1, then h[i+k]=1 (since 1s are at the end, indices increase).
    # So 1-h[i+k]=0.
    # If h[i]=0, term is 0.
    # So Overlap(k) = 0 for all k.
    
    # This is a perfect solution!
    
    return h_step[::-1] # Reverse it to put 1s at the end.

# Let's verify this logic.
# h = [0, 1] (N=2).
# k=0: 0(1) + 1(0) = 0.
# k=1: 0(0) = 0.
# Max 0.
# Yes.

# So the optimal solution for linear shifts is a monotonic step from 0 to 1.

def construct():
    return np.concatenate([np.zeros(N//2), np.ones(N - N//2)])
