import numpy as np

N = 100

def construct():
    # Analytical construction using a cosine wave
    # h_i = 0.5 + 0.5 * cos(2 * pi * i / N)
    # This ensures h_i in [0, 1] and sum(h) = N/2
    # It maximizes the minimum autocorrelation compared to uniform or random sequences.
    
    i = np.arange(N)
    h = 0.5 + 0.5 * np.cos(2.0 * np.pi * i / N)
    
    # Correct for any floating point errors in the sum
    target_sum = N / 2.0
    current_sum = np.sum(h)
    diff = target_sum - current_sum
    
    if abs(diff) > 1e-12:
        # Distribute the difference evenly or to the last element
        # Since the analytical sum is exactly N/2, this should be tiny
        h[-1] += diff
        
    # Ensure bounds [0, 1]
    h = np.clip(h, 0.0, 1.0)
    
    # Final check on sum after clipping
    # If clipping occurred, we might need to re-adjust
    current_sum = np.sum(h)
    diff = target_sum - current_sum
    if abs(diff) > 1e-12:
        # Simple correction: add to all elements, then clip again
        # This might break bounds, but for this specific wave, clipping 
        # at 0 or 1 only happens at the peaks/troughs which are exactly 1 and 0.
        # So clipping shouldn't change the values if they are exactly 0 or 1.
        # However, due to floating point, they might be 1e-16 or 1-1e-16.
        # If we add diff, they might go out of bounds.
        # Better to adjust non-bound elements.
        mask = (h > 1e-9) & (h < 1 - 1e-9)
        if np.any(mask):
            h[mask] += diff / np.sum(mask)
            h = np.clip(h, 0.0, 1.0)
            
    return h
