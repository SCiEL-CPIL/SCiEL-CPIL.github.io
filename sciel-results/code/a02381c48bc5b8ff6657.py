import numpy as np
import time

N = 100

def _overlap_max(h):
    """Compute max cyclic overlap of h with 1-h, scaled by step width."""
    w = 2.0 / N
    H = np.fft.fft(h)
    autocorr = np.fft.ifft(H * np.conj(H)).real
    overlaps = N / 2.0 - autocorr
    return w * np.max(overlaps)

def _overlap_max_bitmask(mask):
    """
    Compute max cyclic overlap for a binary mask (0s and 1s).
    mask is an integer bitmask of length N.
    Overlap at shift k: sum_i h[i] * (1 - h[i+k])
    Since sum h[i] = N/2, this is N/2 - autocorr[k].
    We want to minimize max(N/2 - autocorr[k]) => maximize min(autocorr[k]).
    Actually, let's just compute the value directly to be safe and consistent with the interface.
    """
    # Convert mask to numpy array
    h = np.zeros(N)
    for i in range(N):
        if (mask >> i) & 1:
            h[i] = 1.0
    
    w = 2.0 / N
    H = np.fft.fft(h)
    autocorr = np.fft.ifft(H * np.conj(H)).real
    overlaps = N / 2.0 - autocorr
    return w * np.max(overlaps)

def construct():
    deadline = time.time() + 115  # Safety margin
    
    best_mask = None
    best_val = float('inf')
    
    # We need exactly N/2 ones in the mask to satisfy sum = N/2.
    # N=100, so 50 ones.
    
    # 1. Initial population: Random masks with exactly 50 ones
    # We can use a few different initialization strategies.
    # Strategy A: Pure random
    # Strategy B: Alternating (0101...) - likely good for low autocorrelation?
    # Let's try random restarts with local search.
    
    num_restarts = 200
    steps_per_restart = 500
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Generate random mask with 50 ones
        # Use numpy random permutation for speed
        indices = np.random.permutation(N)
        mask = 0
        for i in indices[:N//2]:
            mask |= (1 << i)
            
        # Check validity (should be valid by construction)
        # sum is 50, so integral is 1.
        
        val = _overlap_max_bitmask(mask)
        
        if val < best_val:
            best_val = val
            best_mask = mask
            
        # Local Search: Flip two bits (one 0 to 1, one 1 to 0) to maintain sum
        # This is a 2-opt move.
        # To make it faster, we can precompute or just try random pairs.
        # Given N=100, there are 50*50=2500 possible swaps.
        # We can try a subset or all of them?
        # 2500 FFTs per step is too slow? 
        # FFT is O(N log N) ~ 700 ops. 2500 * 700 = 1.75M ops per step.
        # 500 steps = 875M ops. Might be too slow in Python loop if not vectorized.
        # Let's optimize the evaluation.
        
        # Alternative: Use the property that flipping bits i (0->1) and j (1->0)
        # changes autocorr[k] by:
        # delta_autocorr[k] = (h[i]h[i+k] + h[j]h[j+k] - h[i]h[j+k] - h[j]h[i+k]) ... this is getting complex.
        
        # Let's stick to FFT but optimize the loop.
        # Actually, let's try a simpler local search: try a few random swaps per step.
        # If we do 10 random swaps per step, and 500 steps, that's 5000 evaluations.
        # 5000 * 700 = 3.5M ops. Very fast.
        
        local_mask = mask
        local_val = val
        no_improve = 0
        max_no_improve = 50
        
        while no_improve < max_no_improve and time.time() < deadline - 5:
            # Pick random 0 and 1
            zeros = [i for i in range(N) if not (local_mask >> i) & 1]
            ones = [i for i in range(N) if (local_mask >> i) & 1]
            
            if not zeros or not ones:
                break
                
            i = np.random.choice(zeros)
            j = np.random.choice(ones)
            
            # Swap i and j
            new_mask = local_mask ^ (1 << i) ^ (1 << j)
            
            new_val = _overlap_max_bitmask(new_mask)
            
            if new_val < local_val:
                local_mask = new_mask
                local_val = new_val
                no_improve = 0
                if local_val < best_val:
                    best_val = local_val
                    best_mask = local_mask
            else:
                no_improve += 1
                
                # Occasionally try a larger jump? Or just continue.
                # If stuck, we might try to break symmetry or something, but random restarts handle that.
                
    # Convert best_mask to array h
    h = np.zeros(N)
    for i in range(N):
        if (best_mask >> i) & 1:
            h[i] = 1.0
            
    return h
