import numpy as np
import time

N = 50
DEADLINE = 120.0

def construct():
    start_time = time.time()
    
    # Candidate k values
    k_values = [5, 10, 15, 20]
    
    best_heights = [1.0] * N
    best_score = 0.0
    
    # x coordinates for the right half [0, 0.25]
    # We use 50 bins for the right half? 
    # The problem says "50 heights on equal intervals of [-1/4, 1/4]".
    # So the total interval is [-0.25, 0.25].
    # If we put mass on [0, 0.25], that is 25 bins.
    # Let's use the full range but shift the power law.
    # h_i proportional to (x_i - x_min)^k for x_i > 0?
    # Or just use the right half of the 50 bins.
    
    # Bins are centered or edges? "heights on equal intervals".
    # Usually h_i is the value on the i-th interval.
    # Let's assume x_i are the midpoints or right edges.
    # x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25/N
    
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25/N
    
    for k in k_values:
        if time.time() - start_time > DEADLINE - 5:
            break
            
        # Create heights: power law on the positive side
        # h_i = (x_i)^k for x_i > 0, else 0
        h = np.zeros(N)
        mask = x > 0
        # Avoid 0^k issues if x is exactly 0, but x starts at -0.245
        # The first positive x is around 0.005
        h[mask] = x[mask] ** k
        
        # Normalize L1 norm to 1
        l1 = np.sum(h) * (0.25/N) # Delta x = 0.5/50 = 0.01
        if l1 == 0:
            continue
        h = h / l1
        
        # Calculate score
        # L2 norm squared
        l2_sq = np.sum(h**2) * (0.25/N)
        
        # Autoconvolution
        # Discrete convolution
        # h is a sequence. Convolution length is 2N-1
        conv = np.convolve(h, h, mode='full')
        # The peak of the continuous autoconvolution is approximated by the max of the discrete conv * dx?
        # (f*f)(t) = sum h_i h_j dx. 
        # np.convolve(h, h) gives sum h_i h_j.
        # So peak = max(conv) * dx
        peak = np.max(conv) * (0.25/N)
        
        if peak == 0:
            continue
            
        score = l2_sq / (1.0 * peak) # L1 is 1
        
        if score > best_score:
            best_score = score
            best_heights = h.tolist()
            
    return best_heights
