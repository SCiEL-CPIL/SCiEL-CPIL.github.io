import time

N = 600

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin
    
    # Strategy: Minimize max(autoconvolution) / ||f||_2^2
    # By using a sparse, alternating pattern, we create disjoint bumps.
    # The peak of the autoconvolution is determined by the largest bump.
    # The L2 norm squared is the sum of the squares of all bumps.
    # Thus, spreading the mass over many disjoint bumps minimizes the ratio.
    
    # Pattern: 1, 0, 1, 0, ...
    # This creates N/2 bumps of width 1 interval.
    # They are separated by 1 interval, so their autoconvolutions do not overlap at the peak.
    
    f = [0.0] * N
    for i in range(0, N, 2):
        f[i] = 1.0
        
    # Optional: Check time, though this is O(N)
    if time.time() > deadline:
        pass # Return current best
        
    return f
