import numpy as np

N = 100


def construct():
    """
    Construct the sequence of step heights.
    
    Returns: sequence of N step heights in [0, 1] whose sum is exactly N/2.
    
    This implementation uses a near-binary sequence with smoothed domain walls.
    It models the system as an Ising spin chain, creating blocks of 0s and 1s.
    The transitions between 0 and 1 are smoothed using a sigmoid function to
    maintain differentiability and satisfy the [0, 1] constraint strictly.
    The number of blocks is chosen to balance the autocorrelation properties.
    """
    # Number of domain walls (transitions). 
    # A single block (2 walls) or a few blocks might be optimal.
    # Let's try 2 blocks of 1s and 2 blocks of 0s (4 walls total, period 2N? No, period N).
    # Let's use 2 blocks of 1s, each of length N/4, and 2 blocks of 0s.
    # Pattern: 0 0 1 1 0 0 1 1 ...
    # This is equivalent to a cosine wave of lower frequency?
    # cos(4*pi*i/N) has period N/2.
    
    # Let's stick to the directive: optimize boundary positions.
    # We will define a binary template and smooth it.
    
    # Template: Alternating blocks of 0 and 1.
    # Let's try a sequence with 2 blocks of 1s and 2 blocks of 0s.
    # Indices 0..N/4-1: 0
    # Indices N/4..N/2-1: 1
    # Indices N/2..3N/4-1: 0
    # Indices 3N/4..N-1: 1
    
    # This is h_i = 0.5 + 0.5 * sign(sin(4*pi*i/N))?
    # sin(4*pi*i/N) has period N/2.
    # sign(sin) is square wave.
    
    # Let's create a smooth square wave.
    i = np.arange(N)
    # Frequency: 2 cycles per N?
    # Phase such that it is 0 in first quarter, 1 in second, etc.
    # sin(4*pi*i/N + pi/2) = cos(4*pi*i/N).
    # cos is 1 at 0, 0 at N/8, -1 at N/4.
    # We want 0, 1, 0, 1.
    # So we want a function that is negative in [0, N/4), positive in [N/4, N/2), etc.
    # sin(4*pi*i/N) is:
    # 0 at 0.
    # + at N/16.
    # 0 at N/8.
    # - at 3N/16.
    # 0 at N/4.
    # This doesn't match 0,1,0,1 blocks of N/4.
    
    # Let's use a simple piecewise linear ramp smoothed by sigmoid.
    # Define centers of 1-blocks and 0-blocks.
    # Block 1 (1s): center N/4, width N/2? No, width N/4.
    # Let's just use the cosine approach but with a higher frequency or different shape?
    # The directive says "near-binary".
    
    # I will implement a soft step function.
    # h_i = sigmoid(alpha * (x_i - threshold))
    
    # Let's try a sequence that is 0 for i < N/4, 1 for N/4 <= i < N/2, etc.
    # We can define a "target" binary sequence and smooth it.
    target = np.zeros(N)
    # 2 blocks of 1s, each N/4 long.
    target[N//4 : N//2] = 1.0
    target[3*N//4 : N] = 1.0
    
    # Smooth the transitions.
    # We can convolve with a small Gaussian or use a sigmoid on the distance to the boundary.
    # Simpler: Use a cosine window that approximates the step.
    # Or just use the cosine wave but with frequency 4*pi/N?
    # h = 0.5 + 0.5 * cos(4*pi*i/N)
    # This has period N/2.
    # Max overlap for this?
    # S_k = N/4 + N/8 cos(4*pi*k/N).
    # Min S_k at 4*pi*k/N = pi => k=N/4.
    # S_min = N/4 - N/8 = N/8.
    # Max Overlap = N/2 - N/8 = 3N/8.
    # Same as the original cosine (which had period N, min at N/2).
    # So changing frequency doesn't help the sum-overlap metric.
    
    # If the directive insists on near-binary, I will provide a smoothed binary sequence.
    # I will use a sigmoid to create soft edges.
    
    # Define boundaries at N/4, N/2, 3N/4.
    # h_i = 0.5 * (1 + tanh(k * (i - b1))) - 0.5 * (1 + tanh(k * (i - b2))) ...
    
    # Let's try a simpler approach: 
    # Use a sequence that is 0.5 everywhere, but add small binary-like perturbations?
    # No, "near-binary" means values close to 0 or 1.
    
    # I will implement a smoothed square wave with 2 periods.
    # h_i = 0.5 + 0.5 * sign(sin(4*pi*i/N))
    # To make it differentiable/soft, replace sign(x) with tanh(alpha*x).
    alpha = 10.0
    signal = np.sin(4.0 * np.pi * i / N)
    h = 0.5 + 0.5 * np.tanh(alpha * signal)
    
    # Ensure sum is N/2
    h_sum = np.sum(h)
    if h_sum != N / 2.0:
        h += (N / 2.0 - h_sum) / N
    h = np.clip(h, 0.0, 1.0)
    h_sum = np.sum(h)
    if h_sum != N / 2.0:
        h += (N / 2.0 - h_sum) / N
        
    return h
