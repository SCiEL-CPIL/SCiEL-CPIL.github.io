import numpy as np

N = 600

def construct():
    # Interval width
    w = 1.0 / (2.0 * N)
    
    # Centers of the intervals
    # Intervals are on [-1/4, 1/4]
    # i-th interval is [-1/4 + i*w, -1/4 + (i+1)*w]
    # Center x_i = -1/4 + (i + 0.5) * w
    i = np.arange(N)
    x = -0.25 + (i + 0.5) * w
    
    # Define the function shape
    # h(x) = (1 - 4|x|)^k
    # k=50 gives a score of ~0.505
    k = 50
    
    # Calculate heights
    # Ensure non-negative
    base = 1.0 - 4.0 * np.abs(x)
    base = np.maximum(base, 0.0)
    
    heights = base ** k
    
    # The function is symmetric and non-negative
    # Return as a list of floats
    return [float(h) for h in heights]
