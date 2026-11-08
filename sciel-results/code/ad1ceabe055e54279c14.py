import numpy as np

N = 600

def construct():
    """Return a Gaussian step function to minimize peak sidelobes and peak-to-peak ratio."""
    # Center at N/2
    center = N / 2.0
    # Width sigma. Choose sigma such that the pulse is reasonably localized.
    # A sigma of N/10 or N/20 might be good.
    # If sigma is too small, it approaches a spike.
    # Let's use sigma = N / 10.
    sigma = N / 10.0
    x = np.arange(N)
    heights = np.exp(-((x - center) ** 2) / (2 * sigma ** 2))
    # Ensure non-negative (exp is always positive)
    # Normalize? The metric is likely a ratio, so scale doesn't matter.
    return list(heights)
