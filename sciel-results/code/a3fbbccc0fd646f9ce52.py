"""Search-based program for Erdos minimum overlap task."""
import numpy as np
import time

N = 100

def _max_overlap(h):
    """Compute max over shifts of mean(h[i]*(1-h[(i+s)%N]))."""
    N = len(h)
    inv_h = 1.0 - h
    best = 0.0
    for s in range(N):
        # overlap for shift s: mean of h[i] * inv_h[(i+s)%N]
        ov = np.dot(h, inv_h[np.roll(np.arange(N), -s)])
        if ov > best:
            best = ov
    return best

def _score(h):
    s = h.sum()
    if abs(s - N/2.0) > 1e-8:
        return float('inf')
    if h.min() < -1e-8 or h.max() > 1.0 + 1e-8:
        return float('inf')
    return _max_overlap(h)

# EVOLVE-BLOCK-START
def construct():
    # The theoretical lower bound for the maximum overlap is N/4 (for sum) or 0.5 (for integral).
    # The constant sequence h_i = 0.5 achieves this bound.
    # Any deviation from constant will create a shift with overlap > N/4.
    # Therefore, the optimal solution is the constant sequence.
    
    h = np.full(N, 0.5)
    
    # Verify constraints
    assert abs(h.sum() - N/2.0) < 1e-8
    assert h.min() >= 0.0 and h.max() <= 1.0
    
    return h
# EVOLVE-BLOCK-END
