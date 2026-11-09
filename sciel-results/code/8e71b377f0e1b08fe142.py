import numpy as np
import time

N = 50

def construct():
    """
    Returns 50 nonnegative step heights on equal intervals of [-1/4, 1/4]
    to maximize the ratio ||h||_2^2 / (||h||_1 * ||h*h||_inf).
    
    Strategy:
    The ratio is maximized by concentrating the mass in as many disjoint
    "spikes" as possible. Disjoint spikes do not contribute to each other's
    convolution peak, keeping the denominator term ||h*h||_inf low relative
    to the numerator ||h||_2^2 and the term ||h||_1.
    
    For a pattern of M disjoint single-bin spikes of height H:
    ||h||_1 = M * H
    ||h||_2^2 = M * H^2
    ||h*h||_inf = H^2 (peak of a single spike's self-convolution)
    
    Ratio R = (M * H^2) / (M * H * H^2) = 1 / H.
    
    Since the problem implies a normalized context (seed score 1.0 for unit spike),
    we assume the evaluator normalizes the input.
    If normalized to ||h||_1 = 1, then M * H = 1 => H = 1/M.
    Then R = 1 / (1/M) = M.
    
    To maximize R, we maximize M, the number of disjoint spikes.
    The maximum number of disjoint single-bin spikes in a sequence of length N
    is floor(N/2) if we require separation by at least one zero, or N if we
    allow adjacency?
    
    Wait, if spikes are adjacent (e.g., 1, 1, 0, 0), they are not disjoint in the
    convolution sense. The convolution of adjacent bins will have a higher peak.
    Specifically, for a block of K adjacent bins, the peak is K * H^2.
    If we have M blocks of size 1, separated by zeros, the peak is H^2.
    
    Max M for size-1 blocks in N=50:
    We can place spikes at indices 0, 2, 4, ..., 48.
    Total M = 25.
    
    If we used blocks of size 1 at indices 0, 1, 2, ..., 49 (M=50, K=1 each? No, they are adjacent).
    If they are adjacent, they form one big block of K=50.
    Peak = 50 * H^2.
    ||h||_1 = 50 * H.
    ||h||_2^2 = 50 * H^2.
    R = (50 H^2) / (50 H * 50 H^2) = 1 / (50 H).
    With ||h||_1 = 1, H = 1/50.
    R = 1 / (50 * 1/50) = 1.
    
    So adjacency is bad. Separation is good.
    Max M with separation is 25.
    R = 25.
    
    We will return the vector corresponding to M=25 disjoint spikes, normalized to sum 1.
    """
    
    # Start time guard
    start_time = time.time()
    deadline = start_time + 140
    
    # Construct the optimal pattern: alternating 1s and 0s
    h = np.zeros(N)
    h[0::2] = 1.0  # Indices 0, 2, 4, ..., 48
    
    # Normalize to sum 1 (assuming L1 normalization is the standard implicit constraint
    # or that the evaluator will normalize, returning a standard scale is safe)
    s = np.sum(h)
    if s > 0:
        h = h / s
    
    # The analytical solution is immediate. 
    # We can use the remaining time to verify or slightly perturb if needed, 
    # but the structure is likely global optimal for this metric.
    
    return list(h)
