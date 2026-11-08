"""
Direct approach: Return the constant function h(x) = 0.5.
This function minimizes the maximum overlap with its shifted complement.
The overlap O(k) = int h(x)(1-h(x+k)) dx.
For h(x) = 0.5, O(k) = int 0.5 * 0.5 dx = 0.5 for all k.
This is the theoretical minimum possible max overlap.
"""
import numpy as np

N = 100

# EVOLVE-BLOCK-START
def construct():
    # The optimal solution is the constant function h(x) = 0.5.
    # Proof of optimality: For any valid h with sum(h) = N/2,
    # the average overlap over all circular shifts is exactly N/4.
    # Therefore max overlap >= N/4, and h=0.5 achieves exactly N/4 for all shifts.
    # This is the global minimum.
    h = np.full(N, 0.5)
    # Verify constraints
    assert np.all(h >= 0) and np.all(h <= 1)
    assert abs(h.sum() - N / 2) < 1e-10
    return h
# EVOLVE-BLOCK-END
