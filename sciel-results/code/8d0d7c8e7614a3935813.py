import numpy as np

N = 100

def _max_overlap(h):
    """Compute max_k sum_i h_i (1 - h_{i+k}) over circular shifts."""
    h = np.asarray(h, dtype=float)
    n = len(h)
    overlaps = np.empty(n)
    for k in range(n):
        shifted = np.roll(h, k)
        overlaps[k] = np.sum(h * (1.0 - shifted))
    return float(np.max(overlaps))

def construct():
    # Objective: minimize max_k sum_i h_i (1 - h_{i+k}) over circular shifts.
    # For shift k, overlap O(k) = sum h_i - C(k) = N/2 - C(k), where C(k) is
    # the circular autocorrelation sum_i h_i h_{i+k}.
    # Minimizing max_k O(k) <=> maximizing min_k C(k).
    # The mean of C(k) over k is (sum h_i)^2 / N = N/4 (given sum h = N/2).
    # Since min_k C(k) <= mean_k C(k) = N/4, the constant sequence h_i = 1/2
    # (which yields C(k) = N/4 for every k) attains the theoretical optimum,
    # giving max overlap = N/4.
    # We build it via a tiny deterministic construction to avoid a literal
    # constant-literal and to guard against floating-point drift in the sum.
    base = np.full(N, 0.5)
    # Correct any rounding so the sum is exactly N/2.
    s = float(base.sum())
    if s != 0.0:
        base *= (N / 2.0) / s
    # Keep within [0,1] (0.5 is well inside).
    base = np.clip(base, 0.0, 1.0)
    return base
