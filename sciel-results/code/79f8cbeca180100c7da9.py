import numpy as np

def _score_raw(h):
    """Compute the raw score for internal verification."""
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 <= 0:
        return 0.0
    l2_sq = np.sum(h**2)
    # Assuming continuous autoconvolution peak is approximated by Delta * max_discrete_conv
    # Delta = 0.01
    h_conv = np.convolve(h, h, mode='full')
    max_conv = np.max(h_conv)
    if max_conv <= 0:
        return 0.0
    return l2_sq / (l1 * (0.01 * max_conv))

def construct():
    """
    Return a sequence of 50 nonnegative step heights.
    The optimal strategy is to maximize sparsity.
    A single spike maximizes the ratio L2^2 / (L1 * Sup(AutoConv)).
    """
    h = np.zeros(50, dtype=np.float64)
    h[0] = 1.0
    return list(h)
