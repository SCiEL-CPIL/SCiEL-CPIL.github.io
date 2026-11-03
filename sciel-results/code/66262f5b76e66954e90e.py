import numpy as np
import time

# Constants
N = 400
# The evaluator kills after 150s. We set a conservative internal deadline.
# The computation is O(1) effectively, so this is just a safety net.
_DEADLINE = 145.0
_T0 = time.monotonic()

def _time_left():
    return _DEADLINE - (time.monotonic() - _T0)

def construct():
    """
    Returns a signed step height vector of length N minimizing
    max|linear_autocorrelation| / (sum f)^2.

    The optimal solution is the constant vector ones(N), which achieves
    the theoretical lower bound of 1/N.
    
    Proof:
    Let S = sum(f). The metric is max_k |C_k| / S^2.
    C_0 = sum(f^2).
    By Cauchy-Schwarz, S^2 <= N * sum(f^2) => sum(f^2) >= S^2 / N.
    Thus max_k |C_k| >= C_0 >= S^2 / N.
    Ratio >= (S^2 / N) / S^2 = 1/N.
    Equality holds for constant vectors.
    """
    # Check time immediately
    if _time_left() < 0.1:
        return np.ones(N, dtype=np.float64)

    # The optimal vector is the constant vector of 1s.
    # Any scaling c * ones(N) yields the same ratio 1/N.
    # We use 1.0 for numerical stability and simplicity.
    f = np.ones(N, dtype=np.float64)
    
    return f
