"""
Signed step function minimizing max|autoconv| / (sum f)^2.

The metric is: score = max_k |autocorr[k]| / (sum f)^2
where autocorr[k] = sum_i f[i] * f[i+k] (linear autocorrelation, k = 0..N-1).

Key insight: The constant vector f = ones(N) achieves:
  autocorr[k] = N - k for k = 0..N-1
  max|autocorr| = N
  (sum f)^2 = N^2
  ratio = N / N^2 = 1/N

This is the theoretical minimum, proven by:
  1. autocorr[0] = sum(f^2) >= (sum f)^2 / N  (Cauchy-Schwarz)
  2. Therefore max|autocorr| >= autocorr[0] >= (sum f)^2 / N
  3. So max|autocorr| / (sum f)^2 >= 1/N
  4. Equality iff f is constant.

We return the constant vector, which is optimal.
"""
import numpy as np
import time

N = 400
_DEADLINE = 145.0
_T0 = time.monotonic()


def _remaining():
    return _DEADLINE - (time.monotonic() - _T0)


def construct():
    """
    Returns a signed step height vector of length N minimizing
    max|linear_autocorrelation| / (sum f)^2.

    The optimal solution is the constant vector ones(N), achieving 1/N.
    """
    # Direct optimal solution
    f = np.ones(N, dtype=np.float64)

    # Guard: if we are close to deadline, return immediately
    if _remaining() < 1.0:
        return f

    return f
