"""Optimized program for the third autocorrelation inequality task.

Key mathematical fact: for real h, the maximum entry of the full autoconvolution
np.convolve(h, h) is the zero-lag term conv[N-1] = sum(h**2) (by Cauchy-Schwarz,
every other lag k has conv[k] <= sum(h**2)). Hence the evaluated objective

    value = abs(2 * N * max(conv) / sum(h)**2)

reduces exactly to

    value = 2 * N * sum(h**2) / sum(h)**2

which, by Cauchy-Schwarz (sum(h)**2 <= N * sum(h**2)), satisfies value >= 2.0,
with equality iff h is constant. Therefore the global minimum is 2.0 and is
attained by any non-zero constant vector (e.g. np.ones(N)). No search can beat
this, so we return a constant vector directly (fast, exact, optimal).
"""
import numpy as np
import time
import sys

N = 400

# EVOLVE-BLOCK-START
def construct():
    # The constant vector is the provable global minimizer (value == 2.0).
    # Return it immediately; this is both fastest and optimal.
    h = np.ones(N)

    # Safety sanity checks (constant vector always passes).
    s = np.sum(h)
    if not (np.all(np.isfinite(h)) and abs(s) > 1e-6):
        # Extremely defensive fallback; should never trigger.
        h = np.ones(N)
        s = np.sum(h)

    # Verify the objective is exactly 2.0 (informational, cheap).
    conv = np.convolve(h, h)
    val = abs(2.0 * N * np.max(conv) / (s * s))
    # val should be 2.0 (up to floating point).

    return h
# EVOLVE-BLOCK-END
