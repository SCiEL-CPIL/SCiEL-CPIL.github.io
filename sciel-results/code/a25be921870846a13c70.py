"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

MAXN = 10_000_000
TARGET = Fraction(3, 2)

def _compute_sigma(n):
    """Compute sigma(k) for all k in [1..n] via divisor sieve."""
    sigma = [0] * (n + 1)
    for d in range(1, n + 1):
        for m in range(d, n + 1, d):
            sigma[m] += d
    return sigma

def construct():
    start = time.time()
    deadline = start + 85.0  # safety margin under 90s

    # Precompute sigma for all n <= MAXN
    sigma = _compute_sigma(MAXN)

    best_err = None
    best_pair = (11, 6)  # fallback: sigma(11)=12, sigma(6)=12

    # Search: for each b, look for a near 1.5*b with sigma(a)==sigma(b)
    # To keep it fast, scan b and check a window around 1.5*b
    # We'll do a coarse pass first, then refine.

    # Coarse pass: step through b, check a in a small window
    # Use a window that grows if we find good candidates
    window = 50
    found_any = False

    # First, a broad scan with small window
    for b in range(1, MAXN + 1):
        if time.time() > deadline:
            break
        sb = sigma[b]
        # a should be near 1.5*b
        center = int(round(1.5 * b))
        lo = max(1, center - window)
        hi = min(MAXN, center + window)
        for a in range(lo, hi + 1):
            if a == b:
                continue
            if sigma[a] == sb:
                # compute error exactly
                err = abs(Fraction(a, b) - TARGET)
                if err > 0:
                    if best_err is None or err < best_err:
                        best_err = err
                        best_pair = (a, b)
                        found_any = True

    if not found_any:
        # fallback
        return [best_pair[0], best_pair[1]]

    # Refine: around the best pair, do a tighter search
    a0, b0 = best_pair
    # Expand window around the best ratio region
    # Do a second pass focusing on b values near b0
    b_lo = max(1, b0 - 5000)
    b_hi = min(MAXN, b0 + 5000)
    window2 = 200
    for b in range(b_lo, b_hi + 1):
        if time.time() > deadline:
            break
        sb = sigma[b]
        center = int(round(1.5 * b))
        lo = max(1, center - window2)
        hi = min(MAXN, center + window2)
        for a in range(lo, hi + 1):
            if a == b:
                continue
            if sigma[a] == sb:
                err = abs(Fraction(a, b) - TARGET)
                if err > 0:
                    if best_err is None or err < best_err:
                        best_err = err
                        best_pair = (a, b)

    # Final: ensure types are Python int
    a, b = best_pair
    return [int(a), int(b)]

# EVOLVE-BLOCK-END
