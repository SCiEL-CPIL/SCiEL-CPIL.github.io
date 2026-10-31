"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 100


def _score(h):
    """Compute max overlap of h with shifted (1-h), treating h as zero outside [0,2]."""
    n = len(h)
    shifted = np.zeros(n)
    shifted[:-1] = h[1:]
    overlap = h * (1.0 - shifted)
    return float(np.max(overlap))


def _overlap_for_2block(k, a, b):
    """Compute max overlap for 2-block with first k elements = a, rest = b."""
    n = N
    if k <= 0 or k >= n:
        return float('inf')
    # Overlaps:
    # x in [0, k-2]: a*(1-a)
    # x = k-1: a*(1-b)
    # x in [k, n-2]: b*(1-b)
    # x = n-1: b*1
    vals = []
    if k >= 2:
        vals.append(a * (1.0 - a))
    vals.append(a * (1.0 - b))
    if n - k >= 2:
        vals.append(b * (1.0 - b))
    vals.append(b)
    return max(vals)


def construct():
    deadline = time.time() + 100.0  # safety margin before 120s kill

    best_h = None
    best_score = float('inf')

    # Try various splits k (number of elements in first block)
    for k in range(1, N):
        # Constraint: k*a + (N-k)*b = N/2
        # b = (N/2 - k*a) / (N-k)
        # Minimize max overlap over a in [0,1] with b in [0,1]
        
        def obj(a_val):
            b_val = (N / 2.0 - k * a_val) / (N - k)
            if b_val < -1e-12 or b_val > 1.0 + 1e-12:
                return 100.0
            b_val = np.clip(b_val, 0.0, 1.0)
            return _overlap_for_2block(k, a_val, b_val)

        # Bounds on a: b = (N/2 - k*a)/(N-k) must be in [0,1]
        # b >= 0: N/2 - k*a >= 0 → a <= N/(2k)
        # b <= 1: N/2 - k*a <= N-k → -k*a <= N/2 - N + k → a >= (N/2 - N + k)/k = 1/2 - 1/k + 1 = 3/2 - 1/k
        a_min = max(0.0, 1.5 - 1.0/k)
        a_max = min(1.0, N / (2.0 * k))
        
        if a_min >= a_max:
            continue
        
        # Sample a few starting points
        best_local = None
        for a0 in [a_min + (a_max - a_min) * t for t in [0.1, 0.3, 0.5, 0.7, 0.9]]:
            a0 = np.clip(a0, a_min + 1e-9, a_max - 1e-9)
            res = minimize(obj, [a0], method='L-BFGS-B', bounds=[(a_min + 1e-9, a_max - 1e-9)])
            if res.success and res.fun < best_score:
                a_opt = res.x[0]
                b_opt = (N / 2.0 - k * a_opt) / (N - k)
                b_opt = np.clip(b_opt, 0.0, 1.0)
                score_val = _overlap_for_2block(k, a_opt, b_opt)
                if score_val < best_score:
                    best_score = score_val
                    h = np.zeros(N)
                    h[:k] = a_opt
                    h[k:] = b_opt
                    # Fix sum
                    s = h.sum()
                    diff = N / 2.0 - s
                    if abs(diff) > 1e-15:
                        h[k - 1] += diff
                    h = np.clip(h, 0.0, 1.0)
                    best_h = h.copy()
        
        if time.time() > deadline:
            break

    if best_h is None:
        # Fallback to original
        A = (3.0 - math.sqrt(5.0)) / 2.0
        B = 1.0 - A
        h = np.zeros(N)
        h[:N//2] = A
        h[N//2:] = B
        best_h = h

    return best_h



