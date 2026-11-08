import numpy as np
import time

N = 50

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a sequence of 50 nonnegative heights.
    Strategy: Local optimization over weight ratios for 2-point and 3-point clusters.
    """
    t0 = time.time()
    deadline = 15.0  # Safety margin well under 120s
    
    def compute_score(h):
        """Compute the ratio: L2^2 / (L1 * sup_norm_autoconv)."""
        h = np.array(h, dtype=float)
        l1 = np.sum(h)
        if l1 < 1e-15:
            return 0.0
        l2_sq = np.sum(h**2)
        # Autoconvolution (linear)
        conv = np.convolve(h, h, mode='full')
        conv_inf = np.max(conv)
        if conv_inf < 1e-15:
            return 0.0
        return l2_sq / (l1 * conv_inf)
    
    best_score = 0.0
    best_h = None
    
    # Try 2-point clusters with various weight ratios
    for r in np.linspace(0.1, 5.0, 200):
        if time.time() - t0 > deadline:
            break
        a = r
        b = 1.0
        total = a + b
        h = [0.0] * N
        h[0] = a / total
        h[1] = b / total
        s = compute_score(h)
        if s > best_score:
            best_score = s
            best_h = h[:]
    
    # Try 3-point clusters with various weight ratios
    for r1 in np.linspace(0.1, 3.0, 50):
        if time.time() - t0 > deadline:
            break
        for r2 in np.linspace(0.1, 3.0, 50):
            if time.time() - t0 > deadline:
                break
            a = r1
            b = 1.0
            c = r2
            total = a + b + c
            h = [0.0] * N
            h[0] = a / total
            h[1] = b / total
            h[2] = c / total
            s = compute_score(h)
            if s > best_score:
                best_score = s
                best_h = h[:]
    
    # Try 4-point clusters
    for r1 in np.linspace(0.1, 2.0, 30):
        if time.time() - t0 > deadline:
            break
        for r2 in np.linspace(0.1, 2.0, 30):
            if time.time() - t0 > deadline:
                break
            for r3 in np.linspace(0.1, 2.0, 30):
                if time.time() - t0 > deadline:
                    break
                a = r1
                b = 1.0
                c = 1.0
                d = r3
                total = a + b + c + d
                h = [0.0] * N
                h[0] = a / total
                h[1] = b / total
                h[2] = c / total
                h[3] = d / total
                s = compute_score(h)
                if s > best_score:
                    best_score = s
                    best_h = h[:]
    
    if best_h is None:
        # Fallback: known good 2-point 2:1
        best_h = [0.0] * N
        best_h[0] = 2.0 / 3.0
        best_h[1] = 1.0 / 3.0
    
    return best_h
# EVOLVE-BLOCK-END
