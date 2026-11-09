import numpy as np
from scipy.special import comb
import time

def construct():
    start_time = time.time()
    deadline = start_time + 570.0  # 30s safety margin
    
    n = 12
    
    # Compute roots of generalized Laguerre L_12^(alpha)(x)
    # L_n^(alpha)(x) = sum_{k=0}^{n} (-1)^k * binom(n+alpha, n-k) * x^k / k!
    
    def get_laguerre_roots(alpha):
        coeffs = np.zeros(n + 1)
        for k in range(n + 1):
            binom_val = comb(n + alpha, n - k, exact=False)
            coeffs[k] = (-1)**k * binom_val / np.math.factorial(k)
        roots = np.roots(coeffs)
        roots = np.real(roots[np.isreal(roots)])
        roots = roots[roots > 0]
        return np.sort(roots)
    
    # Try various alpha values, scale to max=480, and pick the best
    # Based on knowledge: alpha=11 gives near-uniform spacing
    # Let's use alpha=11 scaled to match the current best magnitude
    
    best_roots = None
    for alpha in [8, 9, 10, 11, 12, 13, 14]:
        try:
            roots = get_laguerre_roots(alpha)
            if len(roots) == 12:
                # Scale to max = 480
                scale = 480.0 / roots[-1]
                scaled = roots * scale
                if best_roots is None or alpha == 11:
                    best_roots = scaled
        except Exception:
            continue
    
    if best_roots is None:
        # Fallback to arithmetic progression
        best_roots = np.array([40.0 * (i + 1) for i in range(12)])
    
    # Ensure positive and sorted
    best_roots = np.sort(np.abs(best_roots))
    
    return [float(r) for r in best_roots]
