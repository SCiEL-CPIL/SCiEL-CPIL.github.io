"""Structural change: Pure polynomial basis of higher degree (up to 4) with normalization."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Structural change: Replace mixed degree-3 terms (x^2y, xy^2) with pure degree-4 terms (x^4, y^4).
    This shifts the modeling family towards higher-order pure polynomials, which
    Inspiration 1 suggests is effective (score 7.02 vs 6.53).
    The basis remains normalized to unit RMS to assist BFGS convergence from all-ones.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    
    # Time guard (though calculation is fast, we respect the directive)
    start_time = time.time()
    
    # Precompute basis terms
    # Degrees 0-2 are kept as they form the core shape.
    # Degree 3: Keep x^3, y^3. Drop x^2y, xy^2.
    # Degree 4: Add x^4, y^4.
    
    t0 = np.ones_like(x)
    t1 = x
    t2 = y
    t3 = x * x
    t4 = y * y
    t5 = x * y
    t6 = x * x * x
    t7 = y * y * y
    t8 = x * x * x * x
    t9 = y * y * y * y
    
    def _normalize(t):
        rms = np.sqrt(np.mean(t * t))
        if rms < 1e-12:
            return t
        return t / rms
    
    t0 = _normalize(t0)
    t1 = _normalize(t1)
    t2 = _normalize(t2)
    t3 = _normalize(t3)
    t4 = _normalize(t4)
    t5 = _normalize(t5)
    t6 = _normalize(t6)
    t7 = _normalize(t7)
    t8 = _normalize(t8)
    t9 = _normalize(t9)
    
    pred = (params[0] * t0 + params[1] * t1 + params[2] * t2 +
            params[3] * t3 + params[4] * t4 + params[5] * t5 +
            params[6] * t6 + params[7] * t7 + params[8] * t8 +
            params[9] * t9)
    
    return pred
# EVOLVE-BLOCK-END
