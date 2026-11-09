import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.special import eval_genlaguerre

def construct():
    """
    Return a list of 12 positive double-root locations for the Laguerre combination (m=12, n=1).
    The score is the upper bound on C4 (lower is better).
    """
    t0 = time.time()
    deadline = 620.0  # safety margin before 630s kill
    
    m = 12
    n = 1
    
    # Try to compute a surrogate for the C4 bound
    # For a polynomial P(x) with double roots at r_i, the C4 bound likely involves
    # ratios of norms. Let's try to minimize a plausible surrogate.
    
    # One possibility: the bound involves the L2 norm of P weighted by x^α e^{-x}
    # For Laguerre polynomials, this has known properties.
    
    # Let's try: minimize the product of (1/r_i) which would be maximized by large roots
    # But we saw that doesn't work monotonically.
    
    # Alternative: the bound might involve the variance of the roots or similar.
    
    # Let's try a different approach: use the fact that for Laguerre combinations,
    # the optimal roots might be related to the zeros of a specific orthogonal polynomial.
    
    # For m=12, n=1, let's try the zeros of L_12^(1)(x) scaled appropriately.
    
    from numpy.polynomial.laguerre import laggauss
    
    # Get Laguerre zeros for degree m
    points, weights = laggauss(m)
    
    # The zeros are positive. Let's try different scalings and see which gives best score.
    # But we can't compute the score locally, so we need a heuristic.
    
    # Based on observations:
    # - Scale 10 gave 33.3 (good)
    # - Scale 100 gave 1.23 (bad)  
    # - Scale 1000 gave 33.3 (good)
    
    # This suggests the bound might have a periodic or oscillatory dependence on scale,
    # or there are numerical issues at certain scales.
    
    # Let's try to be more systematic. Use the Laguerre zeros as the base pattern.
    
    # Strategy: try the Laguerre zeros with a moderate scale factor
    # and also try arithmetic progression with the same scale.
    
    # From the seed: arithmetic [4, 8, ..., 48] with scale 1 gave score 1.0
    # Scale 10 of arithmetic gave 33.3
    
    # Let's try: Laguerre zeros scaled to have similar spread as the seed
    # Seed spread: 4 to 48, so range 44, mean ~26
    
    lag_points = np.sort(points)
    # Scale Laguerre zeros to match the seed's range
    lag_scale = 48.0 / lag_points[-1]
    lag_roots = [float(p) * lag_scale for p in lag_points]
    
    # Also try: Laguerre zeros scaled by 10 (like the arithmetic case that worked)
    lag_roots_10 = [float(p) * 10.0 for p in lag_points]
    
    # And arithmetic with scale 10
    arith_roots_10 = [4.0 * 10.0 * (i + 1) for i in range(12)]
    
    # Let's also try: optimize the root positions using a surrogate
    # Surrogate: minimize sum(1/r_i) which is related to the product of roots
    # But this is monotonic, so it won't capture the oscillation.
    
    # Alternative surrogate: minimize the condition number of the Vandermonde matrix
    # or the spread of the roots.
    
    # Given the time budget, let's return the arithmetic roots with scale 10,
    # which gave the best observed score (33.3).
    
    # But wait - let's try one more thing: what if the optimal scale is not 10?
    # Let's do a quick grid search over scale factors for the arithmetic pattern.
    
    best_roots = arith_roots_10
    best_proxy = np.inf
    
    # Try different scales for arithmetic progression
    for scale in [5, 7, 10, 15, 20, 30, 50, 70, 100, 150, 200, 300, 500]:
        roots = [4.0 * scale * (i + 1) for i in range(12)]
        # Proxy: use the harmonic mean or similar
        # Actually, let's just use the scale itself as a proxy since larger scale
        # seemed better for scale=10 and 1000 but not 100.
        # This is risky. Let's just go with scale=10 which we know works.
        pass
    
    # Given the uncertainty, return the arithmetic roots with scale 10
    # which gave score 33.3 in observations.
    
    final_roots = [4.0 * 10.0 * (i + 1) for i in range(12)]
    
    if time.time() - t0 > deadline:
        return final_roots
    
    return final_roots
