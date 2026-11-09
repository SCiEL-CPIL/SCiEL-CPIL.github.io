import time
import numpy as np
from scipy.special import roots_laguerre

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 600.0
    
    try:
        m = 12
        
        # Hypothesis: The "refined" construction might use generalized Laguerre polynomials L_m^(alpha)
        # and the optimal alpha minimizes some spread metric (like variance) of the roots,
        # which in turn minimizes the upper bound on C4.
        
        best_roots = roots_laguerre(m).real.tolist()
        best_metric = float('inf')
        
        # Search over alpha
        # Alpha must be > -1 for roots to be real and positive.
        # Let's try a range of alphas.
        alphas = np.arange(-0.9, 5.1, 0.1)
        
        for alpha in alphas:
            if time.time() > deadline - 60:
                break
            try:
                roots = roots_laguerre(m, alpha)
                # Check if roots are positive and real
                if not np.all(np.isreal(roots)):
                    continue
                roots_real = roots.real
                if np.any(roots_real <= 0):
                    continue
                
                # Metric: Variance of the roots
                # If the bound is related to the spread of the support, lower variance might be better.
                # Alternatively, maybe the "condition number" or "sum of squares" is better.
                # Let's try variance first.
                metric = np.var(roots_real)
                
                # Another possibility: The bound is related to the product of roots or sum.
                # But variance is a good proxy for "tightness".
                
                if metric < best_metric:
                    best_metric = metric
                    best_roots = sorted(roots_real.tolist())
                    
            except Exception:
                continue
                
        return best_roots
        
    except Exception as e:
        # Fallback to seed
        return [4.0 * (i + 1) for i in range(12)]
# EVOLVE-BLOCK-END
