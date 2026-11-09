import numpy as np
import numpy.polynomial.laguerre as npl
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = 600.0
    
    try:
        # Get initial Laguerre roots
        roots, weights = npl.laggauss(12)
        roots = np.sort(roots).astype(float)
        
        # Try different scaling factors to see if we can improve
        # The optimal scaling might not be 1.0
        best_roots = roots
        best_score = float('inf')
        
        # We need to estimate the score. Since we don't have the exact scoring function,
        # let's try a few heuristic approaches.
        # One common approach in uncertainty problems is to optimize for the spread
        # or other statistical properties of the roots.
        
        # Try scaling factors
        for scale in np.linspace(0.5, 2.0, 10):
            candidate_roots = roots * scale
            # Heuristic: minimize the variance or maximize some measure
            # For now, let's just pick the middle scaling
            if scale == 1.0:
                best_roots = candidate_roots
        
        # Actually, let's think differently. The Cohn-Gonçalves construction
        # might use a different polynomial entirely.
        
        # Try using Gegenbauer or other orthogonal polynomials
        # Or try to construct roots that satisfy specific constraints
        
        # For now, return the standard Laguerre roots as they are the natural choice
        return best_roots.tolist()
        
    except Exception:
        # Fallback to seed if something goes wrong
        return [4.0 * (i + 1) for i in range(12)]
