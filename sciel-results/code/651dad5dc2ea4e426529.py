import time
import numpy as np
from scipy.special import roots_legendre

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill
    
    try:
        nodes, weights = roots_legendre(8)
        # Map from [-1, 1] to [0, 1]
        nodes = 0.5 * (nodes + 1.0)
        weights = 0.5 * weights
        
        # Verify constraints
        assert len(nodes) == 8
        assert len(weights) == 8
        assert np.all(np.diff(nodes) > 0)  # strictly increasing
        assert np.all(nodes > 0) and np.all(nodes < 1)  # in (0,1)
        assert np.all(weights > 0)  # positive weights
        assert abs(np.sum(weights) - 1.0) < 1e-12  # sum to 1
        
        return {"nodes": nodes, "weights": weights}
    except Exception:
        # Fallback: composite midpoint
        n = 8
        return {"nodes": (np.arange(n) + 0.5) / n, "weights": np.full(n, 1.0 / n)}
