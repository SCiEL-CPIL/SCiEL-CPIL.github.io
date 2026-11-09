import numpy as np

# Pre-generate a random vector for max n (e.g. 20)
_MAX_N = 20
_rng = np.random.RandomState(42)
_R = _rng.rand(_MAX_N)

# EVOLVE-BLOCK-START
def priority(vector, n):
    # Use a linear score based on a fixed random vector
    # This is fast with NumPy
    v = np.array(vector, dtype=np.float64)
    return float(np.dot(v, _R[:n]))
# EVOLVE-BLOCK-END
