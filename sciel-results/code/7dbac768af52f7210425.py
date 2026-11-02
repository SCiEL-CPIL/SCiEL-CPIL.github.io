import numpy as np

N = 400

def construct():
    # Alternating signs to minimize autocorrelation L1 norm
    # Pattern: 1, -1, 1, -1, ...
    return np.array([1.0 if i % 2 == 0 else -1.0 for i in range(N)])
