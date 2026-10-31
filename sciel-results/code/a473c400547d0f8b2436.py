# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Constructs the optimal signed step function.
    The metric is max(|autoconv|) / (sum(signs))^2.
    For signs in {-1, 1}, max(|autoconv|) >= N (at lag 0) and (sum(signs))^2 <= N^2.
    The minimum ratio is 1/N, achieved when all signs are identical.
    """
    return np.ones(N, dtype=np.int32)
# EVOLVE-BLOCK-END
