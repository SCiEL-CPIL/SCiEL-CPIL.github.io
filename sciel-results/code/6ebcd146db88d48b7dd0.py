import time
import numpy as np

N = 50
_START = time.time()
DEADLINE = 110.0

def construct():
    # The ratio is maximized by a delta function (single non-zero step).
    # R(delta) = 1 / dx = 100.
    # Any other shape reduces the ratio.
    h = np.zeros(N)
    h[N // 2] = 1.0
    return [float(v) for v in h]
