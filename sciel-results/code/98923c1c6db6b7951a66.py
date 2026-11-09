"""OneMax direct binary construction (100 bits)."""
import numpy as np
import time

N = 100
_DEADLINE_MARGIN_S = 5.0

# EVOLVE-BLOCK-START
def construct():
    start = time.monotonic()
    # Vectorized construction of the all-ones solution via a boolean array,
    # converted through a distinct numerical representation pathway.
    arr = np.ones(N, dtype=bool)
    str_arr = arr.astype('U1')
    result = ''.join(str_arr)
    # Explicit deadline guard with safety margin.
    if (time.monotonic() - start) > (150.0 - _DEADLINE_MARGIN_S):
        # Fallback: simplest valid construction if unexpected delay occurred.
        result = "1" * N
    return result
# EVOLVE-BLOCK-END
