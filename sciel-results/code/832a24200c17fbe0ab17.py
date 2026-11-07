import numpy as np
import time

_start_time = time.time()
_DEADLINE = 300.0
_MARGIN = 5.0

def acquisition(mu, sigma, f_best):
    # Time guard
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE - _MARGIN:
        # Fallback to exploitation (lowest mean)
        return -mu

    # LCB for minimization: -mu + beta * sigma
    # Using beta=0.5 for balanced exploration/exploitation
    beta = 0.5
    return -mu + beta * sigma
