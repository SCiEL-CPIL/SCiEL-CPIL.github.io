"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    t0 = time.monotonic()
    limit = t0 + 145.0
    s = "1" * N
    if time.monotonic() < limit:
        return s
    return s
# EVOLVE-BLOCK-END
