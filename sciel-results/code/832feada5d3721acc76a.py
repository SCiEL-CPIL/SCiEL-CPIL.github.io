"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import hashlib
import time

_start_time = time.time()
_DEADLINE = _start_time + 280

def priority(vector, n):
    if time.time() > _DEADLINE:
        return 0.0
    h = int(hashlib.md5(vector).hexdigest(), 16)
    return (h % 1000000007) / 1000000007.0
# EVOLVE-BLOCK-END
