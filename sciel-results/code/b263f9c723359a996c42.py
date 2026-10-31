# EVOLVE-BLOCK-START
import time
_start_time = time.time()
TIME_LIMIT = 280.0  # safety margin

def priority(vector, n):
    # Favor vectors with fewer 1s; slight preference for 2s over 0s
    count_1 = sum(1 for x in vector if x == 1)
    count_2 = sum(1 for x in vector if x == 2)
    return -count_1 + 0.1 * count_2

# EVOLVE-BLOCK-END
