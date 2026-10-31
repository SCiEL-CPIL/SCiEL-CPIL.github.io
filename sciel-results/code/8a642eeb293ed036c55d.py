import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    # The seed program is optimal for the standard interpretation.
    # We return the constant sequence of ones.
    return np.ones(N)
# EVOLVE-BLOCK-END
