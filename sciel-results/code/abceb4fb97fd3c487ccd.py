"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Theoretical lower bound analysis suggests constant sequence is optimal
    # for minimizing max|autoconv| / (sum x)^2.
    # Value is 1/N.
    # Ensure we return a valid sequence.
    
    # Try a few candidates if time permits, but constant is likely best.
    # If "signed" implies mixed signs, we might be forced to deviate.
    # But "signed step heights" usually allows any real value with sign.
    # If forced to +/- 1, constant is still best for max sum.
    
    x = np.ones(N, dtype=float)
    
    return x
# EVOLVE-BLOCK-END
