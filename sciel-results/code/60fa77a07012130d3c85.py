"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    # Deadline guard to ensure we finish well within the 150s limit
    t0 = time.perf_counter()
    limit = t0 + 145.0
    
    # OneMax objective is maximized by a string of all ones.
    # Construct the optimal solution directly.
    optimal_str = "1" * N
    
    # Safety check: if we somehow exceed the time (impossible for this op), return anyway
    if time.perf_counter() > limit:
        return optimal_str
        
    return optimal_str
# EVOLVE-BLOCK-END
