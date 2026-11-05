"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    OneMax (100 bits): maximize the number of ones.
    
    The global optimum is trivially '1' * N (100 ones, score 100).
    Since the problem is deterministic and the optimum is known,
    we return it directly. A deadline guard is included for safety
    per the compute budget directive, though no search is needed.
    """
    start = time.monotonic()
    deadline = start + 140.0  # Safety margin below 150s kill limit
    
    # The optimal solution is all ones
    best = "1" * N
    
    # Verify the solution is valid (length N, binary)
    # This is a no-op check but ensures structural correctness
    if len(best) != N:
        raise ValueError(f"Expected length {N}, got {len(best)}")
    if any(c not in '01' for c in best):
        raise ValueError("Solution must be binary")
    
    # No search needed; return the known optimum immediately
    # The deadline guard is present but won't trigger
    if time.monotonic() > deadline:
        return best  # Fallback (unreachable)
    
    return best
# EVOLVE-BLOCK-END
