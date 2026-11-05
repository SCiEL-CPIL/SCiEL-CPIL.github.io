"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    # OneMax with 100 bits: the global optimum is all ones.
    # The trivial optimal solution is "1" * N, which scores 100.
    # No search is needed; we return the optimal string directly.
    # Deadline guard: if we ever spent time (we don't), return best so far.
    start = time.monotonic()
    deadline = start + 140.0  # safety margin below 150s kill
    # Optimal solution: all ones.
    best = "1" * N
    # No further work needed; return immediately.
    return best
# EVOLVE-BLOCK-END
