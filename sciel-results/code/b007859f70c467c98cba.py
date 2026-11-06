"""OneMax direct binary construction (100 bits).

The objective is to maximize the number of ones in a 100-bit binary string.
The optimal solution is a string of all ones. Since the task is deterministic
and the optimum is trivially known, we return the optimal string directly.
"""
import time

N = 100
DEADLINE = time.time() + 140  # Safety margin before 150s kill

# EVOLVE-BLOCK-START
def construct():
    """Return a 100-bit binary string with all ones (optimal for OneMax)."""
    # Explicit deadline guard as per directive, though computation is O(1)
    if time.time() > DEADLINE:
        return '1' * N
    return '1' * N
# EVOLVE-BLOCK-END
