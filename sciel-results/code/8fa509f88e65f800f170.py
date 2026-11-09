import random
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    # Explicit deadline guard to ensure we finish well before the 150s limit
    deadline = time.time() + 5.0
    
    # Exact local solve: For OneMax, the global optimum is a string of all '1's.
    # This is the most efficient convergence, equivalent to a greedy search
    # that flips all bits in a single optimal step.
    
    if time.time() > deadline:
        # Fallback: Return a valid string if we are already over deadline (unlikely)
        return '0' * N
        
    return '1' * N
# EVOLVE-BLOCK-END
