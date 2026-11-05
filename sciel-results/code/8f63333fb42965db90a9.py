"""
OneMax direct binary construction (100 bits)
Constructs a 100-bit binary string maximizing ones.
"""

import time

N = 100
DEADLINE_SECONDS = 120.0

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the optimal 100-bit binary string for OneMax.
    Uses a sequential greedy decision process for each bit,
    protected by an explicit deadline guard.
    """
    start_time = time.time()
    
    # The optimal bit for OneMax is always '1'
    # We construct the string bit by bit to represent a "sequential search" structure
    bits = []
    
    for i in range(N):
        # Check deadline with safety margin (e.g., 10 seconds remaining)
        if time.time() - start_time > DEADLINE_SECONDS - 10.0:
            # If we somehow ran out of time (impossible for 100 iterations),
            # return what we have so far (which would be all 1s up to that point)
            # But since this loop is nanoseconds fast, this branch is never taken.
            break
        
        # Greedy choice: 1 is always better than 0
        bits.append('1')
        
    return "".join(bits)
# EVOLVE-BLOCK-END
