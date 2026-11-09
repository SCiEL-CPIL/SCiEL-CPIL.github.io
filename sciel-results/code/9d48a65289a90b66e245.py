"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

# EVOLVE-BLOCK-START
def construct():
    # Deadline guard (redundant for O(1) solution but required by directive structure)
    start_time = time.time()
    deadline = start_time + 58.0  # 2s margin before 60s limit

    # The set {63, ..., 128} has size 66.
    # Min sum of two distinct elements is 63+64=127.
    # Sums in the set are only 127 and 128.
    # A forbidden triple requires 3 distinct sums in the set.
    # Since only 2 sums are possible, no forbidden triple exists.
    # This beats the baseline of 64 (range(65, 129)).
    
    # Construct the list of integers
    result = list(range(63, 129))
    
    # Verify time (should be negligible)
    if time.time() > deadline:
        # Fallback to a smaller known valid set if somehow slow (impossible here)
        return list(range(65, 129))
        
    return result
# EVOLVE-BLOCK-END
