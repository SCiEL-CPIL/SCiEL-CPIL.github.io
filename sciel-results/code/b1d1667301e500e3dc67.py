import time

N = 100
DEADLINE = time.time() + 145  # 5s safety margin

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 100-bit binary string maximizing the number of ones.
    Uses direct string construction with a deadline guard.
    """
    start_time = time.time()
    
    # The optimal solution for OneMax is a string of all '1's.
    # Using string multiplication is the most direct and efficient method.
    result = "1" * N
    
    # Check deadline before returning (though construction is O(N) and fast)
    if time.time() - start_time > 140:
        # Fallback to a valid string if somehow slow (unlikely)
        return "1" * N
        
    return result
# EVOLVE-BLOCK-END
