"""OneMax 100-bit binary construction: maximize number of ones."""
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 100-bit binary string maximizing the number of ones.
    The optimal solution for OneMax is all ones. We return it directly.
    Includes a deadline guard as per directive, though this computation is instantaneous.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # The optimal solution for OneMax is all ones
    result = "1" * N
    
    # Check deadline (should never trigger for this simple operation)
    if time.time() > deadline:
        # If somehow we exceeded deadline, return best result found
        return result
    
    return result
# EVOLVE-BLOCK-END
