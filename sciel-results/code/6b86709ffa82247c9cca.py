"""Program for OneMax direct binary construction (100 bits)."""
import time

N = 100
DEADLINE = 150.0
SAFETY_MARGIN = 1.0

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 100-bit binary string maximizing ones.
    
    The task is OneMax: maximize the number of '1' bits in a binary string of length N.
    The theoretical maximum is N (100) ones.
    
    Structural Change:
    Previous implementations used NumPy array initialization and formatting.
    This version uses direct Python string multiplication, which is the most 
    efficient and direct representation for this specific deterministic optimum.
    This reduces overhead and dependency complexity while maintaining the 
    maximal score.
    """
    start_time = time.time()
    
    # Direct construction of the global optimum: all bits are 1.
    candidate = '1' * N
    
    # Deadline guard (safety check)
    if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
        # In a real search loop, we would break here.
        # Here, the operation is instantaneous, so this is just a formal guard.
        pass
        
    return candidate
# EVOLVE-BLOCK-END
