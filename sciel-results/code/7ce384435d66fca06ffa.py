import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a 100-bit binary string maximizing the number of ones.
    The optimal solution is all ones.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # The global optimum for OneMax is all bits set to 1.
    # We construct this directly using integer arithmetic as per directive.
    # 1. Create a mask of N ones: (1 << N) - 1
    # 2. OR with 0 (redundant but follows directive structure)
    # 3. Convert to binary string
    
    mask = (1 << N) - 1
    val = 0 | mask
    
    # Convert to binary string
    # bin(val) gives '0b...' followed by N bits of 1s
    # We strip the '0b' prefix
    result = bin(val)[2:]
    
    # Ensure length is exactly N (though bin(2^N-1) is always length N)
    if len(result) < N:
        result = result.zfill(N)
        
    # Check deadline (will never trigger for this trivial op, but ensures compliance)
    if time.time() > deadline:
        # Fallback: return best found (which is the same)
        pass
        
    return result
# EVOLVE-BLOCK-END
