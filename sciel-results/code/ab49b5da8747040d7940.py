import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a 100-bit binary string maximizing ones.
    Uses an integer with all 100 bits set to 1, then converts to binary string.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    # Create an integer with all 100 bits set to 1
    # This is equivalent to 2^100 - 1
    bits_int = (1 << N) - 1
    
    # Check deadline (this will almost never trigger, but satisfies the directive)
    if time.time() > deadline:
        # Fallback to a valid, high-scoring string if we somehow hit the limit
        return "1" * N

    # Convert integer to binary string and strip the '0b' prefix
    binary_string = bin(bits_int)[2:]
    
    return binary_string
# EVOLVE-BLOCK-END
