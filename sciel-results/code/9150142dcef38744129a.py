"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 100-bit binary string maximizing ones.
    
    Models the construction as a deterministic bitwise OR operation between a 
    zero-initialized integer and a mask of 2^100 - 1. This leverages Python's 
    arbitrary-precision integer arithmetic to generate the all-ones state in 
    constant time. The resulting integer is converted to a binary string using 
    bin(), stripping the '0b' prefix.
    
    Includes an explicit deadline guard to ensure compliance with the 150s limit.
    """
    start_time = time.time()
    # Safety margin: return before the hard 150s kill switch
    deadline = start_time + 149.0 

    # 1. Initialize zero
    val = 0
    
    # 2. Create mask of N bits set to 1 (2^N - 1)
    # For N=100, this is a 100-bit integer with all bits 1
    mask = (1 << N) - 1
    
    # 3. Bitwise OR to set all bits
    # val = 0 | mask => val = mask
    val = val | mask
    
    # Deadline check (constant time operations, so this is effectively unreachable 
    # but required by the directive for safety)
    if time.time() > deadline:
        # Fallback to direct string construction if somehow delayed
        return "1" * N

    # 4. Convert integer to binary string
    # bin(val) returns '0b111...1'
    binary_string = bin(val)[2:]
    
    # 5. Ensure exact length N
    # For all-ones, len(binary_string) should be exactly N.
    # However, we pad/trim to be strictly compliant with format requirements.
    if len(binary_string) < N:
        binary_string = binary_string.zfill(N)
    elif len(binary_string) > N:
        binary_string = binary_string[-N:]
        
    return binary_string
# EVOLVE-BLOCK-END
