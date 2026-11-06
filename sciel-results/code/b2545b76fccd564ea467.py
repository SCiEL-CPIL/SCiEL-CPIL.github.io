"""OneMax direct binary construction (100 bits)."""
import time
import numpy as np

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 100-bit binary string maximizing ones.
    The optimal solution is all ones, which scores 100 (vs 50 for seed).
    Uses NumPy boolean array construction as per directive.
    """
    # Track deadline with safety margin
    start = time.time()
    deadline = start + 140  # 10s safety margin before 150s kill
    
    # Model the binary string as a NumPy boolean array initialized to True
    # This leverages vectorized memory representation.
    bits = np.ones(N, dtype=bool)
    
    # Convert the boolean array to a string of '1's
    # np.array2string or join can be used, but simple join is fast enough for N=100
    # Using a list comprehension for clarity and speed on small N
    result = "".join("1" if b else "0" for b in bits)
    
    # Verify length is exactly N
    if len(result) != N:
        result = result[:N].ljust(N, "1")
        
    # Check deadline (though this operation is near-instant)
    if time.time() > deadline:
        # Return best result found (which is the optimal one)
        return result

    return result
# EVOLVE-BLOCK-END
