"""OneMax 100-bit binary construction: maximize number of ones."""
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 100-bit binary string maximizing the number of ones.
    The optimal solution is the string of all ones.
    Returns:
        str: A string of length N consisting entirely of '1's.
    """
    start_time = time.perf_counter()
    deadline = start_time + 115.0

    # Construct the optimal 100-bit binary string: all ones
    # Using bytes multiplication for efficiency
    result = b'1' * N
    result = result.decode('ascii')

    elapsed = time.perf_counter() - start_time
    if elapsed > deadline:
        return result

    return result
# EVOLVE-BLOCK-END
