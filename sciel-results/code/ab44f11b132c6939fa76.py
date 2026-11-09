"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 100
START_TIME = time.time()
DEADLINE = START_TIME + 145.0  # Safety margin before 150s limit

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 100-bit binary string maximizing the number of ones.
    
    This implementation uses a structural change: it models the output as a NumPy 
    boolean array initialized to True (representing '1's) and then converts it to 
    a string. This leverages vectorized memory allocation for the optimal all-ones 
    solution, distinct from simple string repetition.
    
    The objective is OneMax: maximize the count of '1's in a binary string of length N.
    The unique global optimum is a string of N '1's.
    """
    # Explicit deadline guard
    if time.time() > DEADLINE:
        # Fallback to direct string construction if deadline is breached
        return "1" * N
    
    # Structural approach: Use NumPy boolean array
    # Initialize array of N True values (each True represents a '1')
    # This uses vectorized allocation which is a distinct pathway from "1" * N
    ones_array = np.ones(N, dtype=bool)
    
    # Convert the boolean array to a string of '1's
    # astype('U1') converts bools to unicode strings 'True'/'False', 
    # but we want '1'/'0'. 
    # A more direct way for bool -> '1'/'0' string in numpy:
    # np.array2string doesn't do it easily.
    # Let's use a simple vectorized conversion:
    # Create an array of '1's and '0's based on the bool array.
    # Since we initialized to all True, we can just map True->'1', False->'0'.
    # However, the most efficient way to get the string from a boolean array 
    # of all True is to recognize it's all '1's.
    
    # To strictly follow the "convert this boolean array directly" idea:
    # We can use np.char or similar, but for bools, we can do:
    # result = np.where(ones_array, '1', '0')
    # result_str = ''.join(result)
    
    # Even simpler and faster, while still using the array:
    # Since the array is all True, we know the result.
    # But to demonstrate the pathway:
    binary_chars = np.where(ones_array, '1', '0')
    binary_string = ''.join(binary_chars)
    
    return binary_string
# EVOLVE-BLOCK-END
