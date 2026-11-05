"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 100-bit binary string maximizing ones.
    
    The OneMax objective is maximized by a string of all '1's.
    We use NumPy to create a vector of ones and convert it to a string,
    following the suggested linear algebra/vectorized approach, 
    though simple string repetition is also optimal and faster.
    Here we use the vectorized approach as per the strategy hint.
    """
    # Create a 1D array of ones with length N
    ones_vector = np.ones(N, dtype=int)
    # Convert each element to string and join
    # Using a list comprehension or map is standard, but join on a generator is fine
    return ''.join(str(bit) for bit in ones_vector)
# EVOLVE-BLOCK-END
