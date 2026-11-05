"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a 100-bit binary string maximizing the number of ones.
    Uses NumPy vectorized arithmetic to generate the optimal solution.
    Includes a deadline guard for safety.
    """
    start_time = time.monotonic()
    deadline = start_time + 140.0  # Safety margin below 150s kill limit
    
    # The global optimum for OneMax is all bits set to 1.
    # We construct this directly using NumPy for vectorized generation.
    # Define a 100-dimensional column vector of ones.
    ones_vec = np.ones(N, dtype=np.int8)
    
    # Apply the dot product with a unit vector (scalar 1) to demonstrate linear algebra operation.
    # This results in a scalar array of 100.
    # Then we flatten and convert to binary string.
    # Actually, the strategy says: "defining a 100-dimensional column vector of ones and applying the dot product with a unit vector, then flattening the resulting scalar array to generate the binary string."
    # Wait, dot product of two 100-dim vectors is a scalar. 
    # Let's re-read: "defining a 100-dimensional column vector of ones and applying the dot product with a unit vector, then flattening the resulting scalar array to generate the binary string."
    # This phrasing is slightly ambiguous. If we take the outer product or just element-wise multiplication with a unit vector?
    # "matrix-vector multiplication" is mentioned.
    # If we have a matrix of ones (100x1) and multiply by a vector (1,)? That's 100x1.
    # Let's stick to the simplest interpretation that yields 100 ones:
    # Create a vector of ones.
    # The "dot product with a unit vector" might refer to ensuring the values are 1.
    # Let's just use np.ones(N) as the core construction, wrapped in the requested linear algebra flavor if possible without breaking the "all ones" guarantee.
    
    # To satisfy the "linear algebra operation" strategy:
    # Let A be a 100x1 matrix of ones.
    # Let b be a 1x1 vector of ones (unit vector in 1D).
    # A @ b.T gives a 100x1 matrix of ones.
    # Flatten it.
    
    A = np.ones((N, 1), dtype=np.int8)
    b = np.ones((1, 1), dtype=np.int8)
    
    # Matrix-vector multiplication
    result = A @ b
    
    # Flatten the resulting array
    flat_result = result.flatten()
    
    # Convert to binary string
    # Ensure it's all 1s
    binary_str = ''.join(['1' if x == 1 else '0' for x in flat_result])
    
    # Check deadline (though this returns instantly)
    if time.monotonic() > deadline:
        # Fallback to a valid string if somehow we timed out (impossible here)
        return "1" * N
        
    return binary_str
# EVOLVE-BLOCK-END
