"""OneMax 100-bit binary construction: maximize number of ones."""
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 100-bit binary string maximizing the number of ones.
    The optimal solution is all ones: "1" * 100.
    We return it directly. This is provably optimal for the OneMax objective.
    Includes an explicit deadline guard for robustness, though execution is O(1).
    """
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit
    
    # The optimal solution is all ones
    result = "1" * N
    
    # Check deadline (though it will never be hit for O(1) work)
    if time.time() - start_time > deadline:
         # Fallback to a valid solution if we were somehow stuck (impossible here)
         return "1" * N
         
    return result
# EVOLVE-BLOCK-END
