import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a 100-bit binary string maximizing the number of ones.
    The global maximum is N, achieved by the vector of all ones.
    """
    start_time = time.time()
    # Safety margin: 150s limit, stop at 145s
    deadline = start_time + 145.0
    
    # Direct construction of the optimal solution
    # This is the global optimum for OneMax.
    result = "1" * N
    
    # Check if we somehow exceeded time (impossible for this operation, 
    # but satisfies the "explicit deadline guard" directive structure)
    if time.time() > deadline:
        # In a more complex search, we would return best_so_far here.
        # Here, we return the immediate optimal result.
        return result
        
    return result
# EVOLVE-BLOCK-END
