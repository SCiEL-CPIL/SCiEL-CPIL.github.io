import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a 100-bit binary string maximizing the number of ones.
    The global maximum is N=100, achieved by the vector of all ones.
    This is the theoretical upper bound for the OneMax objective.
    """
    start_time = time.time()
    # Safety margin: 150s limit, stop at 145s
    deadline = start_time + 145.0
    
    # Direct construction of the global optimum
    result = "1" * N
    
    # Explicit deadline guard as required by directive
    if time.time() > deadline:
        return result
        
    return result
# EVOLVE-BLOCK-END
