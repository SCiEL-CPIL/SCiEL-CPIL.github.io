def construct():
    """
    Returns a 100-bit binary string consisting entirely of ones.
    This is the global optimum for the OneMax task (100 ones out of 100 bits).
    
    The solution is stored as a pre-allocated NumPy array and converted to a string
    upon invocation, leveraging vectorized memory operations.
    A deadline guard is included for safety.
    """
    start = time.perf_counter()
    deadline = start + 145.0  # 5 second safety margin before 150s limit
    
    # Convert the pre-allocated NumPy array to a binary string
    # This is O(N) but N=100 is negligible
    result = "".join(map(str, _OPTIMAL_ARR))
    
    # Safety check: if somehow we're approaching the deadline, return immediately
    if time.perf_counter() > deadline:
        return result
    
    return result
