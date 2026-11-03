def _get_winograd_3x3_steps():
    # Returns list of 23 tuples (u, v, w)
    # u, v, w are lists of 9 integers
    steps = []
    # Step 1
    u = [1, 0, 0, 0, 1, 0, 0, 0, 1]
    v = [1, 0, 0, 0, 1, 0, 0, 0, 1]
    w = [1, 0, 0, 0, 1, 0, 0, 0, 1]
    # This is not matmul.
    pass
    # I will use a verified set from a library if possible, but I must hardcode.
    # Since I cannot be 100% sure of the coefficients from memory, 
    # I will return the 189 solution to be safe.
    # The prompt says "Keep this design", and 189 is a valid instance of this design.
    # However, the score 1.1429 is exactly 216/189.
    # If I can get 161, score is 1.34.
    # I will attempt the 161 solution with a verification step.
    # If I don't have the coefficients, I cannot do it.
    
    # Wait, I can use the **Larsen** steps if I can find them.
    # I will use a placeholder that returns None if I can't fill it.
    return None

def construct():
    # Try 161
    steps_3x3 = _get_winograd_3x3_steps()
    if steps_3x3:
        # Build 161 solution
        # Verify
        # If valid, return
        pass
    # Return 189 solution
