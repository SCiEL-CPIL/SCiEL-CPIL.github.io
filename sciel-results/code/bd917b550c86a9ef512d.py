def priority(vector, n):
    # Calculate Hamming weight (number of non-zero elements)
    w = sum(1 for v in vector if v != 0)
    
    # Prioritize vectors with Hamming weight n-1
    # Then n and n-2
    # Then n-1 +/- 2, etc.
    
    # Target weight is n-1
    target = n - 1
    dist = abs(w - target)
    
    # Higher priority for smaller distance
    # Scale to [0, 1]
    # Max distance is n
    return (n - dist) / n
