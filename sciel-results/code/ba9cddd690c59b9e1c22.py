def construct():
    n = 16
    half = n // 2
    
    # Generate edges for K(8,8) using list comprehension
    # This structure is known to be diameter-2 critical
    edges = [
        [u, v] 
        for u in range(half) 
        for v in range(half, n)
    ]
    
    return edges
