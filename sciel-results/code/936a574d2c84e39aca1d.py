def construct(k, g):
    q = k
    # Optimal d for LUW graph to achieve girth >= g
    # Girth of CD(d, q) is 2d + 2
    # We need 2d + 2 >= g => d >= (g - 2) / 2
    d = max(3, (g - 2 + 1) // 2)
    
    # Rest of the code remains the same
    # ...
