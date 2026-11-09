def construct():
    m = 12
    # Return 12 distinct positive roots, all very large
    # to maximize their product and minimize the C4 bound
    upper = 1e6
    # Use a geometric or arithmetic spacing near the upper bound
    roots = [upper - i * 0.1 for i in range(m)]
    return roots
