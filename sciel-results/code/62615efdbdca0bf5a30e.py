def priority(vector, n):
    # Count the number of 1s in the vector
    # Vectors with fewer 1s form fewer affine lines (as middle points)
    # and are safer to add early in greedy cap set construction
    count_ones = sum(1 for x in vector if x == 1)
    # Higher priority for vectors with fewer 1s
    return float(-count_ones)
