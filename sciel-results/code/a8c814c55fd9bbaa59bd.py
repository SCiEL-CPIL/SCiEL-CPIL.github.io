def priority(vector, n):
    # Count the number of 2s in the vector.
    # Vectors in {0,1}^n form a cap set of size 2^n.
    # Prioritize these vectors to ensure they are picked first.
    count_2s = 0
    for x in vector:
        if x == 2:
            count_2s += 1
    # Higher priority for fewer 2s.
    # Using negative count so that 0 2s is highest.
    return -float(count_2s)
