def priority(vector, n):
    # Calculate interaction energy: sum of products of adjacent elements
    # This is a non-linear measure that differs from global count balance.
    energy = 0
    for i in range(n - 1):
        energy += vector[i] * vector[i + 1]
    
    # We want to minimize this energy?
    # Or maximize?
    # If energy is low, adjacent non-zeros are rare.
    # This promotes "sparse" or "alternating" patterns.
    # Let's return negative energy to prioritize low interaction.
    return -float(energy)
