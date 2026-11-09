def priority(vector, n):
    # Use a deterministic hash to generate a pseudo-random priority.
    # Random ordering is known to perform well for greedy cap set construction,
    # often outperforming structured orders like lexicographic or weight-based.
    return float(hash(vector) % 1000000000)
