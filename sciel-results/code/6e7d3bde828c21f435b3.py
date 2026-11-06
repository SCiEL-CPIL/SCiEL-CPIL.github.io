def priority(vector, n):
    # Count occurrences of 0, 1, 2
    c0 = vector.count(0)
    c1 = vector.count(1)
    c2 = vector.count(2)
    
    # Ideal count is n/3
    ideal = n / 3.0
    
    # Deviation from ideal balance
    dev = (c0 - ideal)**2 + (c1 - ideal)**2 + (c2 - ideal)**2
    
    # We want to minimize deviation, so negate it for max priority
    # Add a small deterministic hash to break ties and introduce diversity
    # Using a simple polynomial hash
    h = 0
    for i, val in enumerate(vector):
        h = (h * 31 + val) % 1000000007
    
    # Scale hash to be small relative to deviation?
    # Deviation can be up to n^2. Hash is up to 1e9.
    # We want Balance to dominate.
    # Let's normalize hash to [0, 1]
    return -dev + (h / 1000000007.0)
