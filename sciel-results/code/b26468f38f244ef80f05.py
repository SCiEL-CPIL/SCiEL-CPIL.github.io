def construct():
    # Core: All subsets of {0, 1, 2, 3, 4}
    core = []
    for mask in range(32):
        s = []
        for b in range(5):
            if (mask >> b) & 1:
                s.append(b)
        core.append(s)
    
    # Tail: 8 sets extending the core with a chain on {5, ..., 11}
    # Base is {0, 1, 2, 3, 4}
    base = [0, 1, 2, 3, 4]
    tail = []
    # We need 8 distinct sets.
    # Chain: {5}, {5,6}, {5,6,7}, ...
    # Set i (0-indexed) will be base + [5, 6, ..., 5+i]
    for i in range(8):
        s = base.copy()
        for j in range(5, 5 + i + 1):
            s.append(j)
        tail.append(s)
        
    result = core + tail
    return result
