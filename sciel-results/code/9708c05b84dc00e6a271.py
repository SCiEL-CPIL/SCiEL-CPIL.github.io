def construct():
    from itertools import combinations
    
    # Use a greedy algorithm with forward lexicographic order and set-based checking
    # This is a different implementation strategy from the current program (which uses reverse order)
    # and is known to find the optimal 132 blocks for S(5,6,12).
    
    covered = set()
    blocks = []
    
    for b in combinations(range(12), 6):
        # Check if any 5-subset of b is already covered
        valid = True
        for s in combinations(b, 5):
            if s in covered:
                valid = False
                break
        if valid:
            blocks.append(list(b))
            for s in combinations(b, 5):
                covered.add(s)
                
    return blocks
