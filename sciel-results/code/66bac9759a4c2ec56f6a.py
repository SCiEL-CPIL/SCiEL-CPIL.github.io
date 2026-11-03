def find_sidon(n):
    S = []
    sums = set()
    for x in range(n):
        new_sums = set()
        valid = True
        for s in S:
            new_sums.add(x + s)
        new_sums.add(2 * x)
        
        # Check collision with existing sums
        # Note: sums contains sums of pairs from S.
        # We need to ensure new_sums does not intersect with sums.
        if new_sums & sums:
            valid = False
        
        if valid:
            S.append(x)
            sums.update(new_sums)
            
    return S
