def construct():
    # Candidate sequence: 1, 2, 4, 7, 15, 31, 63, 127, 255, 511, 1023, 2047
    candidates = [1, 2, 4, 7, 15, 31, 63, 127, 255, 511, 1023, 2047]
    
    # Verify distinct subset sums
    sums = {0}
    for a in candidates:
        new_sums = set()
        for s in sums:
            new_sums.add(s + a)
        if len(sums | new_sums) != len(sums) + len(new_sums):
            # Collision found, fallback to binary
            return [1 << i for i in range(12)]
        sums |= new_sums
        
    return candidates
