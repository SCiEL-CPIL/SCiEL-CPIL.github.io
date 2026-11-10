def construct():
    n = 12
    # We maintain the set of all possible differences of subset sums.
    # If we have a set A, let S(A) be the set of subset sums.
    # The difference set D(A) = {s1 - s2 | s1, s2 in S(A)}.
    # A new element x can be added to A if and only if x is not in D(A).
    # This ensures that the new subset sums S(A U {x}) = S(A) U (S(A) + x) are disjoint.
    
    diffs = {0}
    result = []
    
    for _ in range(n):
        # Find the smallest positive integer not in diffs
        x = 1
        while x in diffs:
            x += 1
        
        result.append(x)
        
        # Update the difference set
        # New diffs are old diffs, old diffs + x, and old diffs - x
        new_diffs = set(diffs)
        for d in diffs:
            new_diffs.add(d + x)
            new_diffs.add(d - x)
        diffs = new_diffs
        
    return result
