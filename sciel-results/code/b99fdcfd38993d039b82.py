def construct():
    n = 256
    best_set = list(range(171, 257))  # fallback

    # Greedy set construction: try to build a large set satisfying the condition.
    # For each candidate a (from large to small), decide whether to include it.
    # Condition: for all a < b <= c in the set, a does not divide b+c.
    # If we process from largest to smallest, when considering adding a,
    # all existing elements b, c are > a, so we need to check if any pair
    # (b, c) with b <= c in the current set has b+c divisible by a.
    
    # Actually, let's think differently. Start with empty set, add elements from
    # largest to smallest. When adding a, check against all pairs in current set.
    # But that's O(n^3) worst case. For n=256 that's manageable.
    
    # Alternative: start with all elements [1, n] and greedily remove elements
    # that violate the condition.
    
    # Let's use a smarter approach: start with the high suffix and try to extend
    # downward, removing conflicting elements as needed.
    
    # For a given minimum m, the set [m, n] is valid iff for all a in [m, n],
    # there's no pair b, c in [m, n] with b > a, b <= c, and a | (b+c).
    # Since b, c >= m > a, we have b+c >= 2(m+1) (since b > a >= m, so b >= m+1).
    # The multiples of a to check are ka where ka >= 2(m+1) and ka <= 2n.
    
    # Let's just do a direct greedy: try to include elements from 256 down to 1.
    # Maintain a set S. For candidate a, check if adding a violates any condition.
    # Violation: exists b, c in S with a < b <= c and a | (b+c).
    # Since we add from large to small, when considering a, all elements in S are > a.
    # So we need: no pair (b, c) in S with b <= c such that a | (b+c).
    
    S = set()
    for a in range(256, 0, -1):
        # Check if adding a to S is valid
        valid = True
        # Collect all sums b+c where b, c in S, b <= c
        # We need to check if any such sum is divisible by a
        elems = sorted(S)
        for i in range(len(elems)):
            for j in range(i, len(elems)):
                s = elems[i] + elems[j]
                if s % a == 0:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            S.add(a)
    
    result = sorted(S)
    if len(result) > 0:
        return result
    return best_set
