"""Optimized construct for the forbidden-triple sum problem."""

def construct():
    """
    Return a list of distinct integers in 1..128 maximizing set size,
    subject to: no three distinct members a, b, c exist such that
    a+b, a+c, and b+c are all also in the set.
    
    Strategy: Start with a large candidate set and iteratively remove
    elements that are part of forbidden configurations, using a greedy
    approach that removes the element involved in the most conflicts.
    """
    n = 128
    
    # Start with all numbers
    S = set(range(1, n + 1))
    
    def has_forbidden_triple(S):
        """Check if there's a forbidden triple in S."""
        s = sorted(S)
        # For each pair (a, b) with a < b, check if a+b is in S
        # and if there exists c > b such that a+c in S and b+c in S
        for i, a in enumerate(s):
            for j in range(i + 1, len(s)):
                b = s[j]
                s_ab = a + b
                if s_ab not in S:
                    continue
                # Need c > b such that a+c in S and b+c in S
                for k in range(j + 1, len(s)):
                    c = s[k]
                    s_ac = a + c
                    s_bc = b + c
                    if s_ac in S and s_bc in S:
                        return (a, b, c)
        return None
    
    # Greedy removal: find forbidden triples and remove the element
    # that participates in the most such configurations
    while True:
        triple = has_forbidden_triple(S)
        if triple is None:
            break
        a, b, c = triple
        
        # Count how many forbidden triples each of a, b, c is in
        # Remove the one that's in the most triples
        counts = {a: 0, b: 0, c: 0}
        
        s = sorted(S)
        for i, x in enumerate(s):
            for j in range(i + 1, len(s)):
                y = s[j]
                if x + y not in S:
                    continue
                for k in range(j + 1, len(s)):
                    z = s[k]
                    if (x + z in S) and (y + z in S):
                        # Found forbidden triple (x, y, z)
                        for v in (x, y, z):
                            if v in counts:
                                counts[v] += 1
        
        # Remove the element with the highest count
        worst = max(counts, key=lambda v: counts[v])
        S.remove(worst)
    
    return sorted(S)
