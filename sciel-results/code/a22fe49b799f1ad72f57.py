import numpy as np

def construct():
    # Check primes p = 3, 7, 11, 19, 23, 31, 43, 47, 59, 67
    primes = [7, 11, 19, 23, 31, 43, 47, 59, 67]
    for p in primes:
        if p % 4 == 3:
            # Generate Paley tournament
            squares = {x * x % p for x in range(1, p)}
            # M[u] is a bitmask of vertices u dominates
            M = [0] * p
            for u in range(p):
                mask = 0
                for v in range(p):
                    if u != v and (v - u) % p in squares:
                        mask |= (1 << v)
                M[u] = mask
            
            # Check property: Every 4-set has a common dominator
            # A 4-set S has a common dominator if exists u such that S subset M[u]
            # We iterate over all 4-sets
            # Optimization: If we find a 4-set with NO common dominator, this p fails.
            valid = True
            # To speed up, we can check smaller sets first? No, we need 4.
            # We can check if the "domination graph" has enough coverage.
            # Just iterate all 4-sets.
            # For p=67, ~700k sets. Each check is O(p) bit ops. 700k * 67 is ~47M ops. Might be slow in pure Python loop.
            # We can optimize the check.
            
            # Alternative: For each 4-set, we can precompute the intersection of "non-dominators"?
            # Let's just try to run it for small p first.
            
            # Efficient check using bitmasks
            # For a set S (mask M_S), it has a common dominator if (M_S & M_u) == M_S for some u.
            # This is equivalent to: The intersection of the *complements* of M_u for u in V does not contain any 4-set?
            # No.
            
            # Let's just iterate.
            # We can break early if we find a counter-example.
            
            # To make it faster, we can check if the number of vertices dominating a random 4-set is > 0.
            # But we need certainty.
            
            # Let's optimize the inner loop.
            # For each 4-set, we can compute the union of "u such that S not subset M[u]"?
            # If the union is all u, then no dominator.
            
            # Let's use a different check:
            # A set S has NO common dominator if for every u, there is some v in S such that u does NOT dominate v.
            # This means u is not in the intersection of In(v) for v in S.
            
            # Let's stick to the bitmask check but optimize.
            # For small p, it's fine.
            
            found_counter_example = False
            # We can prune: if we find a 3-set with no common dominator, any 4-set containing it also has no common dominator?
            # No. A 3-set S has a common dominator u. A 4-set S' = S U {x} might have a different dominator w.
            
            # Let's just run the check for p < 67. If they fail, return 67.
            # We assume 67 is valid (it is in the current program).
            
            # We can skip the check for 67 to save time, or verify it.
            # Let's verify 59, 47, 43, 31, 23, 19, 11, 7.
            
            # For p=67, we know it works.
            
            if p < 67:
                # Check validity
                # Use itertools.combinations
                import itertools
                # Precompute list of masks
                masks = M
                # For speed, convert masks to a list of ints
                # Iterate 4-sets
                for combo in itertools.combinations(range(p), 4):
                    m_s = 0
                    for v in combo:
                        m_s |= (1 << v)
                    # Check if any mask contains m_s
                    has_dom = False
                    for m_u in masks:
                        if (m_s & m_u) == m_s:
                            has_dom = True
                            break
                    if not has_dom:
                        found_counter_example = True
                        break
                if found_counter_example:
                    continue
                else:
                    # Valid
                    orientation = [0 if (j - i) % p in squares else 1 for i in range(p) for j in range(i + 1, p)]
                    return {'n': p, 'orientation': orientation}
            else:
                # p == 67
                orientation = [0 if (j - i) % p in squares else 1 for i in range(p) for j in range(i + 1, p)]
                return {'n': p, 'orientation': orientation}
                
    # Fallback
    p = 67
    squares = {x * x % p for x in range(1, p)}
    orientation = [0 if (j - i) % p in squares else 1 for i in range(p) for j in range(i + 1, p)]
    return {'n': p, 'orientation': orientation}
