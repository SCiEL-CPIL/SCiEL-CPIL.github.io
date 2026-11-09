import itertools

def construct():
    n = 192
    selected = []
    
    # We iterate from largest to smallest to maximize the count.
    # Larger numbers are "safer" because they have fewer multiples in the range [1, n].
    for x in range(n, 0, -1):
        # Check if x can be added to 'selected'
        valid = True
        
        # Condition 1: x must not divide the sum of any 3 distinct elements in 'selected'
        # sum(u+v+w) % x != 0 for all distinct u,v,w in selected
        if len(selected) >= 3:
            for u, v, w in itertools.combinations(selected, 3):
                if (u + v + w) % x == 0:
                    valid = False
                    break
        if not valid:
            continue
            
        # Condition 2: For every a in 'selected', a must not divide the sum of any 3 distinct 
        # elements in 'selected' + {x} that do not include a.
        # Since 'selected' is already valid, we only need to check sums involving x.
        # The sums involving x are of the form x + u + v, where u, v are distinct elements in 'selected'.
        for a in selected:
            # We need distinct u, v from 'selected' such that a divides x + u + v.
            # Note: u and v must be distinct from each other.
            # Also u, v are distinct from a? 
            # The problem says "k=3 DISTINCT OTHER selected integers".
            # The set of summands is {x, u, v}. The element being tested is a.
            # So a must not be in {x, u, v}.
            # We know a is in 'selected', so a != x (since x is new).
            # We must ensure a != u and a != v.
            
            # Iterate over pairs u, v from 'selected'
            # Optimization: We can pre-filter or just check.
            for u, v in itertools.combinations(selected, 2):
                if u == a or v == a:
                    continue
                if (x + u + v) % a == 0:
                    valid = False
                    break
            if not valid:
                break
                
        if valid:
            selected.append(x)
            
    return selected
