def construct():
    n = 256
    
    def get_conflicts(x, S):
        """
        Return a list of sets of elements in S that conflict with x.
        Each set represents a triple (a,b,c) involving x that violates the condition.
        """
        conflicts = []
        S_list = list(S)
        
        # Case 1: x is a. (x, b, c)
        for b in S_list:
            if b > x:
                for c in S_list:
                    if c >= b:
                        if (b + c) % x == 0:
                            conflicts.append({x, b, c})
                            
        # Case 2: x is b. (a, x, c)
        for a in S_list:
            if a < x:
                for c in S_list:
                    if c >= x:
                        if (x + c) % a == 0:
                            conflicts.append({x, a, c})
                            
        # Case 3: x is c. (a, b, x)
        for b in S_list:
            if b <= x:
                for a in S_list:
                    if a < b:
                        if (b + x) % a == 0:
                            conflicts.append({x, a, b})
                            
        return conflicts

    def is_valid_with_x(x, S):
        return len(get_conflicts(x, S)) == 0

    # Start with the known good solution
    S = set(range(171, 257))
    
    # Greedy addition
    for x in range(1, 171):
        if is_valid_with_x(x, S):
            S.add(x)
            
    # Local Search with Swaps
    # We try to add elements not in S. If conflict, try to remove an element from the conflict set.
    improved = True
    while improved:
        improved = False
        for x in range(1, 257):
            if x in S:
                continue
            if is_valid_with_x(x, S):
                S.add(x)
                improved = True
            else:
                conflicts = get_conflicts(x, S)
                # Try to resolve by removing one element from each conflict set
                # We only need to remove one element from the set S to break ALL conflicts involving x?
                # No, removing one element might break one conflict but leave others.
                # We need to find a y in S such that S - {y} + {x} is valid.
                # This means y must be part of EVERY conflict triple involving x?
                # Or rather, after removing y, no conflict triple remains.
                # A conflict triple is {x, u, v}. If we remove y, the triple is broken if y in {u, v}.
                # So y must be in the intersection of all conflict triples (excluding x)?
                # Not necessarily. If we remove y, and there is another conflict {x, u', v'} not involving y, it remains.
                # So we need y to be in ALL conflict triples? No, that's too strong.
                # We need to check if S - {y} + {x} is valid.
                # Let's just try removing each y in the union of all conflict elements (excluding x).
                candidates = set()
                for conf in conflicts:
                    candidates.update(conf - {x})
                
                for y in candidates:
                    new_S = S - {y}
                    if is_valid_with_x(x, new_S):
                        S = new_S
                        S.add(x)
                        improved = True
                        break # Restart the loop for x? Or just move on?
                        # Moving on is fine, we found a swap.
        
    return list(S)
