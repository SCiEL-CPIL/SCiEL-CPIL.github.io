def construct():
    import math
    
    # Initial set B: all numbers from 129 to 256
    B = set(range(129, 257))
    A = []
    
    # Precompute multiples for efficiency
    # For each a in 2..128, find multiples in B
    
    # We iterate a from 2 to 128
    # We want to maximize sum(1/x for x in A U B)
    # Greedy: try to add a if gain > cost
    
    # To ensure pairwise LCM > 256 in A, we check against existing A
    # Note: A elements are <= 128.
    
    for a in range(2, 129):
        # Check if a can be added to A
        # Condition: lcm(a, x) > 256 for all x in A
        can_add = True
        for x in A:
            if math.lcm(a, x) <= 256:
                can_add = False
                break
        
        if not can_add:
            continue
            
        # Calculate cost: remove multiples of a from B
        cost = 0.0
        to_remove = []
        for b in B:
            if b % a == 0:
                cost += 1.0 / b
                to_remove.append(b)
        
        gain = 1.0 / a
        
        if gain > cost:
            A.append(a)
            for b in to_remove:
                B.remove(b)
                
    return A + list(B)
