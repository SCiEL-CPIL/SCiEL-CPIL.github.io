def can_add_excluding(x, S, u):
    # Check if x is middle
    for a in S:
        if a == u: continue
        c = 2 * x - a
        if c != x and c != u and c in S:
            return False
    
    # Check if x is endpoint
    for b in S:
        if b == u: continue
        y = 2 * b - x
        if y != x and y != u and y in S:
            return False
            
    return True
