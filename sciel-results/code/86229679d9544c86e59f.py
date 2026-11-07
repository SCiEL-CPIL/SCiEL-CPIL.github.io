def construct():
    # Known maximum number of MOLS of order 10 is 3.
    # We use an optimized backtracking search to find them.
    deadline = time.time() + 80  # Keep 10s buffer for 90s limit
    
    S = [_cyclic()]
    
    # Try to find up to 2 more squares (total 3)
    # If we can find more, we will, but 3 is the theoretical max.
    max_target = 3
    
    while len(S) < max_target:
        if time.time() > deadline - 2:
            break
        new_square = _find_orthogonal(S, deadline)
        if new_square is None:
            break
        S.append(new_square)
        
    if _verify(S):
        return S
        
    # Fallback
    return [_cyclic()]
