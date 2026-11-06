def priority(vector, n):
    c0 = 0
    c1 = 0
    c2 = 0
    for x in vector:
        if x == 0:
            c0 += 1
        elif x == 1:
            c1 += 1
        else:
            c2 += 1
            
    # Structural Score: Balance
    # High score for vectors with mixed 0s, 1s, 2s
    score = c0 * c1 + c1 * c2 + c0 * c2
    
    # Tie-breaker: Hash
    # Using a simple polynomial hash for speed and determinism
    h = 5381
    for x in vector:
        h = (h * 33 + x) & 0xFFFFFFFF
        
    return float(score) + (h / 4294967296.0) * 0.01
