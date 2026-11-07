def construct():
    # Check for triples in range [211, 239]
    # If found, return the one with minimum max
    best_triple = None
    best_max = float('inf')
    
    # Range to check: 211 to 239 (max < 240)
    # If we find a triple with max < 240, it's better than the known pair (240)
    for a in range(211, 240):
        for b in range(a, 240):
            for c in range(b, 240):
                prod = 210 * a * b * c
                # Check if perfect square
                root = int(prod ** 0.5)
                if root * root == prod:
                    m = max(a, b, c)
                    if m < best_max:
                        best_max = m
                        best_triple = (a, b, c)
    
    if best_triple:
        return list(best_triple)
        
    # Fallback to best pair (224, 240)
    # Verify 210 * 224 * 240 is square
    # 210 * 224 * 240 = 210 * 53760 = 11289600
    # sqrt(11289600) = 3360
    # 3360^2 = 11289600. Correct.
    return [224, 240]
