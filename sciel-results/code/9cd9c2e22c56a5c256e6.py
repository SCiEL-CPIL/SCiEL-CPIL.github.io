import math

def construct():
    best_score = 0.0
    best_classes = []
    
    # We search for a common factor K
    # The moduli must be multiples of K
    # We can pick at most K moduli
    # To maximize sum(1/m), we pick the smallest K multiples of K in [20, 200]
    
    for K in range(1, 101): # K can be up to 200, but multiples become sparse
        multiples = [m for m in range(20, 201) if m % K == 0]
        if not multiples:
            continue
        
        # We can pick at most K moduli
        k_pick = min(K, len(multiples))
        
        # Pick the smallest k_pick multiples
        selected_moduli = sorted(multiples)[:k_pick]
        
        # Assign residues 0 .. k_pick-1
        # This ensures r_i != r_j mod K for i != j
        # Since gcd(m_i, m_j) is multiple of K, this ensures disjointness
        classes = []
        for i, m in enumerate(selected_moduli):
            r = i
            classes.append([m, r])
            
        # Verify validity (just in case, though logic says it's valid)
        valid = True
        for i in range(len(classes)):
            for j in range(i+1, len(classes)):
                m1, r1 = classes[i]
                m2, r2 = classes[j]
                g = math.gcd(m1, m2)
                if r1 % g == r2 % g:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            score = sum(1.0 / m for m, _ in classes)
            if score > best_score:
                best_score = score
                best_classes = classes

    # Fallback to seed if something went wrong
    if not best_classes:
        best_classes = [[20 * j, j - 1] for j in range(1, 11)]
        
    return best_classes
