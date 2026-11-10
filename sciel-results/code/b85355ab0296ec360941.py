def construct():
    # Greedy: process moduli from small to large, pick smallest valid residue
    selected = []  # list of (m, r)
    
    for m in range(20, 201):
        for r in range(m):
            # Check if (m, r) conflicts with any selected pair
            conflict = False
            for (m2, r2) in selected:
                g = __import__('math').gcd(m, m2)
                if r % g == r2 % g:
                    conflict = True
                    break
            if not conflict:
                selected.append((m, r))
                break
    
    # Convert to list of lists
    return [[m, r] for m, r in selected]
