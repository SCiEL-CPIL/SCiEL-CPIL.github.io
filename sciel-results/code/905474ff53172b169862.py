def construct():
    from math import gcd
    
    # Generate all moduli in [20, 200]
    moduli = list(range(20, 201))
    # Sort by m ascending (to maximize sum of 1/m)
    moduli.sort()
    
    selected = []
    for m in moduli:
        # Check if m is coprime to all already selected
        compatible = True
        for sm in selected:
            if gcd(m, sm) != 1:
                compatible = False
                break
        if compatible:
            selected.append(m)
    
    return [[m, 0] for m in selected]
