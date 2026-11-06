def construct():
    from fractions import Fraction
    import time
    
    deadline = time.time() + 80
    
    denominators = list(range(2, 49))
    n = len(denominators)
    fracs = [Fraction(1, d) for d in denominators]
    
    # Greedy initialization: add in order of decreasing value
    selected = set()
    total = Fraction(0)
    for i in range(n):
        if total + fracs[i] < 1:
            selected.add(i)
            total += fracs[i]
    
    best_total = total
    best_selected = set(selected)
    
    # Local search: try additions then swaps
    while time.time() < deadline:
        improved = False
        # Try single additions
        for i in range(n):
            if i not in selected:
                nt = total + fracs[i]
                if nt < 1:
                    selected.add(i)
                    total = nt
                    improved = True
                    break
        if improved:
            if total > best_total:
                best_total = total
                best_selected = set(selected)
            continue
        # Try single swaps
        found_swap = False
        for i in list(selected):
            for j in range(n):
                if j not in selected:
                    nt = total - fracs[i] + fracs[j]
                    if nt < 1:
                        selected.discard(i)
                        selected.add(j)
                        total = nt
                        found_swap = True
                        break
            if found_swap:
                break
        if found_swap:
            if total > best_total:
                best_total = total
                best_selected = set(selected)
        else:
            break
    
    return sorted(best_selected)
