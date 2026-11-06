def construct():
    from fractions import Fraction
    import time
    import random
    
    start_time = time.time()
    deadline = 85.0
    
    denominators = list(range(2, 49))
    n = len(denominators)
    recip = [Fraction(1, d) for d in denominators]
    
    best_selected = None
    best_deficit = Fraction(1)
    
    def update(sel, tot):
        nonlocal best_selected, best_deficit
        if tot < 1:
            deficit = 1 - tot
            if deficit < best_deficit:
                best_deficit = deficit
                best_selected = tuple(sorted(sel))
    
    def local_search(sel, tot):
        nonlocal best_selected, best_deficit
        sel = frozenset(sel)
        tot = tot
        while True:
            if time.time() > deadline:
                return
            improved = False
            
            # Try additions
            for i in range(n):
                if time.time() > deadline:
                    return
                if i not in sel:
                    new_tot = tot + recip[i]
                    if new_tot < 1:
                        new_deficit = 1 - new_tot
                        if new_deficit < best_deficit:
                            best_deficit = new_deficit
                            best_selected = tuple(sorted(sel | {i}))
                            sel = sel | {i}
                            tot = new_tot
                            improved = True
            
            # Try swaps
            for i in list(sel):
                if time.time() > deadline:
                    return
                for j in range(n):
                    if j not in sel:
                        new_tot = tot - recip[i] + recip[j]
                        if new_tot < 1:
                            new_deficit = 1 - new_tot
                            if new_deficit < best_deficit:
                                best_deficit = new_deficit
                                new_sel = set(sel)
                                new_sel.remove(i)
                                new_sel.add(j)
                                best_selected = tuple(sorted(new_sel))
                                sel = frozenset(new_sel)
                                tot = new_tot
                                improved = True
                                break
                if improved:
                    break
            
            if not improved:
                break
    
    # Greedy initialization
    sel = set()
    tot = Fraction(0)
    for i in range(n):
        if tot + recip[i] < 1:
            sel.add(i)
            tot += recip[i]
    update(sel, tot)
    local_search(sel, tot)
    
    # Random restarts
    random.seed(42)
    for restart in range(30):
        if time.time() > deadline:
            break
        sel = set()
        tot = Fraction(0)
        order = list(range(n))
        random.shuffle(order)
        for i in order:
            if tot + recip[i] < 1:
                sel.add(i)
                tot += recip[i]
        update(sel, tot)
        local_search(sel, tot)
    
    if best_selected is None:
        return [0]
    
    return list(best_selected)
