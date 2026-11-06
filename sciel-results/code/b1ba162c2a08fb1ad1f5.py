"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    from fractions import Fraction
    
    deadline = time.time() + 85  # safety margin before 90s kill
    
    denominators = list(range(2, 49))
    n = len(denominators)
    inv = [Fraction(1, d) for d in denominators]
    
    # Greedy baseline: pick in order while sum < 1
    selected = set()
    total = Fraction(0, 1)
    for i in range(n):
        if total + inv[i] < 1:
            selected.add(i)
            total += inv[i]
    
    best_selected = set(selected)
    best_total = total
    best_deficit = 1 - total  # positive deficit
    
    # Local search: try to improve by swaps and insertions
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try single insertion: add an unselected fraction if it keeps sum < 1
        # and reduces deficit (i.e., increases sum)
        for i in range(n):
            if i not in selected:
                new_total = total + inv[i]
                if new_total < 1:
                    if new_total > best_total:
                        best_total = new_total
                        best_selected = set(selected) | {i}
                        improved = True
                        # Don't break - keep trying, but update current state
                        selected = best_selected
                        total = best_total
        
        # Try single swap: remove one selected, add one unselected
        for i in list(selected):
            for j in range(n):
                if j not in selected:
                    new_total = total - inv[i] + inv[j]
                    if new_total < 1:
                        if new_total > best_total:
                            best_total = new_total
                            best_selected = (set(selected) - {i}) | {j}
                            improved = True
                            selected = best_selected
                            total = best_total
        
        # Try double swap: remove two selected, add two unselected
        if time.time() < deadline:
            sel_list = list(selected)
            unselected = [j for j in range(n) if j not in selected]
            for a_idx in range(len(sel_list)):
                if time.time() >= deadline:
                    break
                for b_idx in range(a_idx + 1, len(sel_list)):
                    if time.time() >= deadline:
                        break
                    i1, i2 = sel_list[a_idx], sel_list[b_idx]
                    for j1_idx in range(len(unselected)):
                        if time.time() >= deadline:
                            break
                        for j2_idx in range(j1_idx + 1, len(unselected)):
                            j1, j2 = unselected[j1_idx], unselected[j2_idx]
                            new_total = total - inv[i1] - inv[i2] + inv[j1] + inv[j2]
                            if new_total < 1:
                                if new_total > best_total:
                                    best_total = new_total
                                    best_selected = (set(selected) - {i1, i2}) | {j1, j2}
                                    improved = True
                                    selected = best_selected
                                    total = best_total
    
    return list(best_selected)
# EVOLVE-BLOCK-END
