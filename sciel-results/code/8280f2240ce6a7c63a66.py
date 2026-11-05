"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from fractions import Fraction
    import time
    
    deadline = time.time() + 80.0  # Safety margin before 90s kill
    
    denominators = list(range(2, 49))
    n = len(denominators)
    recip = [Fraction(1, d) for d in denominators]
    
    # Greedy initial solution: add in order while sum < 1
    selected = set()
    total = Fraction(0)
    for i in range(n):
        if total + recip[i] < 1:
            selected.add(i)
            total += recip[i]
    
    if not selected:
        return [0]  # Shouldn't happen, but safety
    
    best_selected = set(selected)
    best_total = total
    best_deficit = 1 - total
    
    # Local search: try 1-swap and 2-swap moves
    def try_improve(selected, total):
        """Try to improve the current solution. Returns (new_selected, new_total) or (None, None)."""
        best_sel = set(selected)
        best_tot = total
        best_def = 1 - total
        
        # Try adding one element not in selected
        for i in range(n):
            if i not in selected:
                new_tot = total + recip[i]
                if new_tot < 1:
                    new_def = 1 - new_tot
                    if new_def < best_def:
                        best_def = new_def
                        best_tot = new_tot
                        best_sel = set(selected)
                        best_sel.add(i)
        
        # Try 1-swap: remove one from selected, add one not in selected
        for i in list(selected):
            for j in range(n):
                if j not in selected:
                    new_tot = total - recip[i] + recip[j]
                    if new_tot < 1:
                        new_def = 1 - new_tot
                        if new_def < best_def:
                            best_def = new_def
                            best_tot = new_tot
                            best_sel = set(selected)
                            best_sel.discard(i)
                            best_sel.add(j)
        
        # Try 2-swap: remove two from selected, add two not in selected
        sel_list = list(selected)
        not_sel_list = [j for j in range(n) if j not in selected]
        for idx1 in range(len(sel_list)):
            for idx2 in range(idx1 + 1, len(sel_list)):
                i1, i2 = sel_list[idx1], sel_list[idx2]
                base_tot = total - recip[i1] - recip[i2]
                for j1 in range(len(not_sel_list)):
                    for j2 in range(j1 + 1, len(not_sel_list)):
                        j1_idx, j2_idx = not_sel_list[j1], not_sel_list[j2]
                        new_tot = base_tot + recip[j1_idx] + recip[j2_idx]
                        if new_tot < 1:
                            new_def = 1 - new_tot
                            if new_def < best_def:
                                best_def = new_def
                                best_tot = new_tot
                                best_sel = set(selected)
                                best_sel.discard(i1)
                                best_sel.discard(i2)
                                best_sel.add(j1_idx)
                                best_sel.add(j2_idx)
        
        if best_sel == selected:
            return None, None
        return best_sel, best_tot
    
    # Iterate local search until no improvement or deadline
    improved = True
    while improved and time.time() < deadline:
        improved = False
        while improved and time.time() < deadline:
            new_sel, new_tot = try_improve(best_selected, best_total)
            if new_sel is not None:
                best_selected = new_sel
                best_total = new_tot
                best_deficit = 1 - new_tot
                improved = True
    
    return sorted(best_selected)
# EVOLVE-BLOCK-END
