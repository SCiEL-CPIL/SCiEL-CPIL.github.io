"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin
    
    denominators = list(range(2, 49))
    n = len(denominators)
    fracs = [Fraction(1, d) for d in denominators]
    
    # Greedy initial solution: add in order if fits
    selected = set()
    total = Fraction(0)
    for i in range(n):
        if total + fracs[i] < 1:
            selected.add(i)
            total += fracs[i]
    
    best_selected = set(selected)
    best_total = total
    
    # Local search: try swaps and additions
    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        
        # Try adding single elements
        for i in range(n):
            if i not in selected:
                new_total = total + fracs[i]
                if new_total < 1 and new_total > best_total:
                    selected.add(i)
                    total = new_total
                    best_selected = set(selected)
                    best_total = total
                    improved = True
                    break
        
        if time.time() > deadline:
            break
        
        # Try swaps: remove one, add one
        for rem in list(selected):
            if time.time() > deadline:
                break
            rem_total = total - fracs[rem]
            for add in range(n):
                if add not in selected:
                    new_total = rem_total + fracs[add]
                    if new_total < 1 and new_total > best_total:
                        selected.remove(rem)
                        selected.add(add)
                        total = new_total
                        best_selected = set(selected)
                        best_total = total
                        improved = True
                        break
                if time.time() > deadline:
                    break
    
    return sorted(best_selected)
# EVOLVE-BLOCK-END
