"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = 55.0  # safety margin before 90s kill
    
    n = 256
    
    # Step 1: Greedy from high to low
    selected = set([1])
    for a in range(n, 1, -1):
        if time.time() - start_time > deadline:
            break
        can_add = True
        for b in selected:
            if gcd(a, b) != 1:
                can_add = False
                break
        if can_add:
            selected.add(a)
    
    # Helper: perform best swap (a,b) -> a*b with max gain, repeat
    def do_swaps():
        nonlocal selected, start_time
        while True:
            if time.time() - start_time > deadline:
                break
            best_gain = 0
            best_pair = None
            sel_list = list(selected)
            for i in range(len(sel_list)):
                for j in range(i + 1, len(sel_list)):
                    a = sel_list[i]
                    b = sel_list[j]
                    prod = a * b
                    if prod > n:
                        continue
                    if prod in selected:
                        continue
                    gain = prod - a - b
                    if gain > best_gain:
                        best_gain = gain
                        best_pair = (a, b)
            if best_pair is None:
                break
            a, b = best_pair
            selected.discard(a)
            selected.discard(b)
            selected.add(a * b)
    
    # Helper: try to add remaining numbers
    def do_adds():
        nonlocal selected, start_time
        for a in range(n, 1, -1):
            if time.time() - start_time > deadline:
                break
            if a in selected:
                continue
            can_add = True
            for b in selected:
                if gcd(a, b) != 1:
                    can_add = False
                    break
            if can_add:
                selected.add(a)
    
    # Step 2: Swaps
    do_swaps()
    
    # Step 3: Adds
    do_adds()
    
    # Step 4: More swaps
    do_swaps()
    
    # Step 5: Final adds
    do_adds()
    
    return list(selected)
# EVOLVE-BLOCK-END
