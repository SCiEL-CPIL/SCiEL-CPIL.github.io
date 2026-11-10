"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    
    # Start with the known-good set {66,...,100} which has no subset summing to 200
    # because min 3-element sum is 66+67+68=201 > 200 and no 2-element sum reaches 200
    selected = list(range(66, 101))  # 35 elements
    reachable = {0}
    for x in selected:
        new = {s + x for s in reachable if s + x <= 200}
        reachable |= new
    
    # Try to add more elements from 1 to 65
    for x in range(65, 0, -1):
        if time.time() > deadline:
            break
        new = {s + x for s in reachable if s + x <= 200}
        if 200 not in new:
            selected.append(x)
            reachable |= new
    
    return selected
# EVOLVE-BLOCK-END
