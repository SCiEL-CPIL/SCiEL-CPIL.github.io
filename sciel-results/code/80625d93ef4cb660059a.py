"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

def construct():
    deadline = time.time() + 85.0
    best_result = []
    
    # Try multiple starting strategies
    strategies = [
        list(range(192, 0, -1)),  # descending
        list(range(1, 193)),      # ascending
        # Mix: start with large numbers, then fill in
        [x for x in range(192, 96, -1)] + [x for x in range(96, 0, -1)],
    ]
    
    for strategy in strategies:
        if time.time() > deadline:
            break
        selected = []
        for x in strategy:
            if time.time() > deadline:
                break
            # Check if adding x violates the condition
            valid = True
            # For each existing element a in selected, check if a divides sum of any 3 distinct others from (selected + [x]) excluding a
            # Also check if x divides sum of any 3 distinct others from (selected + [x]) excluding x
            candidate = selected + [x]
            
            for a in candidate:
                others = [v for v in candidate if v != a]
                for t in combinations(others, 3):
                    if sum(t) % a == 0:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                selected.append(x)
        
        if len(selected) > len(best_result):
            best_result = selected
    
    return best_result
# EVOLVE-BLOCK-END
