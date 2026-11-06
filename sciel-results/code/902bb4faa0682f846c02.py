"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

def construct():
    n = 512
    deadline = time.time() + 50.0  # Safety margin for 60s limit
    
    # Precompute ranges of d for each x to speed up validity check
    # For each x, store list of (d, positions) that need to be checked
    # Actually, just iterating d is fine for N=512.
    
    best_set = set()
    best_size = 0
    
    # Initial greedy pass (deterministic) to get a baseline
    selected = set()
    for x in range(1, n + 1):
        valid = True
        # Check if x can be added without forming 4-AP
        # x is 4th term: x-3d, x-2d, x-d
        for d in range(1, (x - 1) // 3 + 1):
            if (x - d) in selected and (x - 2 * d) in selected and (x - 3 * d) in selected:
                valid = False
                break
        if valid:
            # x is 3rd term: x-2d, x-d, x+d
            for d in range(1, min((x - 1) // 2, (n - x) // 1) + 1):
                if (x - d) in selected and (x - 2 * d) in selected and (x + d) in selected:
                    valid = False
                    break
        if valid:
            # x is 2nd term: x-d, x+d, x+2d
            for d in range(1, min((x - 1) // 1, (n - x) // 2) + 1):
                if (x - d) in selected and (x + d) in selected and (x + 2 * d) in selected:
                    valid = False
                    break
        if valid:
            # x is 1st term: x+d, x+2d, x+3d
            for d in range(1, (n - x) // 3 + 1):
                if (x + d) in selected and (x + 2 * d) in selected and (x + 3 * d) in selected:
                    valid = False
                    break
        
        if valid:
            selected.add(x)
            
    if len(selected) > best_size:
        best_size = len(selected)
        best_set = set(selected)
        
    # Randomized Greedy with Restarts
    candidates = list(range(1, n + 1))
    
    while time.time() < deadline:
        random.shuffle(candidates)
        selected = set()
        
        for x in candidates:
            valid = True
            # x is 4th term
            for d in range(1, (x - 1) // 3 + 1):
                if (x - d) in selected and (x - 2 * d) in selected and (x - 3 * d) in selected:
                    valid = False
                    break
            if valid:
                # x is 3rd term
                for d in range(1, min((x - 1) // 2, (n - x) // 1) + 1):
                    if (x - d) in selected and (x - 2 * d) in selected and (x + d) in selected:
                        valid = False
                        break
            if valid:
                # x is 2nd term
                for d in range(1, min((x - 1) // 1, (n - x) // 2) + 1):
                    if (x - d) in selected and (x + d) in selected and (x + 2 * d) in selected:
                        valid = False
                        break
            if valid:
                # x is 1st term
                for d in range(1, (n - x) // 3 + 1):
                    if (x + d) in selected and (x + 2 * d) in selected and (x + 3 * d) in selected:
                        valid = False
                        break
            
            if valid:
                selected.add(x)
                
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = set(selected)
            
    return sorted(best_set)

# EVOLVE-BLOCK-END
