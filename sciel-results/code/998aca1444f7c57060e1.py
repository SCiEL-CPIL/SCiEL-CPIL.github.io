"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

def construct():
    deadline = time.time() + 85.0
    selected = []
    existing_sums = set()
    
    for x in range(1, 4097):
        if time.time() > deadline:
            break
        
        # Quick check: if 3*x is already in existing_sums, skip
        if 3 * x in existing_sums:
            continue
            
        # Check all new sums involving x, with early termination
        valid = True
        new_sums_set = set()
        
        # (x, x, x)
        s = 3 * x
        if s in existing_sums or s in new_sums_set:
            valid = False
        else:
            new_sums_set.add(s)
        
        if valid:
            for i, y in enumerate(selected):
                # (x, x, y)
                s = 2 * x + y
                if s in existing_sums or s in new_sums_set:
                    valid = False
                    break
                new_sums_set.add(s)
                
                # (x, y, y)
                s = x + 2 * y
                if s in existing_sums or s in new_sums_set:
                    valid = False
                    break
                new_sums_set.add(s)
                
                # (x, y, z) for z > y
                for z in selected[i+1:]:
                    s = x + y + z
                    if s in existing_sums or s in new_sums_set:
                        valid = False
                        break
                    new_sums_set.add(s)
                if not valid:
                    break
        
        if valid:
            selected.append(x)
            existing_sums.update(new_sums_set)
    
    return selected
# EVOLVE-BLOCK-END
