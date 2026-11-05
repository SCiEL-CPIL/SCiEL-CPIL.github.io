import random

def has_4ap(selected):
    """Check if selected contains a nonconstant 4-AP."""
    s = set(selected)
    for a in s:
        for d in range(1, 513):
            if a + 3*d <= 512 and a+d in s and a+2*d in s and a+3*d in s:
                return True
    return False

def construct():
    best = set()
    best_size = 0
    
    # Try multiple random orderings
    for trial in range(200):
        random.seed(trial)
        order = list(range(1, 513))
        random.shuffle(order)
        selected = set()
        for x in order:
            # Check if adding x creates a 4-AP
            conflict = False
            for d in range(1, 513):
                # x could be any of the 4 terms
                # Case 1: x is the 4th term: x-3d, x-2d, x-d in selected
                if x - 3*d >= 1 and (x-3*d in selected) and (x-2*d in selected) and (x-d in selected):
                    conflict = True
                    break
                # Case 2: x is the 3rd term: x-2d, x-d, x+d in selected
                if x - 2*d >= 1 and x + d <= 512 and (x-2*d in selected) and (x-d in selected) and (x+d in selected):
                    conflict = True
                    break
                # Case 3: x is the 2nd term: x-d, x+d, x+2d in selected
                if x - d >= 1 and x + 2*d <= 512 and (x-d in selected) and (x+d in selected) and (x+2*d in selected):
                    conflict = True
                    break
                # Case 4: x is the 1st term: x+d, x+2d, x+3d in selected
                if x + 3*d <= 512 and (x+d in selected) and (x+2*d in selected) and (x+3*d in selected):
                    conflict = True
                    break
            if not conflict:
                selected.add(x)
        
        if len(selected) > best_size:
            best_size = len(selected)
            best = selected
    
    return sorted(best)
