import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80
    
    selected = []
    
    for x in range(192, 0, -1):
        if time.time() > deadline:
            break
        
        valid = True
        m = len(selected)
        
        # Check if x divides sum of any triple in selected
        for triple in combinations(selected, 3):
            if sum(triple) % x == 0:
                valid = False
                break
        
        # Check if any existing a divides sum of any pair in selected\{a} plus x
        if valid:
            for a in selected:
                others = [v for v in selected if v != a]
                for pair in combinations(others, 2):
                    if (pair[0] + pair[1] + x) % a == 0:
                        valid = False
                        break
                if not valid:
                    break
        
        if valid:
            selected.append(x)
    
    return selected
