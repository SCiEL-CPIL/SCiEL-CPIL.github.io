import itertools
import time

def construct():
    # Improved candidate: {5, 7, 9, 10, ..., 24}
    # Size 18.
    # Excludes 1, 2, 3, 4, 6, 8.
    candidate = [5, 7] + list(range(9, 25))
    
    # Quick verification
    recips = {i: 1.0/i for i in candidate}
    valid = True
    for x in candidate:
        target = recips[x]
        others = [y for y in candidate if y != x]
        for r in range(1, len(others)+1):
            for subset in itertools.combinations(others, r):
                if abs(sum(recips[y] for y in subset) - target) < 1e-10:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
            
    if valid:
        return candidate
    else:
        # Fallback
        return [7] + list(range(9, 25))
