import time
from itertools import combinations

def construct():
    start = time.time()
    deadline = start + 80  # safety margin
    
    # Greedy: try numbers from largest to smallest
    # A larger number is "safer" because it has fewer multiples in the possible sum range
    selected = []
    
    for x in range(192, 0, -1):
        if time.time() > deadline:
            break
        
        # Check if adding x is valid
        valid = True
        # For each existing element a in selected, check no triple of OTHER elements (including x) sums to multiple of a
        # For x itself, check no triple of existing elements sums to multiple of x
        
        # Check x as the divisor: no 3 from selected should sum to multiple of x
        if len(selected) >= 3:
            for t in combinations(selected, 3):
                s = sum(t)
                if s % x == 0:
                    valid = False
                    break
        
        if valid:
            # Check each existing a: no triple from (selected + [x]) excluding a sums to multiple of a
            for i, a in enumerate(selected):
                if time.time() > deadline:
                    valid = False
                    break
                # Triples from selected+[x] that don't include a
                # This is: triples from selected (excluding a) that include x, or triples from selected (excluding a) not including x
                # Triples not including x: already checked before adding x, so they're fine
                # New triples include x and 2 from selected (excluding a)
                others = [v for j, v in enumerate(selected) if j != i]
                for j in range(len(others)):
                    for k in range(j+1, len(others)):
                        s = others[j] + others[k] + x
                        if s % a == 0:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
        
        if valid:
            selected.append(x)
    
    return selected
