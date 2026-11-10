import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    # Greedy construction from largest to smallest
    # Maintain set of achievable subset sums of reciprocals
    # Prune sums > 1
    
    achievable_sums = {Fraction(0, 1)}
    current_set = []
    
    for x in range(24, 0, -1):
        if time.time() > deadline:
            break
            
        target = Fraction(1, x)
        
        if target in achievable_sums:
            continue
        
        current_set.append(x)
        
        # Update sums
        new_sums = set()
        for s in achievable_sums:
            ns = s + target
            if ns <= 1:
                new_sums.add(ns)
        achievable_sums.update(new_sums)
        
    if not current_set:
        return list(range(13, 25))
        
    return current_set
