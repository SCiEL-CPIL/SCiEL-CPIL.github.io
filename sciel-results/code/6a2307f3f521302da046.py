def construct():
    # We want to maximize the size of a subset of [1, 100] 
    # such that no subset sums to 200.
    # We use a greedy approach: iterate from 100 down to 1.
    # Maintain a set of all possible subset sums of the current chosen set, 
    # limited to sums <= 200.
    # If adding element i would create a sum of 200 (i.e., 200-i is already achievable), 
    # we skip i. Otherwise, we add i.
    
    possible_sums = {0}
    chosen = []
    
    for i in range(100, 0, -1):
        target = 200 - i
        if target in possible_sums:
            # Adding i would create a subset sum of 200
            continue
        else:
            # Safe to add i
            chosen.append(i)
            # Update possible_sums
            # New sums are s + i for s in possible_sums
            # We only care about sums <= 200
            new_sums = set()
            for s in possible_sums:
                ns = s + i
                if ns <= 200:
                    new_sums.add(ns)
            possible_sums.update(new_sums)
            
    return chosen
