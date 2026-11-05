import random

def construct():
    n = 4096
    best_set = []
    best_len = 0
    
    # Number of trials to balance time and improvement
    # 60 seconds limit. Let's try 200 trials.
    num_trials = 200
    
    for _ in range(num_trials):
        candidates = list(range(1, n + 1))
        random.shuffle(candidates)
        
        current_set = []
        used_sums = set()
        
        for x in candidates:
            # Check if x can be added
            # New sums involving x
            # 1. 3x
            # 2. 2x + s for s in current_set, s != x
            # 3. x + s1 + s2 for s1 < s2 in current_set
            
            # Optimization: if x is small, it might conflict with small sums.
            # We just check.
            
            new_sums = []
            new_sums.append(3 * x)
            
            # s in current_set
            # Note: x is not in current_set yet, so s != x is always true
            for s in current_set:
                new_sums.append(2 * x + s)
                
            # Pairs in current_set
            # We need s1 < s2 to avoid duplicates in the generation logic?
            # Actually, the triple {x, s1, s2} is the same as {x, s2, s1}.
            # So we only generate each pair once.
            cs_len = len(current_set)
            for i in range(cs_len):
                s1 = current_set[i]
                for j in range(i + 1, cs_len):
                    s2 = current_set[j]
                    new_sums.append(x + s1 + s2)
                    
            # Check for internal duplicates
            # Convert to set to check uniqueness
            new_sums_set = set(new_sums)
            if len(new_sums_set) != len(new_sums):
                continue
                
            # Check for conflict with existing sums
            if used_sums.intersection(new_sums_set):
                continue
                
            # Valid, add x
            current_set.append(x)
            used_sums.update(new_sums_set)
            
        if len(current_set) > best_len:
            best_len = len(current_set)
            best_set = current_set
            
    return best_set
