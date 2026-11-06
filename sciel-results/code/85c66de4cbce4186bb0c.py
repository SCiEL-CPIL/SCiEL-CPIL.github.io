import time

def construct():
    N = 51975
    start_time = time.time()
    deadline = start_time + 110  # 10 second margin
    
    # Find all divisors of N greater than 1
    divisors = [i for i in range(2, N + 1) if N % i == 0]
    
    # Precompute masks for each (m, r)
    masks = {}
    for m in divisors:
        masks[m] = [0] * m
        for r in range(m):
            mask = 0
            for x in range(r, N, m):
                mask |= (1 << x)
            masks[m][r] = mask
            
    full_mask = (1 << N) - 1
    
    def popcount(x):
        return bin(x).count('1')
        
    # Greedy Phase
    available_moduli = list(divisors)
    uncovered = full_mask
    current_classes = []
    
    while available_moduli:
        if time.time() > deadline:
            break
            
        best_gain = -1
        best_m = None
        best_r = None
        
        for m in available_moduli:
            for r in range(m):
                gain = popcount(uncovered & masks[m][r])
                if gain > best_gain:
                    best_gain = gain
                    best_m = m
                    best_r = r
                    
        if best_m is None:
            break
            
        current_classes.append([best_m, best_r])
        uncovered &= ~masks[best_m][best_r]
        available_moduli.remove(best_m)
        
    # If we ran out of time in greedy, we still need to return something valid.
    # The current_classes list contains valid distinct moduli.
    # We fill in the rest arbitrarily if time is tight, or just return what we have.
    # The problem doesn't require using ALL moduli, just distinct ones.
    # However, using more moduli can only help or be neutral.
    # If we stopped early, we can just return the current list.
    
    # Local Search Phase
    # Only run if we have time
    if time.time() < deadline - 5:
        class_dict = {m: r for m, r in current_classes}
        
        def calc_uncovered(cd):
            unc = full_mask
            for m, r in cd.items():
                unc &= ~masks[m][r]
            return unc
            
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for m in divisors:
                if m not in class_dict:
                    continue
                current_r = class_dict[m]
                current_uncovered = calc_uncovered(class_dict)
                current_score = popcount(current_uncovered)
                
                best_r_for_m = current_r
                best_score = current_score
                
                for r in range(m):
                    if r == current_r:
                        continue
                    class_dict[m] = r
                    new_uncovered = calc_uncovered(class_dict)
                    new_score = popcount(new_uncovered)
                    
                    if new_score < best_score:
                        best_score = new_score
                        best_r_for_m = r
                        
                if best_r_for_m != current_r:
                    class_dict[m] = best_r_for_m
                    improved = True
                else:
                    class_dict[m] = current_r
                    
        result = [[m, r] for m, r in class_dict.items()]
    else:
        result = current_classes
        
    return result
