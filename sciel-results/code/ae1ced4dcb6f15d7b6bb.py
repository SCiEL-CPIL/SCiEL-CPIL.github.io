import math

def construct():
    N = 512
    nums = list(range(1, N))
    weights = {a: 1.0 / (N - a) for a in nums}
    
    # Sort by weight descending
    sorted_nums = sorted(nums, key=lambda a: -weights[a])
    
    available = set(nums)
    selected = []
    
    for a in sorted_nums:
        if a not in available:
            continue
            
        # Find conflicts in available
        conflicts = set()
        for b in available:
            if b == a:
                continue
            if math.gcd(a, b) > 1:
                conflicts.add(b)
                
        loss = sum(weights[b] for b in conflicts)
        
        if weights[a] > loss:
            selected.append(a)
            available.remove(a)
            available.difference_update(conflicts)
        else:
            # We skip 'a', so it is no longer available for selection
            # But its conflicts remain available
            available.remove(a)
            
    return selected
