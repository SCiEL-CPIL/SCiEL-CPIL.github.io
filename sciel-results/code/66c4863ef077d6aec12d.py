def construct():
    import time
    from math import gcd
    
    start_time = time.time()
    deadline = start_time + 80
    
    nums = list(range(2, 257))
    n = len(nums)
    
    # Precompute conflicts as sets
    conflict = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            ai, aj = nums[i], nums[j]
            g = gcd(ai, aj)
            if ai * aj // g <= 256:
                conflict[i].add(j)
                conflict[j].add(i)
    
    small_indices = list(range(0, 15))  # nums 2-16
    large_indices = list(range(15, n))  # nums 17-256
    
    best_sum = 0.0
    best_set = set()
    
    cache = {}
    
    def solve_large(compatible_frozenset):
        if compatible_frozenset in cache:
            return cache[compatible_frozenset]
        
        compatible_large = set(compatible_frozenset)
        if not compatible_large:
            cache[compatible_frozenset] = (set(), 0.0)
            return set(), 0.0
        
        # Greedy by weight
        candidates = sorted(compatible_large, key=lambda x: -1.0/nums[x])
        selected = set()
        sel_sum = 0.0
        for j in candidates:
            if not (conflict[j] & selected):
                selected.add(j)
                sel_sum += 1.0/nums[j]
        
        # Local search (limited iterations)
        for _ in range(10):
            if time.time() > deadline:
                break
            improved = False
            # Try adding
            for j in compatible_large:
                if j not in selected and not (conflict[j] & selected):
                    selected.add(j)
                    sel_sum += 1.0/nums[j]
                    improved = True
            # Try 1-for-1 swaps
            for i in list(selected):
                s_without = selected - {i}
                base_sum = sel_sum - 1.0/nums[i]
                for j in compatible_large:
                    if j not in s_without and not (conflict[j] & s_without):
                        new_sum = base_sum + 1.0/nums[j]
                        if new_sum > sel_sum + 1e-12:
                            selected = s_without | {j}
                            sel_sum = new_sum
                            improved = True
                            break
            if not improved:
                break
        
        result = (selected, sel_sum)
        cache[compatible_frozenset] = result
        return result
    
    # Enumerate subsets of small numbers
    for mask in range(1 << len(small_indices)):
        if time.time() > deadline:
            break
        
        s_small = set()
        for i in range(len(small_indices)):
            if mask & (1 << i):
                s_small.add(small_indices[i])
        
        # Check independence
        valid = True
        for i in s_small:
            if conflict[i] & s_small:
                valid = False
                break
        if not valid:
            continue
        
        # Find compatible large numbers
        compatible_large = frozenset(j for j in large_indices if not (conflict[j] & s_small))
        
        large_set, large_sum = solve_large(compatible_large)
        small_sum = sum(1.0/nums[x] for x in s_small)
        total_sum = small_sum + large_sum
        
        if total_sum > best_sum:
            best_sum = total_sum
            best_set = s_small | large_set
    
    result = sorted(nums[i] for i in best_set)
    if not result:
        return [2]
    return result
