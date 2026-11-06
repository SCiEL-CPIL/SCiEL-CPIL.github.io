def construct():
    import time
    from math import gcd
    
    deadline = time.time() + 85
    
    N = 256
    nums = list(range(2, N+1))
    n = len(nums)
    
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    conflict = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if lcm(nums[i], nums[j]) <= N:
                conflict[i].add(j)
                conflict[j].add(i)
    
    T = 18
    small_idx = [i for i in range(n) if nums[i] <= T]
    large_idx = [i for i in range(n) if nums[i] > T]
    
    # Precompute small-large conflicts
    sl_conf = {}
    for si in small_idx:
        sl_conf[si] = frozenset(lj for lj in large_idx if lj in conflict[si])
    
    def best_large(incompat_large):
        compat = [j for j in large_idx if j not in incompat_large]
        compat.sort(key=lambda j: -1.0/nums[j])
        sel = []
        for j in compat:
            if all(j not in conflict[k] for k in sel):
                sel.append(j)
        return sel
    
    best_val = 0.0
    best_small = set()
    best_large_set = set()
    
    m = len(small_idx)
    for mask in range(1 << m):
        if time.time() > deadline:
            break
        sel_small = set()
        for k in range(m):
            if mask & (1 << k):
                sel_small.add(small_idx[k])
        # Check mutual compatibility
        arr = list(sel_small)
        valid = True
        for i in range(len(arr)):
            for j in range(i+1, len(arr)):
                if arr[j] in conflict[arr[i]]:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            continue
        # Compute incompatible large
        incompat = set()
        for s in sel_small:
            incompat |= sl_conf[s]
        large_sel = best_large(incompat)
        val = sum(1.0/nums[i] for i in sel_small) + sum(1.0/nums[j] for j in large_sel)
        if val > best_val:
            best_val = val
            best_small = sel_small
            best_large_set = set(large_sel)
    
    result = sorted(nums[i] for i in (best_small | best_large_set))
    if not result:
        result = [2]
    return result
