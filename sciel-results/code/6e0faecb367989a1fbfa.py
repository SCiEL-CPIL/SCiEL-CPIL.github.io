import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    N = 256
    
    def is_valid_subset(subset):
        s = list(subset)
        for i, a in enumerate(s):
            for j, b in enumerate(s):
                if i == j: continue
                for k, c in enumerate(s):
                    if i == k or j == k: continue
                    if (b * c) % a == 0:
                        return False
        return True
    
    current_set = set(range(129, N + 1))
    
    # Quick conflict heuristic: for each x, count pairs (y,z) in S\{x} with x|y*z
    def quick_conflicts(s):
        elems = list(s)
        n = len(elems)
        conflicts = {x: 0 for x in s}
        for i in range(n):
            a = elems[i]
            for j in range(n):
                if i == j: continue
                b = elems[j]
                g = math.gcd(a, b)
                d = a // g
                # Count multiples of d in s\{a,b}
                # For small d, many multiples; for large d, few
                if d <= 1:
                    # a|b, so a|b*c for all c
                    conflicts[a] += (n - 2)
                else:
                    # Count c in s\{a,b} with c % d == 0
                    count = 0
                    for k in range(n):
                        if k == i or k == j: continue
                        if elems[k] % d == 0:
                            count += 1
                    conflicts[a] += count
        return conflicts
    
    # Do a few rounds of removal
    for _ in range(10):
        if time.time() > deadline:
            break
        conflicts = quick_conflicts(current_set)
        if not conflicts:
            break
        max_c = max(conflicts.values())
        if max_c == 0:
            break
        max_elem = max(conflicts, key=lambda x: conflicts[x])
        current_set.remove(max_elem)
    
    # Add smaller numbers
    smaller = list(range(2, 129))
    smaller.sort(reverse=True)
    for num in smaller:
        if time.time() > deadline:
            break
        if num in current_set:
            continue
        if is_valid_subset(current_set | {num}):
            current_set.add(num)
    
    result = list(current_set)
    if not is_valid_subset(result):
        result = list(range(129, N + 1))
    return [int(x) for x in result]

