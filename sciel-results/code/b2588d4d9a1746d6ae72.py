import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    n = 512
    
    # Key insight: if S is the set of divisors of M where M <= 512,
    # then for any a,b in S, lcm(a,b) | M <= 512, so the condition is satisfied.
    # Find M <= 512 with the most divisors.
    
    best_M = 1
    best_div_count = 0
    best_divs = [1]
    
    for M in range(1, n + 1):
        divs = [d for d in range(1, n + 1) if M % d == 0]
        if len(divs) > best_div_count:
            best_div_count = len(divs)
            best_M = M
            best_divs = divs[:]
    
    # Start with divisors of best_M
    current_set = set(best_divs)
    
    # Try to greedily add elements that are compatible with all elements in current_set
    # An element x is compatible with set S if lcm(x, s) <= 512 for all s in S
    
    remaining = [x for x in range(1, n + 1) if x not in current_set]
    
    # Sort by lcm with the largest element in current set (proxy for compatibility)
    max_in_set = max(current_set)
    remaining.sort(key=lambda x: x * max_in_set // gcd(x, max_in_set))
    
    # Repeatedly try to add the most compatible element
    changed = True
    while changed and time.time() < deadline:
        changed = False
        new_remaining = []
        for x in remaining:
            # Check if x is compatible with all elements in current_set
            compatible = True
            for s in current_set:
                if x * s // gcd(x, s) > 512:
                    compatible = False
                    break
            if compatible:
                current_set.add(x)
                changed = True
                max_in_set = max(current_set)
            else:
                new_remaining.append(x)
        remaining = new_remaining
        if changed:
            remaining.sort(key=lambda x: x * max_in_set // gcd(x, max_in_set))
    
    result = list(current_set)
    
    if not result:
        return [1]
    
    return result
