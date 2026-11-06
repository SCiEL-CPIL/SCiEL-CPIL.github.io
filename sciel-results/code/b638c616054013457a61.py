"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def construct():
    deadline = time.time() + 80  # 80s budget, 10s safety margin
    
    def lcm(a, b):
        return a * b // math.gcd(a, b)
    
    def is_valid_addition(candidate, current_set, limit=512):
        for x in current_set:
            if lcm(candidate, x) > limit:
                return False
        return True
    
    def greedy_expand(initial_set, limit=512):
        """Greedy expansion: try to add elements that maintain validity."""
        current = set(initial_set)
        # Try all elements from 1 to 512, adding if valid
        for candidate in range(1, limit + 1):
            if time.time() > deadline:
                break
            if candidate not in current and is_valid_addition(candidate, current, limit):
                current.add(candidate)
        return current
    
    def get_divisors(n):
        divs = set()
        i = 1
        while i * i <= n:
            if n % i == 0:
                divs.add(i)
                divs.add(n // i)
            i += 1
        return divs
    
    # Highly composite numbers with many divisors
    hcn_candidates = [360, 420, 480, 504, 336, 240, 300, 360, 420, 480]
    # Also include numbers with many divisors
    for m in range(1, 513):
        if m not in hcn_candidates:
            hcn_candidates.append(m)
    
    best_set = set()
    best_size = 0
    
    # Sort candidates by number of divisors (descending) for efficiency
    def count_divisors(n):
        count = 0
        i = 1
        while i * i <= n:
            if n % i == 0:
                count += 1
                if i != n // i:
                    count += 1
            i += 1
        return count
    
    hcn_candidates = list(set(hcn_candidates))
    hcn_candidates.sort(key=lambda x: -count_divisors(x))
    
    for m in hcn_candidates:
        if time.time() > deadline:
            break
        
        initial = get_divisors(m)
        # Ensure all elements are in [1, 512]
        initial = {x for x in initial if 1 <= x <= 512}
        
        if not initial:
            continue
        
        # Verify initial set is valid
        valid = True
        elems = list(initial)
        for i in range(len(elems)):
            for j in range(i + 1, len(elems)):
                if lcm(elems[i], elems[j]) > 512:
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            continue
        
        # Try greedy expansion
        expanded = greedy_expand(initial)
        
        if len(expanded) > best_size:
            best_size = len(expanded)
            best_set = expanded
    
    # Fallback: at least return {1}
    if not best_set:
        best_set = {1}
    
    return list(best_set)
# EVOLVE-BLOCK-END
