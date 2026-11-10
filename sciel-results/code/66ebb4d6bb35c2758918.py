import time
from math import gcd
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin before 60s limit
    
    N = 512
    
    # Precompute divisors for all L up to N
    def get_divisors(n):
        divs = set()
        for i in range(1, int(n**0.5) + 1):
            if n % i == 0:
                divs.add(i)
                divs.add(n // i)
        return divs
    
    # Find the best divisor set (all divisors of some L form a valid clique)
    best_set = set()
    best_size = 0
    for L in range(1, N + 1):
        divs = get_divisors(L)
        if len(divs) > best_size:
            best_size = len(divs)
            best_set = divs
    
    # Helper: check if adding x to a set keeps it valid
    def can_add(x, s):
        for y in s:
            if x // gcd(x, y) * y > N:
                return False
        return True
    
    # Greedy expansion: repeatedly try to add any compatible element
    def greedy_expand(s):
        changed = True
        while changed and time.time() < deadline:
            changed = False
            candidates = [x for x in range(1, N + 1) if x not in s]
            random.shuffle(candidates)
            for x in candidates:
                if can_add(x, s):
                    s.add(x)
                    changed = True
                    break
        return s
    
    current_set = greedy_expand(set(best_set))
    current_size = len(current_set)
    
    # Local search: try removing one element and adding multiple new ones
    # This can escape local optima where a single element blocks many others
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try each element for removal
        for remove_candidate in list(current_set):
            if time.time() > deadline:
                break
            
            trial_set = current_set - {remove_candidate}
            original_size = len(trial_set)
            
            # Try to greedily add elements back
            expanded = greedy_expand(set(trial_set))
            
            if len(expanded) > current_size:
                current_set = expanded
                current_size = len(current_set)
                improved = True
                break
    
    # Random restarts to find potentially better starting points
    num_restarts = 50
    for _ in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Build a random valid set
        random_set = set()
        # Start with some small numbers that are likely compatible
        for x in range(1, 64):
            if random.random() < 0.3 and can_add(x, random_set):
                random_set.add(x)
        
        # Greedy expand from this random start
        expanded = greedy_expand(random_set)
        
        if len(expanded) > current_size:
            current_set = expanded
            current_size = len(current_set)
    
    # Final attempt: try to add any remaining compatible elements
    final_expanded = greedy_expand(set(current_set))
    
    return list(final_expanded)
