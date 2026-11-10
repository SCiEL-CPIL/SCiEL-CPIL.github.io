import time
from math import gcd
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
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
    
    # Precompute compatibility bitsets
    # compat[i] is a bitset where bit j is 1 if lcm(i, j+1) <= N
    # We use 0-indexed: element x corresponds to bit (x-1)
    compat = [0] * (N + 1)
    for i in range(1, N + 1):
        mask = 0
        for j in range(1, N + 1):
            if i // gcd(i, j) * j <= N:
                mask |= (1 << (j - 1))
        compat[i] = mask
    
    # Helper: check if adding x to a set keeps it valid using bitsets
    def can_add(x, s_mask):
        # x is compatible with all elements in s if compat[x] has bits set for all elements in s
        # Check if all bits in s_mask are also in compat[x]
        return (compat[x] & s_mask) == s_mask
    
    # Faster: precompute bitset for the set and update incrementally
    # We'll track the set as a bitset for faster operations
    
    # Greedy expansion using bitsets - improved version
    def greedy_expand(s):
        s_mask = 0
        for x in s:
            s_mask |= (1 << (x - 1))
        
        all_mask = (1 << N) - 1
        
        changed = True
        while changed and time.time() < deadline:
            changed = False
            # Find all candidates not in s
            candidates_mask = all_mask & ~s_mask
            
            if not candidates_mask:
                break
                
            # Try candidates in order of most compatible first (greedy heuristic)
            # Count how many existing elements each candidate is compatible with
            candidate_list = []
            m = candidates_mask
            pos = 0
            while m:
                if m & 1:
                    x = pos + 1
                    # Count compatibility with current set
                    compat_count = bin(compat[x] & s_mask).count('1')
                    candidate_list.append((compat_count, x))
                m >>= 1
                pos += 1
            
            # Sort by compatibility count (descending) - prefer elements that fit well
            candidate_list.sort(key=lambda t: -t[0])
            
            # Try to add compatible elements
            for _, x in candidate_list:
                if time.time() >= deadline:
                    break
                if (compat[x] & s_mask) == s_mask:
                    s.add(x)
                    s_mask |= (1 << (x - 1))
                    changed = True
        
        return s
    
    current_set = greedy_expand(set(best_set))
    current_size = len(current_set)
    
    # Local search: try removing one element and adding multiple new ones
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for remove_candidate in list(current_set):
            if time.time() > deadline:
                break
            
            trial_set = current_set - {remove_candidate}
            
            # Try to greedily add elements back
            expanded = greedy_expand(set(trial_set))
            
            if len(expanded) > current_size:
                current_set = expanded
                current_size = len(current_set)
                improved = True
                break
    
    # Random restarts with better initialization
    num_restarts = 100
    for _ in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Start with a random subset of the best divisor set, then expand
        random_set = set()
        for x in best_set:
            if random.random() < 0.5:
                random_set.add(x)
        
        # Also try adding some random small numbers
        for x in range(1, 32):
            if random.random() < 0.2 and x not in random_set:
                s_mask = 0
                for y in random_set:
                    s_mask |= (1 << (y - 1))
                if can_add(x, s_mask):
                    random_set.add(x)
        
        expanded = greedy_expand(random_set)
        
        if len(expanded) > current_size:
            current_set = expanded
            current_size = len(current_set)
    
    # Final local search pass
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for remove_candidate in list(current_set):
            if time.time() > deadline:
                break
            
            trial_set = current_set - {remove_candidate}
            
            # Try to greedily add elements back
            expanded = greedy_expand(set(trial_set))
            
            if len(expanded) > current_size:
                current_set = expanded
                current_size = len(current_set)
                improved = True
                break
    
    final_expanded = greedy_expand(set(current_set))
    
    return list(final_expanded)
