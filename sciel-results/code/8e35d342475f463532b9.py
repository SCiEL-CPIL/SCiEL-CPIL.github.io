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
    def can_add(x, s):
        # x is compatible with all elements in s if compat[x] has bits set for all elements in s
        # Build bitset for s
        s_mask = 0
        for y in s:
            s_mask |= (1 << (y - 1))
        # Check if all bits in s_mask are also in compat[x]
        return (compat[x] & s_mask) == s_mask
    
    # Faster: precompute bitset for the set and update incrementally
    # We'll track the set as a bitset for faster operations
    
    # Greedy expansion using bitsets
    def greedy_expand(s):
        s_mask = 0
        for x in s:
            s_mask |= (1 << (x - 1))
        
        changed = True
        while changed and time.time() < deadline:
            changed = False
            # Find all candidates not in s
            all_mask = (1 << N) - 1
            candidates_mask = all_mask & ~s_mask
            
            # Try candidates in random order
            while candidates_mask and time.time() < deadline:
                # Pick a random bit from candidates_mask
                # Convert to list of indices for random selection
                candidates = []
                m = candidates_mask
                while m:
                    if m & 1:
                        candidates.append(0)
                    m >>= 1
                # Actually, let's just iterate through all candidates
                # But that's slow. Let's use a different approach.
                
                # Pick a random candidate
                # We need to pick a random set bit from candidates_mask
                # Use Python's random with the bit positions
                bits = []
                m = candidates_mask
                pos = 0
                while m:
                    if m & 1:
                        bits.append(pos)
                    m >>= 1
                    pos += 1
                
                if not bits:
                    break
                    
                idx = random.choice(bits)
                x = idx + 1  # Convert back to 1-indexed
                
                if can_add(x, s):
                    s.add(x)
                    s_mask |= (1 << (x - 1))
                    candidates_mask &= ~(1 << (x - 1))
                    changed = True
                else:
                    # Remove this candidate from consideration
                    candidates_mask &= ~(1 << idx)
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
    
    # Random restarts
    num_restarts = 50
    for _ in range(num_restarts):
        if time.time() > deadline:
            break
        
        random_set = set()
        for x in range(1, 64):
            if random.random() < 0.3 and can_add(x, random_set):
                random_set.add(x)
        
        expanded = greedy_expand(random_set)
        
        if len(expanded) > current_size:
            current_set = expanded
            current_size = len(current_set)
    
    final_expanded = greedy_expand(set(current_set))
    
    return list(final_expanded)
