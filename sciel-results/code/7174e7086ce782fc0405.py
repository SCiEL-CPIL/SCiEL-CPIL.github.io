import time
import random

P = {'n': 128}

def construct():
    n = P['n']
    deadline = time.time() + 58  # Safety margin before 60s limit
    
    # Precompute conflict hyperedges for the 3-uniform hypergraph.
    # A triple (a, b, c) is forbidden if a, b, c are distinct and
    # a+b, a+c, b+c are all in the domain 1..n.
    # We represent the set S as a bitmask (Python int) for bit-parallel operations.
    
    # Precompute for each pair (a, b) with a < b, the set of c > b such that
    # the triple (a, b, c) is "potentially forbidden" (i.e., all pair sums are in 1..n).
    # Actually, the condition depends on the set S.
    # Forbidden triple: a, b, c in S, and a+b in S, a+c in S, b+c in S.
    
    # Let's precompute all "structurally possible" forbidden triples.
    # A triple {a, b, c} with a < b < c is structurally possible if:
    # a+b <= n, a+c <= n, b+c <= n.
    # If such a triple is in S, and its sums are in S, it's invalid.
    
    # To check validity of a set S (bitmask), we need to ensure no such triple is fully contained in S.
    # This is a 3-SAT-like constraint satisfaction, but for maximization.
    
    # Greedy approach:
    # 1. Start with a large valid set.
    #    The set {65, ..., 128} is valid because any sum of two elements is >= 130 > 128.
    #    Size = 64.
    # 2. Try to add elements from 1..64.
    #    When adding x, we must check if it creates a forbidden triple with existing elements.
    
    # Efficient validity check for adding x to S:
    # Let S be the current set (bitmask).
    # We want to check if S ∪ {x} is valid.
    # A new forbidden triple must involve x.
    # Case 1: x is one of the elements {a, b, c}.
    #   WLOG c = x. Then a, b < x.
    #   We need a, b in S, a != b, and a+b in S, a+x in S, b+x in S.
    #   Note: a+x and b+x are > x, so they are in the "upper" part of the set potentially.
    # Case 2: x is one of the sums {a+b, a+c, b+c}.
    #   WLOG a+b = x. Then a, b < x.
    #   We need a, b, c in S (c != a, b), and a+c in S, b+c in S.
    #   Note: a+c and b+c are > c? Not necessarily. But a+c and b+c must be in S.
    
    # Let's implement a fast check using bitmasks.
    # S is an integer where bit i is 1 if i+1 is in S? Or bit i if i is in S?
    # Let's use bit i for value i. So bit 1 is value 1, ..., bit 128 is value 128.
    # This requires an integer with 129 bits. Python ints handle this fine.
    
    def get_bitmask(lst):
        mask = 0
        for x in lst:
            mask |= (1 << x)
        return mask
    
    def has_bit(mask, x):
        return (mask >> x) & 1
    
    def set_bit(mask, x):
        return mask | (1 << x)
    
    def clear_bit(mask, x):
        return mask & ~(1 << x)
    
    # Precompute structural triples:
    # For each pair (a, b) with a < b, list all c > b such that a+c <= n and b+c <= n.
    # Actually, we can just iterate c.
    
    # To speed up the check for Case 1:
    # Given x, we need to find if there exist a, b in S, a < b < x, such that:
    # (a+b) in S AND (a+x) in S AND (b+x) in S.
    # Let A_x = {a in S | a < x and (a+x) in S}.
    # For each a in A_x, we need b in S, a < b < x, such that (a+b) in S and (b+x) in S.
    # Let B_a = {b in S | a < b < x and (b+x) in S}.
    # We need intersection of B_a and {b | a+b in S} to be non-empty.
    
    # To speed up Case 2:
    # Given x, we need to find if there exist a, b in S, a < b, a+b=x, and c in S, c!=a,b,
    # such that (a+c) in S and (b+c) in S.
    # For each pair (a, x-a) in S, check if there is c.
    
    # Let's try a randomized greedy with bitmasks.
    
    base_set = set(range(65, n + 1))
    base_mask = get_bitmask(base_set)
    
    candidates = list(range(1, 65))
    
    best_size = len(base_set)
    best_mask = base_mask
    
    def is_valid_addition(mask, x):
        """Check if adding x to mask keeps it valid."""
        if has_bit(mask, x):
            return True
        
        # Case 1: x is the largest element in the triple {a, b, x}
        # Need a, b in mask, a < b < x, such that:
        # (a+b) in mask AND (a+x) in mask AND (b+x) in mask
        
        # Collect a in mask such that a < x and (a+x) in mask
        # Note: a+x > x, so a+x is not x.
        valid_a = []
        for a in range(1, x):
            if has_bit(mask, a) and has_bit(mask, a + x):
                valid_a.append(a)
        
        for a in valid_a:
            # Need b in mask, a < b < x, such that (a+b) in mask and (b+x) in mask
            for b in range(a + 1, x):
                if has_bit(mask, b) and has_bit(mask, a + b) and has_bit(mask, b + x):
                    return False
        
        # Case 2: x is a sum, say x = a+b
        # Need a, b in mask, a < b, a+b=x, and c in mask, c != a, b,
        # such that (a+c) in mask and (b+c) in mask.
        
        for a in range(1, x // 2 + 1):
            b = x - a
            if b <= a:
                continue
            if has_bit(mask, a) and has_bit(mask, b):
                # Check for c
                for c in range(1, n + 1):
                    if c == a or c == b:
                        continue
                    if has_bit(mask, c):
                        if has_bit(mask, a + c) and has_bit(mask, b + c):
                            return False
        
        return True
    
    # Optimization: Precompute for each x, the list of (a, b) pairs that sum to x?
    # And for each x, the list of a such that a+x <= n?
    
    # Let's run multiple random restarts.
    
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        
        current_mask = base_mask
        current_size = len(base_set)
        
        # Randomize candidate order
        c = candidates.copy()
        random.shuffle(c)
        
        # Greedy add
        for x in c:
            if has_bit(current_mask, x):
                continue
            if is_valid_addition(current_mask, x):
                current_mask = set_bit(current_mask, x)
                current_size += 1
        
        if current_size > best_size:
            best_size = current_size
            best_mask = current_mask
        
        # Local Search: Try to swap
        # If we have time, try to remove an element and add two?
        # Or just try to add more?
        # The greedy might get stuck.
        
        # Simple local search: try to add any remaining candidate
        added_any = True
        while added_any and time.time() < deadline - 2:
            added_any = False
            for x in candidates:
                if has_bit(current_mask, x):
                    continue
                if is_valid_addition(current_mask, x):
                    current_mask = set_bit(current_mask, x)
                    current_size += 1
                    added_any = True
                    if current_size > best_size:
                        best_size = current_size
                        best_mask = current_mask
        
        # If stuck, try a swap: remove y, add z
        # This is expensive. Let's just do more restarts.
        
    # Convert best_mask to list
    result = []
    for i in range(1, n + 1):
        if has_bit(best_mask, i):
            result.append(i)
    
    return result

