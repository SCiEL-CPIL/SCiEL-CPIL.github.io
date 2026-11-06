import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin
    
    n = 256
    # S is the set of selected numbers
    # We build it by iterating from n down to 1
    # S will be a sorted list for efficient access if needed, but set is good for membership
    S = set()
    
    # We iterate v from n down to 1
    # For each v, we check if adding v to S violates the condition.
    # Condition: For all a < b <= c in S_new, a does not divide b+c.
    # Since we add v in decreasing order, v is the smallest element in S_new.
    # So v can only be 'a'.
    # We must check: For all b, c in S such that b <= c, v does NOT divide b+c.
    
    # Optimization:
    # We can store S in a sorted list to iterate pairs?
    # Or just nested loops over S.
    # Size of S is at most 256.
    # 256^2 / 2 = 32768 pairs.
    # 256 iterations * 32768 checks = 8.4 million checks.
    # Each check is O(1) modulo.
    # This is very fast in Python if implemented carefully.
    
    S_list = []  # Keep a list for iteration
    S_set = set() # For O(1) membership if needed, but we only need pairs from S
    
    # To speed up, we can precompute sums? No, S changes.
    
    for v in range(n, 0, -1):
        if time.time() > deadline:
            break
            
        valid = True
        # Check all pairs b, c in S_list
        # b, c are > v
        L = len(S_list)
        for i in range(L):
            b = S_list[i]
            # c must be >= b
            for j in range(i, L):
                c = S_list[j]
                # b, c are in S, so b > v, c > v
                # Check if v divides b+c
                if (b + c) % v == 0:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            S_list.append(v)
            S_set.add(v)
            
    # The result must be a list (or set) of integers.
    # The task says "construct() -> a nonempty subset".
    # Usually a list is fine.
    
    # If S is empty (shouldn't happen, 256 is always valid), return [256]
    if not S_set:
        return [256]
        
    return sorted(S_set)

# EVOLVE-BLOCK-END
