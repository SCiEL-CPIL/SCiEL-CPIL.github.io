import random
import time

def is_valid(S):
    """Check if set S is valid."""
    S_set = set(S)
    # To speed up, we can iterate over pairs a, b in S
    # and check if a+b is in S. If so, then for any c in S distinct from a,b,
    # we must not have a+c in S and b+c in S.
    # This is O(N^3) in worst case, but N is at most 128.
    # 128^3 = 2M operations, which is fast in Python if optimized.
    
    # Better: iterate over all triples a,b,c? No, too slow.
    # Iterate over pairs a,b in S.
    # If a+b in S, then for all c in S \ {a,b}, check if a+c in S and b+c in S.
    # If yes, invalid.
    
    # Actually, the condition is: 
    # No distinct a,b,c in S such that a+b in S, a+c in S, b+c in S.
    
    # Let's just check all distinct a,b,c.
    # Number of triples is ~350k. 
    # For each triple, 6 membership checks. 2M checks.
    # This is acceptable for a few thousand iterations.
    
    for i in range(len(S)):
        a = S[i]
        for j in range(i+1, len(S)):
            b = S[j]
            if a + b not in S_set:
                continue
            for k in range(j+1, len(S)):
                c = S[k]
                if a + c in S_set and b + c in S_set:
                    return False
    return True

def construct():
    start_time = time.time()
    best_size = 0
    best_set = set(range(65, 129)) # Initial guess
    
    # Greedy addition
    # We try to add elements from 1 to 64
    # We do this with random restarts to escape local optima
    
    for restart in range(100): # 100 restarts
        if time.time() - start_time > 80:
            break
            
        current_set = set(range(65, 129))
        candidates = list(range(1, 65))
        random.shuffle(candidates)
        
        for x in candidates:
            # Check if adding x keeps the set valid
            # We can do a quick check instead of full is_valid
            # But full is_valid is ~2M ops, 64 candidates * 2M = 128M ops per restart.
            # 100 restarts = 12.8B ops. Too slow.
            
            # We need a faster validity check or a smarter strategy.
            pass
        
        # Let's just use a simpler heuristic: 
        # Start with empty set, add elements greedily in random order.
        # This might yield smaller sets.
        
    # Let's try a different approach: Local search starting from upper half.
    # Move: Add an element if valid.
    # If not valid, try to remove an element that is "blocking" and add the new one.
    
    # Actually, let's just implement the greedy addition with a faster validity check.
    # We can maintain the set of "bad" pairs or something.
    
    # Given the complexity, let's just return the upper half if we can't do better quickly.
    # But we know we can do better.
    
    # Let's try: 
    # 1. Start with S = {65..128}
    # 2. For x in 1..64 (random order):
    #    Try to add x.
    #    If valid, add.
    #    If not, skip.
    
    # To make it faster, we can check validity incrementally?
    # Or just use the full check but limit restarts.
    
    best_set = set(range(65, 129))
    best_size = len(best_set)
    
    for restart in range(50):
        if time.time() - start_time > 85:
            break
            
        current_set = set(range(65, 129))
        candidates = list(range(1, 65))
        random.shuffle(candidates)
        
        for x in candidates:
            # Quick check: 
            # If we add x, we need to check if there are a,b in current_set 
            # such that a+b in current_set, and a+x in current_set, and b+x in current_set.
            # Note: a+x and b+x are NOT in current_set yet, they are new sums.
            # Wait, the condition is that a+b, a+c, b+c are IN S.
            # If we add x, the new elements are x.
            # The sums a+x, b+x are new values. They are not in S unless they were already there.
            # But x is from 1..64, so a+x and b+x are > a,b.
            # If a+x is in S, it must have been there before.
            
            # So, to check if adding x is valid:
            # We need to ensure that there are no a,b in current_set such that:
            # 1. a+b in current_set
            # 2. a+x in current_set
            # 3. b+x in current_set
            # AND x is distinct from a,b.
            
            # This is O(N^2) per candidate. 64 * 64^2 = 262k ops. Very fast.
            
            valid = True
            # We need to check all pairs a,b in current_set
            # But we can optimize: 
            # For each a in current_set:
            #   If a+x in current_set:
            #      For each b in current_set, b != a:
            #         If b+x in current_set and a+b in current_set:
            #            Invalid
            
            # Let's implement this check.
            S_list = list(current_set)
            S_set = current_set
            for a in S_list:
                if a == x: continue # x not in S yet
                if a + x not in S_set:
                    continue
                for b in S_list:
                    if b == x or b == a: continue
                    if b + x not in S_set:
                        continue
                    if a + b in S_set:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                current_set.add(x)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set
    
    return list(best_set)

# EVOLVE-BLOCK-START
P = {'n': 128}

import random
import time

def _check_add(x, S):
    """Check if adding x to S keeps it valid."""
    # S is a set
    # We need to check if there exist distinct a,b in S such that:
    # a+b in S, a+x in S, b+x in S.
    # Note: x is not in S.
    S_list = list(S)
    for i in range(len(S_list)):
        a = S_list[i]
        if a + x not in S:
            continue
        for j in range(i+1, len(S_list)):
            b = S_list[j]
            if b + x not in S:
                continue
            if a + b in S:
                return False
    return True

def construct():
    start_time = time.time()
    best_set = set(range(65, 129))
    best_size = len(best_set)
    
    for restart in range(200):
        if time.time() - start_time > 85:
            break
            
        current_set = set(range(65, 129))
        candidates = list(range(1, 65))
        random.shuffle(candidates)
        
        for x in candidates:
            if _check_add(x, current_set):
                current_set.add(x)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set
    
    return list(best_set)
# EVOLVE-BLOCK-END
