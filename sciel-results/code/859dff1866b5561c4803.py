"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

N = 256

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin
    
    best_set = set(range(192, 257))
    best_size = len(best_set)
    
    # Precompute divisibility constraints? No, just check on the fly.
    # The condition is: for all a < b <= c in S, a does NOT divide (b+c).
    # This is equivalent to: for all a in S, and for all pairs (b,c) in S with a < b <= c,
    # (b+c) % a != 0.
    
    # To speed up checking if we can add x to a set S:
    # 1. If x is in S, fail.
    # 2. For every a in S:
    #    a. If a < x, then we need to check pairs (b,c) where b,c are in S.
    #       Wait, the condition is for a < b <= c.
    #       If we add x, x can be 'a', 'b', or 'c'.
    #       
    #       Case 1: x is 'a'. Then x < b <= c. b,c are in S.
    #       We need: for all b,c in S with x < b <= c, (b+c) % x != 0.
    #       
    #       Case 2: x is 'b'. Then a < x <= c. a,c are in S.
    #       We need: for all a,c in S with a < x <= c, (x+c) % a != 0.
    #       
    #       Case 3: x is 'c'. Then a < b <= x. a,b are in S.
    #       We need: for all a,b in S with a < b <= x, (b+x) % a != 0.
    
    def can_add(x, S):
        # S is a set of integers.
        if x in S:
            return False
        
        # Case 1: x is 'a'
        # Check all b, c in S such that x < b <= c
        # We can iterate b in S where b > x. Then c in S where c >= b.
        # Actually, since S is a set, we can just iterate through elements.
        # Optimization: precompute sorted list?
        # For small sets, O(|S|^2) is fine.
        
        # Let's just iterate all pairs (b,c) in S with b <= c.
        # But we need to filter based on x's role.
        
        # Let's do it simply:
        # For every a in S:
        #   If a < x:
        #      Check if there exists b,c in S with a < b <= c and (b+c) % a == 0? No, that's already true in S.
        #      We need to check new pairs involving x.
        
        # New pairs involving x:
        # 1. x is the smallest (a=x). Pairs (b,c) in S with x < b <= c.
        #    Check if (b+c) % x == 0 for any such pair.
        # 2. x is the middle (b=x). Pairs (a,c) in S with a < x <= c.
        #    Check if (x+c) % a == 0 for any such pair.
        # 3. x is the largest (c=x). Pairs (a,b) in S with a < b <= x.
        #    Check if (b+x) % a == 0 for any such pair.
        
        # Let's implement these checks.
        
        # Case 1: x is a
        # Find all b in S with b > x.
        # For each b, find all c in S with c >= b.
        # Check (b+c) % x != 0.
        # Optimization: if x is large, few b,c > x.
        
        # Case 2: x is b
        # Find all a in S with a < x.
        # Find all c in S with c >= x.
        # Check (x+c) % a != 0.
        
        # Case 3: x is c
        # Find all a in S with a < x.
        # Find all b in S with a < b <= x.
        # Check (b+x) % a != 0.
        
        # To make this faster, we can precompute for the current S?
        # But S changes.
        
        # Let's just iterate.
        
        # Case 1
        for b in S:
            if b > x:
                for c in S:
                    if c >= b:
                        if (b + c) % x == 0:
                            return False
        
        # Case 2
        for a in S:
            if a < x:
                for c in S:
                    if c >= x:
                        if (x + c) % a == 0:
                            return False
        
        # Case 3
        for a in S:
            if a < x:
                for b in S:
                    if a < b <= x:
                        if (b + x) % a == 0:
                            return False
        
        return True

    # Heuristic search
    # Start with a random subset of [1, 256]?
    # Or start with the upper tail and try to add smaller numbers.
    # Or start empty and greedily add?
    
    # Multi-start greedy:
    # Try different starting points or random orders.
    
    # Strategy 1: Start with upper tail, try to add numbers from 191 down to 1.
    # If we can add x, add it.
    # This is deterministic.
    
    # Strategy 2: Randomized greedy.
    # Shuffle the candidates [1, ..., 256] and try to add them in that order.
    # Keep the best found.
    
    # Let's try Strategy 2 with multiple restarts.
    
    # Also, we can try to "remove" numbers from a dense set to allow adding others?
    # That's harder.
    
    # Let's stick to adding.
    
    # We can also try different initial seeds.
    # E.g., start with a random subset of size 10, then greedily add.
    
    # Let's implement a function that takes a set S and tries to extend it by adding elements from a list of candidates.
    
    def extend_set(S, candidates):
        S = set(S)
        for x in candidates:
            if x not in S and can_add(x, S):
                S.add(x)
        return S
    
    # Try deterministic extension from upper tail
    candidates = list(range(191, 0, -1))
    S1 = extend_set(set(range(192, 257)), candidates)
    if len(S1) > best_size:
        best_set = S1
        best_size = len(S1)
    
    # Try randomized extensions
    # We have time for maybe 10-20 restarts?
    # can_add is O(|S|^2). |S| ~ 100. So 10^4 ops per check.
    # 256 candidates. 256 * 10^4 = 2.5 * 10^6 ops per restart.
    # 10 restarts = 2.5 * 10^7 ops. Should be fast in Python? Maybe 1-2 seconds.
    # We have 85 seconds. So we can do many more.
    
    # Let's do 50 restarts.
    random.seed(42)
    
    for _ in range(50):
        if time.time() > deadline:
            break
        
        # Random start: pick a random subset of [1, 256]?
        # Or just shuffle [1, 256] and greedily add.
        candidates = list(range(1, 257))
        random.shuffle(candidates)
        S = set()
        for x in candidates:
            if can_add(x, S):
                S.add(x)
        
        if len(S) > best_size:
            best_set = S
            best_size = len(S)
    
    # Local search: try to swap elements?
    # Or try to add more elements after a greedy pass?
    # The greedy pass already tries all candidates.
    
    # What if we start with a different base?
    # E.g., start with a random set of 50 numbers, then extend.
    
    for _ in range(20):
        if time.time() > deadline:
            break
        
        # Random base
        base = set(random.sample(range(1, 257), 50))
        # Clean base: remove invalid elements?
        # Greedy construction ensures validity.
        # So let's just greedily build from a random order, but maybe bias towards larger numbers?
        
        # Let's try: shuffle [1, 256] but with a bias.
        # Or just shuffle.
        
        candidates = list(range(1, 257))
        random.shuffle(candidates)
        S = set()
        for x in candidates:
            if can_add(x, S):
                S.add(x)
        
        if len(S) > best_size:
            best_set = S
            best_size = len(S)
    
    # Try to improve best_set by local search?
    # E.g., try to add any element not in best_set.
    # The greedy pass did this, but maybe the order matters.
    # We can try to add elements in different orders to best_set.
    
    # Let's try to add elements to best_set in a few different orders.
    missing = [x for x in range(1, 257) if x not in best_set]
    
    # Order 1: Ascending
    S = set(best_set)
    for x in missing:
        if can_add(x, S):
            S.add(x)
    if len(S) > best_size:
        best_set = S
        best_size = len(S)
    
    # Order 2: Descending
    S = set(best_set)
    for x in reversed(missing):
        if can_add(x, S):
            S.add(x)
    if len(S) > best_size:
        best_set = S
        best_size = len(S)
    
    # Order 3: Random
    for _ in range(10):
        if time.time() > deadline:
            break
        random.shuffle(missing)
        S = set(best_set)
        for x in missing:
            if can_add(x, S):
                S.add(x)
        if len(S) > best_size:
            best_set = S
            best_size = len(S)
    
    return list(best_set)

# EVOLVE-BLOCK-END
