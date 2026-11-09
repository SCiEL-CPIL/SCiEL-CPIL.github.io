import time
import random

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 85.0
    
    def is_valid_set(S):
        """Full validation for safety."""
        elems = sorted(S)
        for i, a in enumerate(elems):
            for j in range(i+1, len(elems)):
                b = elems[j]
                for k in range(j, len(elems)):
                    c = elems[k]
                    if (b + c) % a == 0:
                        return False
        return True

    def can_add(S_bool, x):
        """
        Check if x can be added to set represented by S_bool (list of bools, 1-indexed).
        S_bool[i] is True if i is in S.
        """
        # 1. x is the divisor a
        # Need: no b, c in S such that x < b <= c and (b+c) % x == 0
        for b in range(x + 1, n + 1):
            if S_bool[b]:
                for c in range(b, n + 1):
                    if S_bool[c] and (b + c) % x == 0:
                        return False
        
        # 2. x is the middle term b
        # Need: no a, c in S such that a < x <= c and (x+c) % a == 0
        for a in range(1, x):
            if S_bool[a]:
                for c in range(x, n + 1):
                    if S_bool[c] and (x + c) % a == 0:
                        return False
        
        # 3. x is the largest term c
        # Need: no a, b in S such that a < b < x and (b+x) % a == 0
        for a in range(1, x):
            if S_bool[a]:
                for b in range(a + 1, x):
                    if S_bool[b] and (b + x) % a == 0:
                        return False
        
        return True

    best_size = 0
    best_set = set()
    
    # Multi-start greedy
    # Try different starting tails and random orders for adding smaller elements
    for trial in range(50):
        if time.time() > deadline - 10:
            break
            
        # Start with a tail [k, 256]
        # k between 150 and 200
        k = random.randint(150, 200)
        S_bool = [False] * (n + 1)
        S_set = set()
        
        # Initialize with tail
        for i in range(k, n + 1):
            S_bool[i] = True
            S_set.add(i)
            
        # Candidates to add: 1 to k-1
        candidates = list(range(1, k))
        random.shuffle(candidates)
        
        # Greedily add
        for x in candidates:
            if time.time() > deadline - 5:
                break
            if can_add(S_bool, x):
                S_bool[x] = True
                S_set.add(x)
                
        # Try to add remaining candidates in random order again? 
        # Or just check if we can add any from the full range 1..256 not in S?
        # The tail elements are already there. We only added from 1..k-1.
        # Are there elements in [k, 256] that we might have missed? No, we started with them.
        # But maybe removing a tail element allows adding 2 smaller ones?
        
        # Local Search: Try swaps
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            # Try to remove one element and add two? Too expensive.
            # Try to remove one element and add one different one? Size stays same.
            # We want to increase size.
            
            # Try to add any element not in S
            for x in range(1, n + 1):
                if not S_bool[x]:
                    if can_add(S_bool, x):
                        S_bool[x] = True
                        S_set.add(x)
                        improved = True
                        # Restart loop as S changed
                        break
                        
        if len(S_set) > best_size:
            best_size = len(S_set)
            best_set = set(S_set)
            
    # Final Polish: Try to add any remaining elements to best_set
    S_bool_best = [False] * (n + 1)
    for x in best_set:
        S_bool_best[x] = True
        
    for x in range(1, n + 1):
        if not S_bool_best[x]:
            if can_add(S_bool_best, x):
                S_bool_best[x] = True
                best_set.add(x)
                
    # Final validation
    if is_valid_set(best_set):
        return list(best_set)
        
    # Fallback
    return list(range(171, 257))
