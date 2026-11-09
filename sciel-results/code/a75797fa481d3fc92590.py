import random
import math
import time

def construct():
    N = 256
    
    # Precompute conflicts
    # conflict[x] is a bitmask or set of numbers that conflict with x?
    # Actually, for 3-AP, x conflicts with y if there exists z such that {x,y,z} is a 3-AP.
    # It's faster to check validity on the fly or precompute pairs.
    
    # Let's precompute for each x, the list of pairs (a,b) in [1,N] such that {x,a,b} is a 3-AP.
    # Or simpler: For a set S, x is valid if for all y in S, the third term to form AP with x,y is not in S.
    
    def is_valid_add(x, S):
        # S is a set of ints
        # Check if adding x creates a 3-AP
        for y in S:
            # APs involving x and y:
            # 1. x, y, z => z = 2y - x
            # 2. y, x, z => z = 2x - y
            # 3. z, x, y => z = 2x - y (same as 2? No. z, x, y means x is middle. 2x = z+y => z = 2x-y. 
            #    Wait. 
            #    If x, y, z are in AP, one is the average of the other two.
            #    Case 1: x is middle: 2x = y + z => z = 2x - y
            #    Case 2: y is middle: 2y = x + z => z = 2y - x
            #    Case 3: z is middle: 2z = x + y => z = (x+y)/2
            
            # We check if z is in S.
            
            z1 = 2 * x - y
            if 1 <= z1 <= N and z1 != x and z1 != y and z1 in S:
                return False
                
            z2 = 2 * y - x
            if 1 <= z2 <= N and z2 != x and z2 != y and z2 in S:
                return False
                
            s = x + y
            if s % 2 == 0:
                z3 = s // 2
                if 1 <= z3 <= N and z3 != x and z3 != y and z3 in S:
                    return False
        return True

    def get_weight(S):
        return sum(1.0 / x for x in S)

    def local_search(S):
        # S is a set
        improved = True
        S = set(S)
        while improved:
            improved = False
            # Try adding any element
            for x in range(1, N + 1):
                if x not in S and is_valid_add(x, S):
                    S.add(x)
                    improved = True
                    # Restart loop to check again? Or just continue. 
                    # Adding one might block others, but adding is always good for weight.
                    # So we should just add ALL valid elements greedily first.
                    break # Break to restart the for loop with the new S
            
            # Try 1-for-1 swaps
            if not improved:
                for x in range(1, N + 1):
                    if x in S:
                        for y in range(1, N + 1):
                            if y not in S:
                                # Try remove x, add y
                                S.remove(x)
                                if is_valid_add(y, S):
                                    # Check if weight increases
                                    # 1/y > 1/x  => y < x
                                    if y < x:
                                        S.add(y)
                                        improved = True
                                        break
                                S.add(x) # Revert if not taken or failed
                                if improved:
                                    break
                if improved:
                    continue # Restart local search
            
            # Try 1-for-2 swaps: Remove x, add y1, y2
            # This is more expensive but can capture cases where two small numbers beat one large number.
            if not improved:
                # Only try for large x (low weight) to save time
                for x in range(50, N + 1): 
                    if x in S:
                        S.remove(x)
                        found = False
                        # Try to add two elements y1, y2
                        # We want 1/y1 + 1/y2 > 1/x
                        # y1, y2 must be valid with each other and with S\{x}
                        candidates = [y for y in range(1, N + 1) if y not in S and y != x]
                        # Sort by weight descending to try best pairs first
                        candidates.sort(key=lambda v: -1.0/v)
                        
                        # Limit candidates for speed
                        for i, y1 in enumerate(candidates[:50]):
                            if not is_valid_add(y1, S):
                                continue
                            S.add(y1)
                            for y2 in candidates[i+1:i+50]:
                                if y2 == y1: continue
                                if not is_valid_add(y2, S):
                                    continue
                                # Check weight
                                if (1.0/y1 + 1.0/y2) > (1.0/x):
                                    S.add(y2)
                                    found = True
                                    improved = True
                                    break
                            if found:
                                break
                            S.remove(y1)
                        
                        if found:
                            improved = True
                            break # Restart local search with new S
                        else:
                            S.add(x) # Revert x
                            
        return S

    best_set = set()
    best_weight = 0.0
    
    # Multiple random restarts
    # Time limit check
    start_time = time.time()
    max_time = 55.0 # Leave buffer
    
    for restart in range(100):
        if time.time() - start_time > max_time:
            break
            
        # Randomized Greedy Init
        S = set()
        # Randomly shuffle order, but bias towards small numbers?
        # Or just random subset?
        # Let's use a random permutation of 1..N
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        
        for x in candidates:
            if is_valid_add(x, S):
                S.add(x)
                
        # Local Search
        S = local_search(S)
        
        w = get_weight(S)
        if w > best_weight:
            best_weight = w
            best_set = set(S)
            
    return list(best_set)
