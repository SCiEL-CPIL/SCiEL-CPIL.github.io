import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin
    
    # Reverse Greedy: Try to include large numbers first
    # This tends to maximize the sum of reciprocals because small numbers
    # have high weight and are less likely to be blocked by large number choices
    # compared to small numbers blocking large ones? 
    # Actually, small numbers block many large ones. Large numbers block few small ones.
    # Example: 1 blocks 2,3 (1,2,3), 3,5 (1,3,5) etc.
    # 256 blocks 254,255 (254,255,256) and 255,256? No.
    # If we pick 1, we cannot pick 2 and 3 together.
    # If we pick 256, we cannot pick 254 and 255 together.
    # Since 1/1 > 1/256, it is crucial to include 1 if possible.
    # Does picking 256 prevent picking 1? No.
    # Does picking 1 prevent picking 256? No.
    # However, the structure of 3-APs means that small numbers are "hubs".
    # A number $x$ is in a 3-AP if there exist $a, c$ such that $x = (a+c)/2$.
    # If we pick small numbers, we restrict the choices for large numbers.
    # If we pick large numbers, we restrict choices for other large numbers, but small numbers are "safe" from being blocked by large numbers unless the large numbers form a progression with them.
    # E.g., 1, 128, 255. If I pick 255 and 128, I block 1.
    # So picking large numbers CAN block small numbers.
    
    # Let's stick to the standard heuristic for this specific "reciprocal weight" problem:
    # The optimal set is often related to the "Salem-Spencer" type sets or just greedy.
    # Literature suggests that for maximizing sum of reciprocals, one should prioritize small integers.
    # Wait, if I prioritize small integers, I get 1, 2, 4, 8... (Powers of 2 are 3-AP free).
    # Sum of reciprocals of powers of 2 is 1.
    # Greedy smallest-first:
    # 1 (ok)
    # 2 (ok, 1,2 no AP yet)
    # 3 (no, 1,2,3 is AP)
    # 4 (ok, 1,2,4 no AP; 2,4 no AP)
    # 5 (ok, 1,4,5? No. 1,2,5? No. 1,4,? No. 2,4,5? No. 1,2,4,5. 
    #    Check 1,2,4,5 for APs:
    #    1,2,3 (3 not in)
    #    1,3,5 (3 not in)
    #    1,4,7 (7 not in)
    #    2,3,4 (3 not in)
    #    2,4,6 (6 not in)
    #    3,4,5 (3 not in)
    #    4,5,6 (6 not in)
    #    So 5 is ok.)
    
    # The current program does smallest-first.
    # Is reverse-first better?
    # Let's try a randomized greedy with multiple restarts and keep the best.
    # But 90s is a lot. We can do many restarts.
    
    best_set = []
    best_score = 0.0
    
    # Heuristic 1: Smallest First (Current)
    # Heuristic 2: Largest First
    # Heuristic 3: Randomized Order with bias towards small numbers
    
    def score_set(s):
        if not s: return 0.0
        return sum(1.0/x for x in s)
    
    def is_3ap_free(s):
        s_set = set(s)
        # Check all pairs
        # O(n^2) check
        # For a, b in s, if 2b-a in s or 2a-b in s -> AP
        # More efficient: for a in s, for b in s:
        #   c = 2*b - a
        #   if c in s_set and c != a and c != b: return False
        #   c = 2*a - b
        #   if c in s_set and c != a and c != b: return False
        # Actually just checking 2*b-a is enough if we iterate carefully?
        # If a, b, c is AP, then b is midpoint.
        # So for each pair (a, c), check if (a+c)/2 is integer and in s.
        # Or for each pair (a, b), check if 2*b-a is in s.
        
        # Using set for O(1) lookup
        # We only need to check if adding a new element x creates an AP.
        # But for full validation:
        for i in range(len(s)):
            for j in range(i+1, len(s)):
                a, b = s[i], s[j]
                # Check if a, b, c is AP => c = 2b - a
                c = 2*b - a
                if c in s_set and c > b: # c must be distinct and > b to avoid duplicates/reverses
                    return False
                # Check if c, a, b is AP => c = 2a - b
                c = 2*a - b
                if c in s_set and c < a:
                    return False
        return True

    def greedy_add(order):
        s = []
        s_set = set()
        for x in order:
            # Check if x can be added
            # x creates AP if there exist a, b in s such that:
            # 1. a, b, x is AP => x = 2b - a => a = 2b - x
            # 2. a, x, b is AP => x = (a+b)/2 => a+b = 2x
            # 3. x, a, b is AP => b = 2a - x => b = 2a - x
            
            # Efficient check:
            # For each a in s:
            #   if 2*a - x in s_set: return False (x, a, 2a-x)
            #   if 2*x - a in s_set: return False (a, x, 2x-a)
            #   # Also need to check if x is the midpoint? 
            #   # If a, x, b is AP, then b = 2x - a. This is covered by "2*x - a in s_set" if we treat a as one end.
            #   # Wait. 
            #   # Case 1: x is smallest. x, a, b. b = 2a - x. Check if 2a-x in s.
            #   # Case 2: x is middle. a, x, b. b = 2x - a. Check if 2x-a in s.
            #   # Case 3: x is largest. a, b, x. b = 2b - a => a = 2b - x. 
            #   #    This is equivalent to checking if there is a pair (a, b) in s such that 2b - a = x.
            #   #    Which is: for each b in s, check if 2b - x in s.
            
            # So we need to check:
            # 1. For each a in s: is 2*a - x in s? (x smallest)
            # 2. For each a in s: is 2*x - a in s? (x middle)
            # 3. For each b in s: is 2*b - x in s? (x largest)
            
            # Note: 2*a - x and 2*b - x are the same check if we just iterate all elements in s.
            # Let y be an element in s.
            # If 2*y - x in s, then we have an AP involving x and two elements from s?
            # Let's verify.
            # If 2*y - x = z (z in s), then x, y, z are in AP?
            # x + z = 2y. Yes.
            # So the condition is: Does there exist y, z in s such that x + z = 2y?
            # This covers all cases where x is part of a 3-AP with two existing elements.
            
            can_add = True
            for y in s:
                z = 2*y - x
                if z in s_set:
                    can_add = False
                    break
                # Also check if x is the midpoint?
                # If x is midpoint, x = (a+b)/2 => 2x = a+b.
                # This is NOT covered by "2y - x in s" where y, z in s.
                # In that case, the existing elements are a and b.
                # So we must also check: Does there exist a, b in s such that a + b = 2x?
                
            # So two checks:
            # 1. Is x an endpoint? Exists y, z in s s.t. x + z = 2y. (z = 2y - x)
            # 2. Is x the midpoint? Exists a, b in s s.t. a + b = 2x.
            
            if can_add:
                # Check midpoint case
                for a in s:
                    b = 2*x - a
                    if b in s_set:
                        can_add = False
                        break
            
            if can_add:
                s.append(x)
                s_set.add(x)
        return s

    # Strategy:
    # 1. Run Greedy with Smallest First
    # 2. Run Greedy with Largest First
    # 3. Run several Randomized Greedies with bias towards small numbers
    
    orders = [
        list(range(1, 257)),          # Smallest first
        list(range(256, 0, -1)),      # Largest first
    ]
    
    # Add some randomized orders
    import random
    random.seed(42)
    for _ in range(10):
        # Bias towards small numbers
        # We can just shuffle, but maybe sort by 1/x + random_noise?
        # Or just simple shuffle
        o = list(range(1, 257))
        random.shuffle(o)
        orders.append(o)
        
    # Also try "Weighted Random" where probability of picking next smallest available is higher
    # But simple shuffle might be enough to explore different local optima.
    
    for o in orders:
        if time.time() > deadline:
            break
        s = greedy_add(o)
        sc = score_set(s)
        if sc > best_score:
            best_score = sc
            best_set = s
            
    # Local Search on best_set
    # Try to swap elements or add/remove
    # Given time, maybe just return best_set
    
    return best_set
