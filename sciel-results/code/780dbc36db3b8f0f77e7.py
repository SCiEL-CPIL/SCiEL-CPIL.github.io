import time
import random

def construct():
    N = 256
    deadline = time.time() + 80.0  # Safety margin
    
    # Helper to check if adding x to set S creates a 3-AP
    # S is a set of ints
    def is_valid_addition(x, S):
        # Check if x forms a 3-AP with any two elements in S
        # AP: a, b, c such that 2b = a+c
        # If x is part of the AP, it can be:
        # 1. a: x, b, c => 2b = x+c => c = 2b-x. Check if c in S.
        # 2. b: a, x, c => 2x = a+c => c = 2x-a. Check if c in S.
        # 3. c: a, b, x => 2b = a+x => a = 2b-x. Check if a in S.
        
        # Optimization: 
        # For each b in S:
        #   if 2b - x in S: return False (x is c, or x is a? No, if 2b-x in S, let a=2b-x. Then a,b,x is AP.)
        #   if 2x - b in S: return False (x is b? No, if 2x-b in S, let c=2x-b. Then b,x,c is AP? No. 
        #      Wait. If x is the middle term, a+c=2x. So for each a in S, c=2x-a. If c in S, then a,x,c is AP.)
        
        # Let's just iterate through all pairs in S? No, that's O(|S|^2).
        # Better:
        # For each b in S:
        #   a = 2*b - x
        #   if a in S and a != x: return False
        #   c = 2*x - b
        #   if c in S and c != x: return False
        #   # Also check if x is the middle?
        #   # If x is middle, a+c=2x. 
        #   # This is covered if we iterate b? No.
        #   # If x is middle, then for any a in S, c = 2x - a. If c in S, then a,x,c is AP.
        #   # So we need to check: for any a in S, is 2x - a in S?
        
        # So, for each b in S (acting as the middle of a potential AP involving x? No, b is just an element).
        # Let's rephrase:
        # A 3-AP involving x and two elements from S.
        # Case 1: x is an endpoint.
        #   Subcase 1a: x, a, b. 2a = x+b => b = 2a-x. For each a in S, if b=2a-x in S, invalid.
        #   Subcase 1b: a, b, x. 2b = a+x => a = 2b-x. For each b in S, if a=2b-x in S, invalid.
        # Case 2: x is the middle.
        #   Subcase 2a: a, x, b. 2x = a+b => b = 2x-a. For each a in S, if b=2x-a in S, invalid.
        
        # So we need to check:
        # 1. For each a in S: is (2*a - x) in S?
        # 2. For each b in S: is (2*b - x) in S? (Same as 1, just variable name change)
        # 3. For each a in S: is (2*x - a) in S?
        
        # Note: In 1 and 2, we are checking if x is an endpoint.
        # In 3, we are checking if x is the middle.
        
        # We must ensure the elements are distinct.
        # If 2*a - x == a, then x=a, but x not in S, so ok.
        # If 2*a - x == x, then a=x, but x not in S, so ok.
        # If 2*x - a == a, then x=a, but x not in S, so ok.
        # If 2*x - a == x, then a=x, but x not in S, so ok.
        
        # So we just check membership.
        
        for a in S:
            # Check if x is endpoint (x, a, ?) or (?, a, x)
            # If x, a, c is AP: 2a = x+c => c = 2a-x. If c in S, invalid.
            c = 2*a - x
            if c in S:
                return False
            
            # If a, x, c is AP: 2x = a+c => c = 2x-a. If c in S, invalid.
            c2 = 2*x - a
            if c2 in S:
                return False
                
        return True

    def get_weight(S):
        return sum(1.0 / x for x in S)

    # Initial Greedy
    S = set()
    for x in range(1, N + 1):
        if is_valid_addition(x, S):
            S.add(x)
            
    best_S = set(S)
    best_weight = get_weight(S)
    
    # Hill Climbing
    while time.time() < deadline:
        improved = False
        
        # 1. Try adding any valid x
        added = False
        for x in range(1, N + 1):
            if x not in S and is_valid_addition(x, S):
                S.add(x)
                added = True
                break # Restart loop to re-evaluate
        if added:
            improved = True
            if get_weight(S) > best_weight:
                best_weight = get_weight(S)
                best_S = set(S)
            continue # Restart the while loop to check again
        
        # 2. Try swaps: remove y, add x < y
        # We want to maximize weight, so we prefer small x.
        # Iterate y in S (largest first? or smallest?)
        # Let's iterate x (candidate to add) from 1 to N, and y (candidate to remove) from S.
        # This is O(N * |S| * Cost(IsValid)).
        # |S| ~ 100, N=256. 25600 checks. Each check is O(|S|) ~ 100. Total 2.5M ops.
        # This is fast.
        
        best_swap = None
        best_gain = 0.0
        
        # To speed up, we can precompute valid additions for the "removed" state?
        # No, just do it directly.
        
        # Sort S to iterate?
        # We want to find a swap that increases weight.
        # Gain = 1/x - 1/y. Since x < y, gain > 0.
        # We just need ANY valid swap.
        
        found_swap = False
        # Try smaller x first to get bigger gains?
        for x in range(1, N + 1):
            if x in S:
                continue
            
            # Check if there exists y in S such that S \ {y} U {x} is valid.
            # And x < y (for positive gain, though even if x>y, we might want to add x if it allows more additions later? 
            # But simple hill climbing usually swaps for immediate gain).
            # Actually, if x > y, weight decreases. We only want x < y.
            
            # Optimization: If x is not in S, check if we can remove some y to make x valid.
            # This is complex. Let's stick to: 
            # For each y in S:
            #   For each x < y not in S:
            #     if is_valid_addition(x, S - {y}):
            #        perform swap.
            
            # To avoid O(|S|^2) nested loops in Python being slow:
            # We can iterate y in S.
            # Then check if x can be added to S\{y}.
            
            pass
        
        # Let's implement the swap search more efficiently.
        # We will iterate through potential additions x (1..N).
        # For each x, we want to know if there is a y in S such that removing y makes x valid.
        
        # Actually, let's just iterate through y in S (to remove) and x not in S (to add).
        # We prioritize x < y.
        
        S_list = list(S)
        swap_performed = False
        for y in S_list:
            # Try to replace y with a smaller x
            for x in range(1, y):
                if x in S:
                    continue
                # Check if S\{y} U {x} is valid
                S_temp = S - {y}
                if is_valid_addition(x, S_temp):
                    # Perform swap
                    S.remove(y)
                    S.add(x)
                    swap_performed = True
                    improved = True
                    break
            if swap_performed:
                break
        
        if swap_performed:
            if get_weight(S) > best_weight:
                best_weight = get_weight(S)
                best_S = set(S)
            continue
        
        # If no swap found, try a "kick" to escape local optimum?
        # Or just break if no improvement.
        if not improved:
            # Try a random removal to escape?
            if len(S) > 0 and random.random() < 0.1:
                y = random.choice(list(S))
                S.remove(y)
                improved = True
                continue
            else:
                break

    return list(best_S)
