import time
import random

N = 256

def construct():
    start_time = time.time()
    deadline = start_time + 25.0  # Safety margin for 30s limit
    
    best_set = None
    best_size = float('inf')
    
    # We can try a few different heuristics or random restarts if time allows
    # But the "max blocking" greedy is the primary strategy.
    
    def is_feasible(x, S, Sums):
        # Check if x can be added to S without breaking Sidon property
        # New sums: 2x, x+s for s in S
        # Existing sums: Sums
        # Collision if any new sum is in Sums, or if new sums collide with each other
        # New sums are distinct from each other?
        # 2x == x+s => x==s (impossible since x not in S)
        # x+s1 == x+s2 => s1==s2 (distinct s)
        # So new sums are distinct internally.
        # Just check against Sums.
        
        if 2*x in Sums:
            return False
        for s in S:
            if x + s in Sums:
                return False
        return True

    def get_feasible_set(S, Sums, current_feasible):
        # Update current_feasible based on new S, Sums
        # Actually, we can just recompute or filter.
        # Since N is small, recomputing is fine, but filtering is faster.
        new_feasible = set()
        for x in current_feasible:
            if is_feasible(x, S, Sums):
                new_feasible.add(x)
        return new_feasible

    def greedy_max_blocking():
        S = []
        Sums = set()
        Feasible = set(range(1, N + 1))
        
        while Feasible:
            # If time is up, break and return best found so far? 
            # No, we need a complete valid set. 
            # The greedy loop is fast, so we shouldn't time out inside.
            
            best_x = None
            max_blocks = -1
            
            for x in Feasible:
                # Calculate how many other elements in Feasible would be blocked by adding x
                # Simulate adding x
                # New Sums would be Sums ∪ {2x} ∪ {x+s for s in S}
                # We don't actually need to update global Sums, just check conflicts.
                
                blocks = 0
                # For each y in Feasible \ {x}, check if y becomes infeasible
                for y in Feasible:
                    if y == x:
                        continue
                    # Check if y is feasible with x added
                    # y is infeasible if:
                    # 1. 2y in Sums ∪ {2x} ∪ {x+s}
                    # 2. y+s in Sums ∪ {2x} ∪ {x+s'} for some s, s' in S ∪ {x}
                    
                    # Efficient check:
                    # y conflicts with x if:
                    # a) 2y == 2x (y=x, handled)
                    # b) 2y == x + s for some s in S
                    # c) y + s == 2x for some s in S
                    # d) y + s == x + s' for some s, s' in S
                    # e) y + s == 2y (y=s, impossible)
                    # f) y + x == 2y (x=y, handled)
                    # g) y + x == x + s (y=s, impossible)
                    
                    # Let's just use the is_feasible logic but with temporary additions.
                    # To be fast, we can check specific conditions.
                    
                    # Condition 1: 2y collides with new sums
                    if 2*y == 2*x: pass # y=x
                    elif 2*y in Sums:
                        # y was already infeasible? No, y is in Feasible, so 2y not in Sums.
                        # So this check is redundant if we trust Feasible.
                        # But adding x doesn't add 2y.
                        pass
                    
                    # New sums involving x: 2x, x+s
                    # Check if 2y is in {2x} ∪ {x+s}
                    if 2*y == 2*x:
                        pass
                    else:
                        # Check 2y == x + s
                        s_candidate = 2*y - x
                        if s_candidate in S:
                            blocks += 1
                            continue # Already blocked, no need to check further for this y?
                            # Actually, if it's blocked, it's blocked.
                        
                    # Check if y+s collides with new sums
                    # y+s == 2x => s = 2x - y
                    if (2*x - y) in S:
                        blocks += 1
                        continue
                        
                    # y+s == x+s' => s - s' = x - y
                    # This is equivalent to saying there exist s, s' in S such that s - s' = x - y
                    # Or s + y = s' + x
                    # Let's just iterate S for y to be safe and simple, N is small.
                    
                    blocked = False
                    for s in S:
                        # Check y+s against 2x
                        if y + s == 2*x:
                            blocked = True
                            break
                        # Check y+s against x+s'
                        # y + s = x + s' => s' = y + s - x
                        s_prime = y + s - x
                        if s_prime in S and s_prime != s: # s_prime != s is implied if x!=y
                            blocked = True
                            break
                        # Check y+s against Sums (old)
                        # If y+s in Sums, y was already infeasible?
                        # Yes, y is in Feasible, so y+s not in Sums for any s in S.
                        
                    if not blocked:
                        # Check 2y against x+s
                        # 2y = x + s => s = 2y - x
                        # We checked this above.
                        
                        # Wait, I missed one: 2y vs x+s
                        # If 2y == x + s, then y is blocked.
                        # I checked `s_candidate = 2*y - x` in S.
                        
                        # And y+s vs 2x.
                        # I checked `2*x - y` in S.
                        
                        # And y+s vs x+s'.
                        # I iterated S.
                        
                        # So if not blocked, y remains feasible.
                        pass
                    else:
                        pass # blocked counted
                
                # Wait, the logic above is messy. Let's just simulate.
                # It's O(|S|) per y, and |Feasible| is small enough.
                
            # Let's rewrite the inner loop cleanly.
            
            best_x = None
            max_blocks = -1
            for x in Feasible:
                blocks = 0
                for y in Feasible:
                    if y == x: continue
                    # Check if y becomes infeasible if x is added
                    # y is infeasible if any sum involving y is in NewSums
                    # NewSums = Sums ∪ {2x} ∪ {x+s for s in S}
                    
                    # 1. 2y in NewSums
                    if 2*y == 2*x:
                        # y=x, skipped
                        pass
                    elif 2*y in Sums:
                        # y was infeasible, contradiction.
                        pass
                    else:
                        # Check 2y == x+s
                        if (2*y - x) in S:
                            blocks += 1
                            continue
                    
                    # 2. y+s in NewSums for some s in S ∪ {x}
                    # y+s in Sums? No, y feasible.
                    # y+s == 2x?
                    if (2*x - y) in S:
                        blocks += 1
                        continue
                    # y+s == x+s'?
                    # y+s - s' = x
                    # s - s' = x - y
                    # Check if there exist s, s' in S such that s - s' = x - y
                    # And s != s' (implied if x!=y)
                    # And s, s' are distinct from y? No, s,s' in S.
                    
                    diff = x - y
                    found_pair = False
                    # Iterate s in S
                    # s' = s - diff
                    for s in S:
                        s_prime = s - diff
                        if s_prime in S and s_prime != s:
                            found_pair = True
                            break
                    if found_pair:
                        blocks += 1
                        continue
                        
                    # If we reach here, y is still feasible
                    pass
                
                if blocks > max_blocks:
                    max_blocks = blocks
                    best_x = x
            
            # Add best_x
            S.append(best_x)
            Sums.add(2*best_x)
            for s in S[:-1]: # All previous elements
                Sums.add(best_x + s)
            
            # Update Feasible
            new_feasible = set()
            for y in Feasible:
                if y == best_x: continue
                if is_feasible(y, S, Sums):
                    new_feasible.add(y)
            Feasible = new_feasible
            
        return S

    # Run the greedy algorithm
    # To improve, we can try multiple restarts with slight variations?
    # The "max blocking" is deterministic. 
    # Maybe randomize tie-breaking?
    
    best_set = greedy_max_blocking()
    best_size = len(best_set)
    
    # If we have time, try a few random restarts with different tie-breaking
    # or a "lookahead" of 1 step.
    # But 16 is the current best. Can we get 15?
    # Theoretical minimum for N=256 is around sqrt(N) ~ 16.
    # Actually, maximal Sidon sets are usually larger than minimum Sidon sets.
    # Minimum Sidon set size is ~sqrt(N). Maximal can be smaller?
    # No, maximal means you can't add more. 
    # A random maximal set is usually O(N^0.75) or similar? 
    # Wait, "Small inclusion-maximal Sidon sets".
    # The problem asks to *minimize* the cardinality of a *maximal* set.
    # This is a known problem. For N=256, the answer is likely around 10-15.
    # The greedy "smallest first" gives 16.
    # The "max blocking" greedy should give smaller.
    
    return best_set
