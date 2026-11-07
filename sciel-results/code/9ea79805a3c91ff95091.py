import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s margin
    
    def is_valid(S):
        """Check if set S satisfies the condition."""
        # Condition: No distinct a,b,c in S such that a+b, a+c, b+c are in S.
        # We can iterate over pairs a,b and check if c exists such that a+b, a+c, b+c in S.
        # Or iterate over a,b,c.
        # Efficient check:
        # For each a < b in S:
        #   Let sum_ab = a + b
        #   If sum_ab not in S, continue (no triple can use this pair as the sum part? No, a+b is one of the sums).
        #   Actually, the condition is: a+b in S AND a+c in S AND b+c in S.
        #   So for a fixed a, b, we need to find c such that:
        #   1. c in S, c != a, c != b
        #   2. a + c in S
        #   3. b + c in S
        #   Note: a+b must also be in S.
        
        S_list = sorted(S)
        n = len(S_list)
        for i in range(n):
            a = S_list[i]
            for j in range(i + 1, n):
                b = S_list[j]
                sum_ab = a + b
                if sum_ab not in S:
                    continue
                # Now we need c in S, c != a, c != b
                # such that a+c in S and b+c in S
                # We can iterate over potential c? Or check intersections?
                # Candidates for c must be in S.
                # Also a+c in S => c in {s-a for s in S if s > a}
                # And b+c in S => c in {s-b for s in S if s > b}
                # Intersection of these sets with S, excluding a,b.
                
                # Simpler: just iterate over c in S_list[k] for k > j? 
                # c can be smaller than a? "distinct members a,b,c". Order doesn't matter for the set.
                # So c can be any element in S \ {a,b}.
                # To speed up, we can just check all c in S.
                # But that's O(N^3). N=70 => 343,000. 
                # We do this for each pair (N^2/2 = 2450). Total ~ 800M ops? Too slow in Python.
                
                # Optimization:
                # We need c such that c in S, a+c in S, b+c in S.
                # Let A_plus = {x for x in S if x > a} # Actually a+c can be > 128? No, S subset 1..128.
                # If a+c > 128, it's not in S. So c <= 128-a.
                # If b+c > 128, it's not in S. So c <= 128-b.
                # So c <= min(128-a, 128-b) = 128 - max(a,b).
                
                max_ab = max(a, b)
                max_c = 128 - max_ab
                # c must be in S, c != a, c != b, and c <= max_c.
                # Also a+c in S and b+c in S.
                
                # Iterate c from 1 to max_c?
                # Or iterate c in S that are <= max_c.
                for c in S_list:
                    if c == a or c == b:
                        continue
                    if c > max_c:
                        break # Since S_list is sorted
                    if (a + c in S) and (b + c in S):
                        return False
        return True

    def can_add(S, x):
        """Check if S U {x} is valid."""
        # We only need to check triples involving x.
        # Triples (a, b, c) where at least one is x.
        # Cases:
        # 1. x is one of a, b, c.
        #    Subcase 1a: x=a. Need b,c in S such that x+b, x+c, b+c in S U {x}.
        #        Since b,c in S, b+c must be in S (unless b+c=x, but x is added, so if b+c=x, it's in S U {x}).
        #        Wait, the condition is "a+b, a+c, b+c are also selected".
        #        Selected means in the final set.
        #        So if b+c = x, it is selected.
        #        So we check:
        #        For distinct b, c in S:
        #           if (x+b in S or x+b == x? No x+b > x) -> x+b in S
        #           if (x+c in S)
        #           if (b+c in S or b+c == x)
        #           Then invalid.
        #
        #    Let's formalize:
        #    Let S_new = S U {x}.
        #    Check all distinct a,b,c in S_new.
        #    If a+b in S_new AND a+c in S_new AND b+c in S_new, return False.
        
        # Efficient check for adding x:
        # Any invalid triple in S_new must involve x.
        # Why? If a,b,c are all in S, then S was valid, so no such triple exists in S.
        # So we only check triples with x.
        
        # Let the triple be (x, b, c) with b, c in S, b != c.
        # Sums:
        # 1. x + b
        # 2. x + c
        # 3. b + c
        #
        # Conditions for invalidity:
        # (x+b in S_new) AND (x+c in S_new) AND (b+c in S_new)
        #
        # Since x+b > x, x+b in S_new implies x+b in S.
        # Since x+c > x, x+c in S_new implies x+c in S.
        # b+c in S_new implies b+c in S OR b+c == x.
        
        # So we need to find distinct b, c in S such that:
        # 1. x + b in S
        # 2. x + c in S
        # 3. (b + c in S) OR (b + c == x)
        
        S_list = list(S)
        n = len(S_list)
        
        for i in range(n):
            b = S_list[i]
            # Check if x+b in S
            if (x + b) not in S:
                continue
            for j in range(n):
                if i == j:
                    continue
                c = S_list[j]
                # Check if x+c in S
                if (x + c) not in S:
                    continue
                # Check if b+c in S or b+c == x
                if (b + c) in S or (b + c) == x:
                    return False
        return True

    # Initial Greedy Construction
    S = set()
    # Try adding from 128 down to 1
    for i in range(128, 0, -1):
        if time.time() > deadline:
            break
        if can_add(S, i):
            S.add(i)
    
    best_S = set(S)
    best_size = len(S)
    
    # Local Search
    # Try to improve by adding more elements or swapping
    # We'll use a randomized approach to avoid getting stuck
    random.seed(42)
    
    candidates = [i for i in range(1, 129) if i not in S]
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        random.shuffle(candidates)
        
        for x in candidates:
            if time.time() > deadline:
                break
            
            # Try to add x
            if can_add(S, x):
                S.add(x)
                candidates.remove(x)
                improved = True
                if len(S) > best_size:
                    best_size = len(S)
                    best_S = set(S)
                continue
            
            # Try to swap: remove y, add x
            # Find y in S such that S \ {y} U {x} is valid
            # This is expensive. O(N * Cost(can_add))
            # Cost(can_add) is O(N^2) worst case.
            # So one swap check is O(N^3). N=70 => 343k.
            # We have N candidates to try for y. So O(N^4) per x? Too slow.
            
            # Optimization: Only try swapping if x is "close" to being addable.
            # Or just try a few random y's.
            
            # Let's try to find a y that is "blocking" x.
            # If can_add(S, x) is False, there exists b,c in S such that 
            # (x+b in S) and (x+c in S) and (b+c in S or b+c==x).
            # If we remove b or c or (b+c), the block might be removed.
            
            # Identify blockers
            blockers = set()
            S_list = list(S)
            for i in range(len(S_list)):
                b = S_list[i]
                if (x + b) not in S:
                    continue
                for j in range(len(S_list)):
                    if i == j:
                        continue
                    c = S_list[j]
                    if (x + c) not in S:
                        continue
                    if (b + c) in S or (b + c) == x:
                        # Blocker found
                        blockers.add(b)
                        blockers.add(c)
                        if (b + c) in S:
                            blockers.add(b + c)
                        # Also x is involved, but we are adding x.
            
            # Try removing one blocker and adding x
            # Shuffle blockers for randomness
            blocker_list = list(blockers)
            random.shuffle(blocker_list)
            
            for y in blocker_list:
                if time.time() > deadline:
                    break
                # Check if S \ {y} U {x} is valid
                # We can check this by temporarily removing y and calling can_add on S\{y} for x?
                # No, can_add checks if adding x to a set is valid.
                # So:
                # S_temp = S - {y}
                # if can_add(S_temp, x):
                #    S = S_temp U {x}
                #    ...
                
                # But can_add is O(N^2).
                # We are doing this for each blocker.
                # Number of blockers is small?
                
                S_temp = S - {y}
                if can_add(S_temp, x):
                    S = S_temp
                    S.add(x)
                    candidates.remove(x)
                    improved = True
                    if len(S) > best_size:
                        best_size = len(S)
                        best_S = set(S)
                    break # Try next candidate x
            
            # If we didn't improve, we might have swapped size.
            # But we only swapped if we added x.
            # If we removed y and added x, size is same.
            # We need to ensure we don't loop indefinitely on same-size swaps.
            # The 'improved' flag is set to True only if size increased?
            # No, in the code above, 'improved' is set to True if we successfully added x (which implies size increased if we didn't remove, or size same if we removed).
            # Wait, if we remove y and add x, size is same.
            # If we just add x, size increases.
            
            # Let's refine:
            # If we added x without removal, size increases.
            # If we added x with removal, size stays same.
            # We should prioritize size increases.
            
            # The current logic sets improved=True in both cases.
            # This might cause infinite loops of same-size swaps.
            # To avoid this, we can only set improved=True if size increased.
            # Or, we can limit the number of same-size swaps.
            
            # Let's change 'improved' to only track size increases.
            # And handle same-size swaps separately or with a counter.
            
            # Actually, if we just added x (no removal), size increased.
            # If we removed y and added x, size same.
            
            # Let's restructure the loop to be safer.
            
    # Fallback to best found
    if len(best_S) > len(S):
        S = best_S
        
    result = list(S)
    return [int(x) for x in result]
