import time
import random

P = {'n': 128}

def construct():
    n = P['n']
    deadline = time.time() + 55  # Safety margin
    
    # Base set: {65, ..., 128}
    # Any a, b in this set have a+b >= 130 > 128, so a+b not in S.
    # Thus no triple with two elements from this set is forbidden.
    base_set = set(range(65, n + 1))
    candidates = list(range(1, 65))
    
    best_size = len(base_set)
    best_set = base_set.copy()
    
    def is_valid_addition(S, x):
        """
        Check if adding x to S keeps S valid.
        S is assumed to be valid.
        x is not in S.
        We only need to check triples involving x.
        """
        # Case 1: x is a member of the triple {a, b, x}
        # Need a, b in S such that:
        # a+x in S, b+x in S, and (a+b in S or a+b == x)
        
        # Optimization: Precompute sums? No, S changes.
        # Iterate a in S
        # If a+x in S, then check b
        
        # Let's collect candidates for a such that a+x in S
        # Let A = {a in S | a+x in S}
        # For each a in A, we need b in S such that:
        # b+x in S AND (a+b in S or a+b == x)
        
        # To make it faster, let's just iterate pairs if S is small enough?
        # S size is ~64+. Pairs ~2000. 
        # We can optimize by checking membership in S (set lookup O(1)).
        
        s_list = list(S)
        m = len(s_list)
        
        # Case 1: x is a member
        for i in range(m):
            a = s_list[i]
            ax = a + x
            if ax > n or ax not in S:
                continue
            # Now find b
            for j in range(m):
                if i == j: continue
                b = s_list[j]
                bx = b + x
                if bx > n or bx not in S:
                    continue
                ab = a + b
                if ab == x or ab in S:
                    return False
        
        # Case 2: x is a sum
        # Subcase 2a: x = a+b
        # Need a, b, c in S such that a+b=x, a+c in S, b+c in S
        # Iterate a in S
        # b = x - a
        # if b in S and b != a:
        #   check c in S such that a+c in S and b+c in S
        
        for a in s_list:
            b = x - a
            if b <= 0 or b > n: continue
            if b not in S or b == a: continue
            
            # Now we have pair (a,b) summing to x.
            # Need c in S such that a+c in S and b+c in S
            # c must be such that c+x-a in S and c+x-b in S? No.
            # a+c in S and b+c in S.
            
            # Iterate c? Or optimize?
            # Let's iterate c in S
            for c in s_list:
                if c == a or c == b: continue
                ac = a + c
                bc = b + c
                if ac > n or bc > n: continue
                if ac in S and bc in S:
                    return False
            
        # Subcase 2b: x = a+c
        # Need a, b, c in S such that a+c=x, a+b in S, b+c in S
        # This is symmetric to 2a if we rename variables.
        # In 2a, we iterated a, found b=x-a, then checked c.
        # Here, we iterate a, find c=x-a, then check b.
        # The condition is: a+b in S and b+c in S.
        # This is exactly the same structure as 2a:
        # In 2a: Pair (a,b) sums to x. Check c for (a+c in S, b+c in S).
        # In 2b: Pair (a,c) sums to x. Check b for (a+b in S, c+b in S).
        # Since the set S is the same and we iterate all a, and for each a we check all potential partners,
        # and then check all third elements, 2a and 2b and 2c are covered by the same logic if we are careful?
        
        # Let's re-verify.
        # 2a covers triples where the sum of the first two elements (in iteration order) is x.
        # The condition is symmetric for the two elements forming the sum x.
        # If x = a+b, the condition is existence of c s.t. a+c in S, b+c in S.
        # If x = a+c, the condition is existence of b s.t. a+b in S, c+b in S.
        # These are the same condition: "Existence of a third element t in S such that t+a in S and t+b in S".
        # So 2a covers all cases where x is the sum of two elements in S.
        
        # Wait, did I miss any?
        # The triple is {u, v, w}.
        # Sums: u+v, u+w, v+w.
        # One of these is x.
        # If x = u+v, we check existence of w.
        # If x = u+w, we check existence of v.
        # If x = v+w, we check existence of u.
        # My loop in 2a iterates `a` and finds `b` such that `a+b=x`.
        # Then it checks for `c` such that `a+c in S` and `b+c in S`.
        # This covers the case where x is the sum of the pair (a,b) and c is the third element.
        # Since the triple is unordered, and we check all pairs (a,b) that sum to x, we cover all cases where x is a sum of two elements in the triple.
        
        return True

    def try_add(S, x):
        if x in S:
            return S, False
        if is_valid_addition(S, x):
            S.add(x)
            return S, True
        return S, False

    # Strategy: Multiple random restarts with greedy addition + local search
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        
        # Randomize candidate order
        c = candidates.copy()
        random.shuffle(c)
        
        current_set = base_set.copy()
        
        # Greedy add
        for x in c:
            if x in current_set:
                continue
            current_set, _ = try_add(current_set, x)
        
        # Local Search: Try swaps
        # Try to remove an element y from current_set and add a new element z
        # that was previously blocked.
        # We can iterate through current_set and for each y, remove it, 
        # then try to add any candidate z not in set.
        # If we can add at least one z, and the new size is same or larger?
        # Actually, we want to increase size.
        # If we remove y and add z, size is same.
        # But maybe adding z allows adding w later?
        # Simpler: Just try to find a swap that allows a net gain?
        # Or just accept the greedy result. The main issue was validity.
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set.copy()
        
        # If we have time, do a quick local search
        if time.time() < deadline - 5:
             improved = True
             while improved and time.time() < deadline - 2:
                 improved = False
                 # Try swapping
                 for y in list(current_set):
                     if y in base_set: # Don't swap out base set elements? 
                         # Actually, base set elements are "safe" but might block others.
                         # Let's allow swapping any element.
                         pass
                     
                     # Remove y
                     current_set.remove(y)
                     
                     # Try to add any candidate
                     added_any = False
                     for z in c:
                         if z in current_set:
                             continue
                         new_set, added = try_add(current_set, z)
                         if added:
                             current_set = new_set
                             added_any = True
                             # If we added one, we might add more?
                             # Let's just add one per swap attempt to keep it simple
                             # or loop?
                             break # Only one swap at a time?
                             
                     if added_any:
                         # We removed y and added z. Size is same.
                         # But maybe this configuration allows more additions?
                         # We don't know. Let's just restart the greedy phase?
                         # No, that's expensive.
                         # Let's just mark improved to try other swaps.
                         improved = True
                     else:
                         # Put y back
                         current_set.add(y)
                         
                 # If we found a swap, we might have changed the set.
                 # We should re-greedy add on the new set?
                 # That's a full restart.
                 # Let's just do a few random restarts instead of complex local search.
        
    return list(best_set)
