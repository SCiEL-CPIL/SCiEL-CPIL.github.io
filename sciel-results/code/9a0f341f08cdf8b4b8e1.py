import random
import time

def construct():
    # Seed for reproducibility in this run, but we want best found
    random.seed(42)
    N = 128
    deadline = time.time() + 55  # Safety margin
    
    # Initial state: Odds
    S = [i for i in range(1, N+1) if i % 2 == 1]
    
    def has_violation(S_set):
        # Fast check: return True if any violation exists
        # For speed, we check specific patterns or use a heuristic
        # This is O(n^3) which is 2M ops. Might be slow in Python loop.
        # We need a faster check or smaller candidate pool.
        pass

    # Better: Use the structure found in hint.
    # The hint program achieved 82.
    # I will try to construct a set of size 85+ manually or via quick greedy.
    
    # Greedy construction:
    # Start with empty set.
    # Add numbers in random order if they don't create a violation.
    # But checking violation is expensive.
    
    # Alternative: Start with a known good large set and improve.
    # Set: Odds (64) + some Evens.
    # Which evens can we add?
    # If we add e, we check if it creates a violation.
    
    S = set(range(1, N+1, 2))
    
    def is_valid_addition(S, x):
        # Check if adding x to S creates a violation.
        # A violation is a triple a,b,c in S_new such that a+b, a+c, b+c in S_new.
        # Since S was valid, the only new violations must involve x.
        # x can be an element or a sum.
        
        # Case 1: x is an element (a=x, or b=x, or c=x)
        # We need b,c in S such that x+b, x+c, b+c in S_new.
        # Note x is not in S yet.
        # Sums involving x: x+b, x+c.
        # These sums must be in S_new.
        # Since x > 0, x+b > b.
        
        # Case 2: x is a sum.
        # a+b = x.
        # Then we need a,c in S such that a+c, b+c in S_new.
        
        # This check is still O(n^2).
        
        # Optimization:
        # If we only add x, we check all pairs (u,v) in S.
        # 1. If u+v == x:
        #    Check if u+w and v+w are in S for any w? No.
        #    We need a,b,c distinct.
        #    If a+b=x, we need a+c, b+c in S_new.
        #    c must be in S.
        #    If a+c in S and b+c in S, then {a,b,c, x, a+c, b+c} is a violation.
        #    So if there exists c in S such that a+c in S and b+c in S, we cannot add x.
        
        # 2. If x is an element:
        #    We need b,c in S such that x+b in S_new, x+c in S_new, b+c in S_new.
        #    x+b in S_new means x+b in S (since x+b > x, x not in S yet? No, x is being added).
        #    Actually, if x+b == x, impossible.
        #    So we check:
        #    For all b,c in S:
        #       if b+c in S:
        #           if x+b in S and x+c in S:
        #               return False # Violation found
        
        # This is O(n^2) per addition. 64 elements. 64*64*64? No, n^2.
        # 64 candidates to add. 4096 checks per candidate. 250k ops. Fast enough.
        
        S_list = list(S)
        for i in range(len(S_list)):
            for j in range(i+1, len(S_list)):
                b = S_list[i]
                c = S_list[j]
                if b + c in S:
                    # If b+c is already in S, then adding x creates violation if:
                    # x+b in S and x+c in S
                    if x + b in S and x + c in S:
                        return False
                # Also check if x+b or x+c equals some existing sum?
                # The condition is "a+b, a+c, b+c are also selected".
                # The elements a,b,c must be in S_new.
                # The sums a+b, a+c, b+c must be in S_new.
                
                # If x is an element:
                # Let a=x.
                # Sums: x+b, x+c, b+c.
                # We need x+b in S_new, x+c in S_new, b+c in S_new.
                # b,c in S.
                # x+b in S_new => x+b in S (since x+b > x, and x is new).
                # x+c in S => x+c in S.
                # b+c in S => b+c in S.
                
                # So the check above covers the case where x is the largest element.
                # What if x is a smaller element?
                # Symmetric. The check covers all pairs b,c.
        
        # Check if x is a sum:
        # Let a+b = x.
        # We need a,c,b such that a+c in S_new, b+c in S_new.
        # a,b in S.
        # a+c in S_new => a+c in S (if a+c != x) or a+c=x?
        # If a+c = x, then x is in S_new.
        # So we need:
        # For all a,b in S with a+b == x:
        #    For all c in S:
        #       if a+c in S_new and b+c in S_new:
        #          return False
        #       a+c in S_new:
        #          if a+c == x: True (x is added)
        #          else: a+c in S
        #       Same for b+c.
        
        for i in range(len(S_list)):
            a = S_list[i]
            if a < x:
                b = x - a
                if b in S and b > a:
                    # Found pair a,b in S with sum x
                    for c in S_list:
                        if c == a or c == b: continue
                        # Check sums
                        sum_ac = a + c
                        sum_bc = b + c
                        # Are they in S_new?
                        ok_ac = (sum_ac == x) or (sum_ac in S)
                        ok_bc = (sum_bc == x) or (sum_bc in S)
                        if ok_ac and ok_bc:
                            return False
        
        return True

    # Try to add evens
    evens = [i for i in range(2, N+1, 2)]
    random.shuffle(evens)
    
    for x in evens:
        if time.time() > deadline: break
        if is_valid_addition(S, x):
            S.add(x)
            
    # Try to add remaining odds? No, started with all odds.
    # Try to swap?
    
    return sorted(list(S))
