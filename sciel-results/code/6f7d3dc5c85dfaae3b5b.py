import time
import random

def construct():
    start_time = time.time()
    deadline = 80.0  # safety margin
    
    N = 128
    
    # Helper: check if adding x to set S creates a violation
    # A violation exists if there are distinct a,b,c in S∪{x} such that
    # a+b, a+c, b+c are all in S∪{x}
    # We need to check: for each pair (a,b) in S, if a+b in S∪{x}, 
    # and for each c in S∪{x} distinct from a,b, check a+c and b+c in S∪{x}
    
    def is_valid_set(S):
        """Check if set S has no forbidden triple."""
        S_list = list(S)
        n = len(S_list)
        S_set = set(S_list)
        for i in range(n):
            for j in range(i+1, n):
                a, b = S_list[i], S_list[j]
                ab = a + b
                if ab not in S_set:
                    continue
                for k in range(n):
                    if k == i or k == j:
                        continue
                    c = S_list[k]
                    if a + c in S_set and b + c in S_set:
                        return False
        return True
    
    def check_add(S, x):
        """Check if adding x to set S maintains validity."""
        if x in S:
            return True
        S_list = list(S)
        n = len(S_list)
        S_set = set(S_list)
        # Check triples involving x
        # Case 1: x is one of a,b,c
        for i in range(n):
            for j in range(i+1, n):
                a, b = S_list[i], S_list[j]
                # Check if {x, a, b} or {a, b, c} where c=x creates violation
                # Sub-case: x, a, b are the triple
                # Need x+a, x+b, a+b all in S∪{x}
                if a + b in S_set and x + a in S_set and x + b in S_set:
                    return False
                # Sub-case: a, b, x where x+a, x+b, a+b... same as above
                # Check if a, x, b: same triple
                # Case 2: x is the sum, e.g., x = a+b, and there's c such that a+c, b+c in S
                # Actually we need: for distinct a,b,c in S∪{x}, all three sums in S∪{x}
                # If x is in the triple as a member:
                #   (x,a,b): need x+a, x+b, a+b all in S∪{x}
                #   a+b in S (checked), x+a in S or x+a==x (impossible for a>0), x+b in S
                # If x is a sum:
                #   a+b=x, a+c in S, b+c in S, and a,b,c in S
                for i in range(n):
                    for j in range(i+1, n):
                        a, b = S_list[i], S_list[j]
                        if a + b == x:
                            # Need to find c in S, c!=a,b, with a+c in S and b+c in S
                            for k in range(n):
                                if k == i or k == j:
                                    continue
                                c = S_list[k]
                                if a + c in S_set and b + c in S_set:
                                    return False
        # Case 3: x is not in the triple but x is one of the sums
        # e.g., a+b in S, a+c=x, b+c in S, with a,b,c in S
        # This means x = a+c for some a,c in S, and b+c in S, and a+b in S
        for i in range(n):
            for j in range(i+1, n):
                a, b = S_list[i], S_list[j]
                if a + b not in S_set:
                    continue
                # x = a+c or x = b+c for some c
                # If x = a+c, then c = x-a, need c in S, c!=a,b, and b+c in S
                c = x - a
                if c in S_set and c != a and c != b and b + c in S_set:
                    return False
                c = x - b
                if c in S_set and c != a and c != b and a + c in S_set:
                    return False
        return True
    
    # Start with all odds
    S = set(range(1, N+1, 2))
    best_S = set(S)
    best_size = len(S)
    
    # Try to add evens greedily
    evens = list(range(2, N+1, 2))
    random.shuffle(evens)
    
    for x in evens:
        if time.time() > deadline - 5:
            break
        if check_add(S, x):
            S.add(x)
            if len(S) > best_size:
                best_S = set(S)
                best_size = len(S)
    
    # Local search: try removing one element and adding another
    improvements = True
    while improvements and time.time() < deadline - 5:
        improvements = False
        S_list = list(S)
        random.shuffle(S_list)
        candidates = [x for x in range(1, N+1) if x not in S]
        random.shuffle(candidates)
        
        for remove_elem in S_list:
            if time.time() > deadline - 5:
                break
            S_remove = S - {remove_elem}
            for add_elem in candidates:
                if time.time() > deadline - 5:
                    break
                if check_add(S_remove, add_elem):
                    new_S = S_remove | {add_elem}
                    if len(new_S) > len(S):
                        S = new_S
                        improvements = True
                        if len(S) > best_size:
                            best_S = set(S)
                            best_size = len(S)
                        break
                    elif len(new_S) == len(S) and random.random() < 0.1:
                        # Occasional swap for diversity
                        S = new_S
                        if len(S) > best_size:
                            best_S = set(S)
                            best_size = len(S)
    
    # Try different starting points if time allows
    if time.time() < deadline - 15:
        # Try starting with upper half
        S2 = set(range(N//2+1, N+1))
        best2 = set(S2)
        best2_size = len(S2)
        
        # Greedy add from S2
        remaining = [x for x in range(1, N+1) if x not in S2]
        random.shuffle(remaining)
        for x in remaining:
            if time.time() > deadline - 10:
                break
            if check_add(S2, x):
                S2.add(x)
                if len(S2) > best2_size:
                    best2 = set(S2)
                    best2_size = len(S2)
        
        if best2_size > best_size:
            best_S = best2
            best_size = best2_size
    
    # Final validation
    result = list(best_S)
    # Ensure all are Python int
    result = [int(x) for x in result]
    return result

# Pre-compute a good static solution as fallback
# All odds: 1,3,5,...,127 = 64 elements, always valid
_FALLBACK = list(range(1, 129, 2))

def construct():
    start_time = time.time()
    deadline = 80.0
    
    N = 128
    
    def check_add(S, x):
        """Check if adding x to set S maintains validity."""
        if x in S:
            return True
        S_list = list(S)
        n = len(S_list)
        S_set = set(S_list)
        
        # Check all triples in S ∪ {x}
        # For each pair (a,b) in S, check if a+b in S∪{x}
        # Then for each c in S∪{x} \ {a,b}, check if a+c and b+c in S∪{x}
        
        # Efficient: iterate over pairs in S
        for i in range(n):
            for j in range(i+1, n):
                a, b = S_list[i], S_list[j]
                ab = a + b
                if ab not in S_set and ab != x:
                    continue
                # Now check all c in S ∪ {x} distinct from a, b
                for k in range(n):
                    if k == i or k == j:
                        continue
                    c = S_list[k]
                    ac = a + c
                    bc = b + c
                    if ac in S_set or ac == x:
                        if bc in S_set or bc == x:
                            return False
                # Check c = x
                ac = a + x
                bc = b + x
                if (ac in S_set or ac == x) and (bc in S_set or bc == x):
                    # Need a, b, x all distinct (they are since x not in S)
                    return False
        
        return True
    
    # Strategy 1: Start with odds
    S = set(range(1, N+1, 2))
    best_S = set(S)
    best_size = len(S)
    
    evens = list(range(2, N+1, 2))
    random.shuffle(evens)
    
    for x in evens:
        if time.time() > deadline - 5:
            break
        if check_add(S, x):
            S.add(x)
            if len(S) > best_size:
                best_S = set(S)
                best_size = len(S)
    
    # Local search
    if time.time() < deadline - 10:
        improvements = True
        max_iters = 500
        iters = 0
        while improvements and iters < max_iters and time.time() < deadline - 5:
            improvements = False
            iters += 1
            S_list = list(S)
            random.shuffle(S_list)
            candidates = [x for x in range(1, N+1) if x not in S]
            random.shuffle(candidates)
            
            for remove_elem in S_list:
                if time.time() > deadline - 5:
                    break
                S_remove = S - {remove_elem}
                for add_elem in candidates:
                    if time.time() > deadline - 5:
                        break
                    if check_add(S_remove, add_elem):
                        new_S = S_remove | {add_elem}
                        if len(new_S) > best_size:
                            S = new_S
                            best_S = set(S)
                            best_size = len(S)
                            improvements = True
                            break
    
    # Strategy 2: Try upper half start
    if time.time() < deadline - 15:
        S2 = set(range(N//2+1, N+1))
        best2 = set(S2)
        best2_size = len(S2)
        
        remaining = [x for x in range(1, N+1) if x not in S2]
        random.shuffle(remaining)
        for x in remaining:
            if time.time() > deadline - 10:
                break
            if check_add(S2, x):
                S2.add(x)
                if len(S2) > best2_size:
                    best2 = set(S2)
                    best2_size = len(S2)
        
        if best2_size > best_size:
            best_S = best2
            best_size = best2_size
    
    result = [int(x) for x in best_S]
    return result
