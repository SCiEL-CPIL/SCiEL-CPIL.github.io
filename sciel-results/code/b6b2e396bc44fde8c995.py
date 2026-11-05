import time

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 80s limit to be safe
    
    best_set = list(range(91, 181))
    max_size = len(best_set)
    
    # Try different lower bounds for the base set
    # We know 91 works with size 90.
    # We try to lower the bound and remove "bad" elements.
    
    for lower_bound in range(1, 91):
        if time.time() > deadline:
            break
            
        S_base = set(range(lower_bound + 1, 181))
        if not S_base:
            continue
        
        # Identify elements in S_base that form a triple with two other elements in S_base
        # a is in S_base, b,c in S_base, b != c
        # 1/a = 1/b + 1/c  =>  a = bc / (b+c)
        # (b-a)(c-a) = a^2
        
        bad_elements = set()
        
        for a in S_base:
            # Check if there exist b, c in S_base, b != c
            # such that (b-a)(c-a) = a^2
            # Let u = b-a, v = c-a
            # u*v = a^2
            # b = a+u, c = a+v
            # b, c must be in S_base
            # lower_bound < a+u <= 180
            # lower_bound < a+v <= 180
            # u != v (since b != c)
            
            a_sq = a * a
            
            # Find divisors of a_sq
            # Iterate u from 1 to sqrt(a_sq)
            # Or just iterate possible u values?
            # u must satisfy:
            # 1 <= u <= 180 - a
            # a_sq / u <= 180 - a
            # u >= a_sq / (180 - a)
            
            max_u = 180 - a
            if max_u <= 0:
                continue
                
            min_u = max(1, (a_sq + max_u - 1) // max_u) # ceil(a_sq / (180-a))
            
            if min_u > max_u:
                continue
                
            found = False
            # Iterate divisors u in [min_u, max_u]
            # To be efficient, iterate d from 1 to sqrt(a_sq)
            # if a_sq % d == 0:
            #   u1 = d
            #   u2 = a_sq // d
            #   check if u1 in range
            #   check if u2 in range
            
            limit = int(a_sq ** 0.5)
            for d in range(1, limit + 1):
                if a_sq % d == 0:
                    u1 = d
                    v1 = a_sq // d
                    u2 = v1
                    v2 = u1
                    
                    # Check pair (u1, v1)
                    if min_u <= u1 <= max_u and min_u <= v1 <= max_u:
                        if u1 != v1:
                            # b = a + u1, c = a + v1
                            # Ensure b, c are in S_base (which is guaranteed by range checks)
                            # Ensure b, c are distinct (u1 != v1)
                            bad_elements.add(a)
                            found = True
                            break
                            
                    # Check pair (u2, v2) - same as above if we just swap, but u2 might be different
                    # Actually u2 = a_sq/d. If d != sqrt, u2 is the other divisor.
                    if u2 != u1:
                        if min_u <= u2 <= max_u and min_u <= v2 <= max_u:
                            if u2 != v2:
                                bad_elements.add(a)
                                found = True
                                break
            if found:
                continue
                
        candidate_set = S_base - bad_elements
        if len(candidate_set) > max_size:
            max_size = len(candidate_set)
            best_set = list(candidate_set)
            
    return best_set
