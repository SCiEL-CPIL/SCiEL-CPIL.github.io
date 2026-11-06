import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    N = 256
    
    # Helper to check if adding x to S is valid
    # S is a set, in_set is a boolean list
    def can_add(x, S, in_set):
        # Check if x is already in S
        if in_set[x]:
            return False
        
        # 1. x is middle: a, x, c -> a + c = 2x
        # a < x < c
        # c = 2x - a
        for a in S:
            if a < x:
                c = 2 * x - a
                if c > x and c <= N and in_set[c]:
                    return False
            # 2. x is end: a, b, x -> a + x = 2b -> b = (a+x)/2
            # a < b < x
            elif a < x: # Wait, a must be < x for x to be end? 
                # If a > x, then x is start?
                pass
        
        # Let's re-structure the check clearly
        # For each a in S:
        # Case 1: a, x, c (x is middle). c = 2x - a. Need c > x and c in S.
        # Case 2: a, b, x (x is end). b = (a+x)/2. Need b integer, b < x, b in S.
        #          Note: a can be < x or > x? 
        #          If a > x, then a, b, x implies a > b > x, so b < x.
        #          If a < x, then a < b < x.
        #          So for any a in S, if (a+x) is even, b = (a+x)//2.
        #          If b in S and b != x and b != a, then conflict?
        #          Actually if a, b, x is AP, then a+x=2b.
        #          If b in S, and a in S, and x is candidate, then {a,b,x} is AP.
        #          We need to ensure a, b, x are distinct. Since x not in S yet, x != a, x != b.
        #          So just check if b in S.
        
        # Case 3: x, b, c (x is start). c = 2b - x. Need c > b and c in S.
        #          b > x.
        
        for a in S:
            # Case 1: x is middle
            if a < x:
                c = 2 * x - a
                if c > x and c <= N and in_set[c]:
                    return False
            
            # Case 2: x is end
            # a and x form an AP with b.
            # b = (a + x) / 2
            if (a + x) % 2 == 0:
                b = (a + x) // 2
                # b must be in S. b must be distinct from a and x.
                # Since x not in S, b != x.
                # If b == a, then a+x=2a -> x=a, impossible since x not in S.
                # So just check if b in S.
                if 1 <= b <= N and in_set[b]:
                    return False
            
            # Case 3: x is start
            if a > x:
                c = 2 * a - x
                if c > a and c <= N and in_set[c]:
                    return False
        
        return True

    def get_weight(S):
        return sum(1.0 / x for x in S)

    def local_search(initial_S):
        S = set(initial_S)
        in_set = [False] * (N + 1)
        for x in S:
            in_set[x] = True
        
        best_weight = get_weight(S)
        
        improved = True
        while improved:
            improved = False
            if time.time() > deadline:
                break
                
            # 1. Try to add any element
            added = False
            best_v = -1
            best_v_val = 0.0
            for v in range(1, N + 1):
                if not in_set[v]:
                    if can_add(v, S, in_set):
                        val = 1.0 / v
                        if val > best_v_val:
                            best_v_val = val
                            best_v = v
            if best_v != -1:
                S.add(best_v)
                in_set[best_v] = True
                best_weight += best_v_val
                improved = True
                continue # Restart loop as adding might unlock others? No, adding restricts. But we want to add all possible.
                # Actually, adding one element might make another invalid. 
                # But we want to maximize weight, so we should add the best one.
                # After adding, we should re-evaluate.
                
            # 2. Try to swap
            # Remove u, add v.
            # We want 1/v > 1/u => v < u.
            # So we look for v < u.
            # Iterate u in S, v in [1, u-1] not in S.
            # This is O(|S| * N).
            
            best_swap = None
            best_gain = 0.0
            
            # Optimization: Only consider v that are "small"
            # We can iterate v from 1 to N, and for each v not in S, check if there is a u in S such that u > v and swap is valid.
            # To speed up, we can pre-check if v can be added to S \ {u} for some u.
            
            # Let's iterate u in S
            for u in list(S):
                if time.time() > deadline:
                    return S, best_weight
                
                # Try to remove u
                in_set[u] = False
                S.remove(u)
                
                # Now try to add some v < u
                # We want the best v (smallest v) that can be added.
                # Since we want max weight gain, and gain = 1/v - 1/u, we want min v.
                # So find the smallest v < u not in S such that can_add(v, S, in_set) is True.
                
                found_v = -1
                for v in range(1, u):
                    if not in_set[v]:
                        if can_add(v, S, in_set):
                            found_v = v
                            break # Smallest v is best for this u
                
                if found_v != -1:
                    gain = 1.0 / found_v - 1.0 / u
                    if gain > best_gain:
                        best_gain = gain
                        best_swap = (u, found_v)
                
                # Restore u
                S.add(u)
                in_set[u] = True
                
            if best_swap:
                u, v = best_swap
                S.remove(u)
                in_set[u] = False
                S.add(v)
                in_set[v] = True
                best_weight += best_gain
                improved = True
                continue
        
        return S, best_weight

    # Generate initial sets
    candidates = []
    
    # 1. Greedy
    S_greedy = set()
    in_set = [False] * (N + 1)
    for x in range(1, N + 1):
        if can_add(x, S_greedy, in_set):
            S_greedy.add(x)
            in_set[x] = True
    candidates.append(S_greedy)
    
    # 2. Base 3 (0, 1)
    S_b3_01 = set()
    for x in range(1, N + 1):
        # Check base 3 representation
        n = x
        valid = True
        while n > 0:
            if n % 3 == 2:
                valid = False
                break
            n //= 3
        if valid:
            S_b3_01.add(x)
    candidates.append(S_b3_01)
    
    # 3. Base 3 (0, 2)
    S_b3_02 = set()
    for x in range(1, N + 1):
        n = x
        valid = True
        while n > 0:
            if n % 3 == 1:
                valid = False
                break
            n //= 3
        if valid:
            S_b3_02.add(x)
    candidates.append(S_b3_02)
    
    # 4. Random Greedy (multiple restarts)
    random.seed(42)
    for _ in range(5):
        S_rand = set()
        in_set = [False] * (N + 1)
        items = list(range(1, N + 1))
        random.shuffle(items)
        for x in items:
            if can_add(x, S_rand, in_set):
                S_rand.add(x)
                in_set[x] = True
        candidates.append(S_rand)
        
    best_S = set()
    best_w = 0.0
    
    for S_init in candidates:
        if time.time() > deadline:
            break
        S_res, w_res = local_search(S_init)
        if w_res > best_w:
            best_w = w_res
            best_S = S_res
            
    # Final check: try to add any remaining elements to best_S
    # (local search should have done this, but just in case)
    in_set = [False] * (N + 1)
    for x in best_S:
        in_set[x] = True
    for v in range(1, N + 1):
        if not in_set[v]:
            if can_add(v, best_S, in_set):
                best_S.add(v)
                in_set[v] = True
                
    return list(best_S)
