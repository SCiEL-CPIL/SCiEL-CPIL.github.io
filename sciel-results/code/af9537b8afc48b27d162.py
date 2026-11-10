import time
import random

def construct():
    """
    Improved strategy:
    1. Multiple greedy initializations (descending, ascending, random orderings)
    2. Fast can_add using sorted list and early termination
    3. Local search with addition + 2-for-1 swaps
    4. Multiple restarts with diverse seeds
    """
    start_time = time.time()
    deadline = start_time + 55.0
    
    n = 256
    best_set = set()
    best_size = 0
    
    # Precompute: for each a, list of (b,c) pairs with a<b<=c and a | (b+c)
    # This helps us quickly check if adding x creates a violation
    # Actually, let's keep can_add as is but optimize it
    
    def can_add(x, S):
        """Check if x can be added to S without violating constraints."""
        # Convert to sorted list once
        S_list = sorted(S)
        S_set = S  # already a set
        
        # Case 1: x is 'a' (smallest). Need b,c in S with x < b <= c and x | (b+c)
        # For each b in S with b > x, check if there exists c in S with c >= b and (b+c) % x == 0
        for b in S_list:
            if b <= x:
                continue
            target_rem = (-b) % x
            # Need c in S with c >= b and c % x == target_rem
            # Since S_list is sorted, we can break early if c < b
            for c in S_list:
                if c < b:
                    continue
                if c % x == target_rem:
                    return False
                if c > x + 256:  # c is at most 256
                    break
        
        # Case 2: x is 'b' (middle). Need a,c in S with a < x <= c and a | (x+c)
        for a in S_list:
            if a >= x:
                break  # S_list is sorted, so all subsequent a >= x
            target_rem = (-x) % a
            # Need c in S with c >= x and c % a == target_rem
            for c in S_list:
                if c < x:
                    continue
                if c % a == target_rem:
                    return False
        
        # Case 3: x is 'c' (largest). Need a,b in S with a < b < x and a | (b+x)
        # Note: b must be strictly less than x since x is not in S yet
        for a in S_list:
            if a >= x:
                break
            target_rem = (-x) % a
            for b in S_list:
                if b <= a:
                    continue
                if b >= x:
                    break
                if b % a == target_rem:
                    return False
        
        return True

    def greedy_add(S, order):
        """Add elements from order to S if possible."""
        S = S.copy()
        for x in order:
            if time.time() > deadline:
                break
            if x not in S and can_add(x, S):
                S.add(x)
        return S

    # Strategy 1: Multiple greedy initializations
    # Descending
    S1 = greedy_add(set(), range(n, 0, -1))
    if len(S1) > best_size:
        best_size = len(S1)
        best_set = S1.copy()
    
    # Ascending
    S2 = greedy_add(set(), range(1, n + 1))
    if len(S2) > best_size:
        best_size = len(S2)
        best_set = S2.copy()
    
    # Random orderings
    for _ in range(3):
        if time.time() > deadline:
            break
        order = list(range(1, n + 1))
        random.shuffle(order)
        S3 = greedy_add(set(), order)
        if len(S3) > best_size:
            best_size = len(S3)
            best_set = S3.copy()

    # Strategy 2: Local search with additions and swaps
    def local_search(S, max_time=4.0):
        S = S.copy()
        local_deadline = time.time() + max_time
        improved = True
        
        while improved and time.time() < local_deadline:
            improved = False
            
            # Phase 1: Try to add any element
            candidates = [x for x in range(1, n + 1) if x not in S]
            random.shuffle(candidates)
            for x in candidates:
                if time.time() > local_deadline:
                    break
                if can_add(x, S):
                    S.add(x)
                    improved = True
                    break
            
            # Phase 2: Try 1-for-1 swaps (remove one, add one)
            if not improved and time.time() < local_deadline:
                S_list = list(S)
                random.shuffle(S_list)
                for e in S_list:
                    if time.time() > local_deadline:
                        break
                    S_temp = S.copy()
                    S_temp.remove(e)
                    # Try to add any element not in S_temp
                    candidates = [x for x in range(1, n + 1) if x not in S_temp]
                    random.shuffle(candidates)
                    for x in candidates:
                        if time.time() > local_deadline:
                            break
                        if can_add(x, S_temp):
                            S_temp.add(x)
                            if len(S_temp) > len(S):
                                S = S_temp
                                improved = True
                            break
                    if improved:
                        break
            
            # Phase 3: Try 2-for-1 swaps (remove two, add up to three)
            if not improved and time.time() < local_deadline - 1:
                S_list = list(S)
                random.shuffle(S_list)
                # Try removing pairs
                for i in range(min(len(S_list), 20)):
                    if time.time() > local_deadline:
                        break
                    for j in range(i + 1, min(len(S_list), 20)):
                        if time.time() > local_deadline:
                            break
                        e1, e2 = S_list[i], S_list[j]
                        S_temp = S.copy()
                        S_temp.remove(e1)
                        S_temp.remove(e2)
                        # Try to add elements
                        candidates = [x for x in range(1, n + 1) if x not in S_temp]
                        random.shuffle(candidates)
                        added = 0
                        for x in candidates:
                            if time.time() > local_deadline:
                                break
                            if can_add(x, S_temp):
                                S_temp.add(x)
                                added += 1
                                if added >= 3:
                                    break
                        if len(S_temp) > len(S):
                            S = S_temp
                            improved = True
                            break
                    if improved:
                        break
        
        return S

    # Run local search multiple times
    iterations = 0
    while time.time() < deadline - 3:
        iterations += 1
        
        # Start from current best
        current = local_search(best_set, max_time=3.0)
        if len(current) > best_size:
            best_size = len(current)
            best_set = current
        
        # Every few iterations, try a fresh random seed
        if iterations % 5 == 0:
            S_rand = set()
            order = list(range(1, n + 1))
            random.shuffle(order)
            for x in order[:30]:
                if time.time() > deadline:
                    break
                if can_add(x, S_rand):
                    S_rand.add(x)
            current = local_search(S_rand, max_time=3.0)
            if len(current) > best_size:
                best_size = len(current)
                best_set = current
        
        # Also try starting from different greedy seeds periodically
        if iterations % 15 == 0:
            order = list(range(1, n + 1))
            random.shuffle(order)
            S_greedy = greedy_add(set(), order)
            current = local_search(S_greedy, max_time=3.0)
            if len(current) > best_size:
                best_size = len(current)
                best_set = current

    return sorted(best_set)
