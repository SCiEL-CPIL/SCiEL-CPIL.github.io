import time
import random
import numpy as np

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill

    # Precompute conflict graph:
    # For each element a in [1, 256], find all pairs (b, c) with a < b <= c <= 256
    # such that a divides (b + c).
    # We model this as: if we include a, b, c all in our set, that's a violation.
    # But we want a set S such that NO such triple exists.
    # This is a 3-uniform hypergraph independent set problem.
    #
    # Alternative approach: Think of it as a graph where we mark "conflict pairs"
    # For a fixed a, if b and c are in S and a is in S, and a | (b+c), that's bad.
    # 
    # Let's think differently. For each element x, define which other elements it "conflicts" with.
    # Actually, the constraint is: for ALL a < b <= c in S, a does NOT divide b+c.
    # 
    # Let's precompute for each pair (b, c) with b <= c, the set of a's that divide b+c and a < b.
    # If we include b and c in S, then we CANNOT include any such a.
    #
    # This suggests a different perspective: build a conflict structure where including 
    # certain elements forces exclusion of others.
    
    # Let's use a greedy approach with random restarts and local search.
    # Key insight: Start with a large set and try to add elements, using a fast validity check.
    
    # Precompute: for each x in [1, 256], which values of (b+c) are multiples of x?
    # For fast checking when adding x:
    # - x as 'a': need to check if any b,c in S with b,c > x have b+c divisible by x
    # - x as 'b': need to check if any a,c in S with a < x <= c have x+c divisible by a
    # - x as 'c': need to check if any a,b in S with a < b <= x have b+x divisible by a
    
    # Let's precompute a bitset representation for each element to speed up checks.
    # For each x, precompute a bitmask of all y such that there exists a conflict where x and y are involved.
    # This is complex because conflicts are 3-way.
    
    # Instead, let's use a different strategy: 
    # Precompute for each pair (b, c) with b <= c, the list of a's (a < b) that divide b+c.
    # Then, when checking if we can add element x to set S:
    # Case 1: x plays role of 'a'. Need to check all pairs (b, c) in S with b, c > x.
    #          If any such pair has x dividing b+c, conflict.
    # Case 2: x plays role of 'b'. Need to check all a in S with a < x, and all c in S with c >= x.
    #          If any such triple has a dividing x+c, conflict.
    # Case 3: x plays role of 'c'. Need to check all pairs (a, b) in S with a < b <= x.
    #          If any such pair has a dividing b+x, conflict.
    
    # To make this fast, precompute for each x a bitmask of "conflict indicators".
    # Actually, let's just use sorted lists and do efficient checks.
    
    # Strategy: 
    # 1. Start with S = {171, ..., 256} (known valid, size 86)
    # 2. Try to add elements from {1, ..., 170} in random order
    # 3. If an element can be added without creating violations, add it
    # 4. If stuck, do local search: try removing one element and adding another
    # 5. Repeat with random restarts
    
    def count_violations(S_list):
        """Count violations in sorted list S_list."""
        cnt = 0
        m = len(S_list)
        for i in range(m):
            a = S_list[i]
            for j in range(i + 1, m):
                b = S_list[j]
                for k in range(j, m):
                    c = S_list[k]
                    if (b + c) % a == 0:
                        cnt += 1
        return cnt

    def is_valid_full(S):
        """Check if set S is valid."""
        S_list = sorted(S)
        m = len(S_list)
        for i in range(m):
            a = S_list[i]
            for j in range(i + 1, m):
                b = S_list[j]
                for k in range(j, m):
                    c = S_list[k]
                    if (b + c) % a == 0:
                        return False
        return True

    def can_add(x, S):
        """Check if x can be added to set S without creating violations. S is a set."""
        # Case 1: x is 'a' (smallest in the triple)
        # Need b, c in S with b > x, c >= b, and x | (b+c)
        greater = [v for v in S if v > x]
        for i in range(len(greater)):
            b = greater[i]
            for j in range(i, len(greater)):
                c = greater[j]
                if (b + c) % x == 0:
                    return False
        
        # Case 2: x is 'b' (middle in the triple)
        # Need a in S with a < x, and c in S with c >= x, and a | (x+c)
        less = [v for v in S if v < x]
        geq = [v for v in S if v >= x]
        for a in less:
            for c in geq:
                if (x + c) % a == 0:
                    return False
        
        # Case 3: x is 'c' (largest in the triple)
        # Need a, b in S with a < b <= x, and a | (b+x)
        leq = [v for v in S if v <= x]
        for i in range(len(leq)):
            a = leq[i]
            for j in range(i + 1, len(leq)):
                b = leq[j]
                if (b + x) % a == 0:
                    return False
        
        return True

    best_S = None
    best_size = 0

    # Try multiple strategies with random restarts
    for restart in range(100):
        if time.time() > deadline - 5:
            break
        
        random_seed = random.randint(0, 10**9)
        random.seed(random_seed)
        
        # Start with a base set - use {171..256} or a random valid subset
        S = set(range(171, n + 1))
        current_size = len(S)
        
        if current_size > best_size:
            best_size = current_size
            best_S = set(S)
        
        # Try to add elements in random order
        candidates = list(range(1, 171))
        random.shuffle(candidates)
        
        added_any = True
        while added_any and time.time() < deadline - 5:
            added_any = False
            for x in candidates:
                if x in S:
                    continue
                if can_add(x, S):
                    S.add(x)
                    current_size += 1
                    added_any = True
                    if current_size > best_size:
                        best_size = current_size
                        best_S = set(S)
        
        # Local search: try swaps to potentially unlock more additions
        for _ in range(50):
            if time.time() > deadline - 5:
                break
            if not S:
                break
            
            # Try a random swap
            y = random.choice(list(S))
            S.remove(y)
            
            added_any = False
            for x in candidates:
                if x in S:
                    continue
                if can_add(x, S):
                    S.add(x)
                    added_any = True
                    if len(S) > best_size:
                        best_size = len(S)
                        best_S = set(S)
                    break
            
            if not added_any:
                S.add(y)  # put back
            else:
                # Continue trying to add more
                for x in candidates:
                    if x in S:
                        continue
                    if can_add(x, S):
                        S.add(x)
                        if len(S) > best_size:
                            best_size = len(S)
                            best_S = set(S)
        
        # Try a different starting point for next restart
        if restart % 3 == 1:
            # Try starting with a slightly different range
            S = set(range(180, n + 1))
        elif restart % 3 == 2:
            S = set(range(160, n + 1))
    
    if best_S is None or len(best_S) == 0:
        # Fallback: return the known valid set
        return list(range(171, n + 1))
    
    return list(best_S)

# EVOLVE-BLOCK-END
