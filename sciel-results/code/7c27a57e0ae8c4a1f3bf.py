import time
import random
from itertools import combinations

def construct():
    start = time.time()
    deadline = start + 85.0  # Safety margin within 90s limit
    N = 180
    
    # Precompute: for each pair (a, b) with a < b, compute c = a*b/(b-a) if integer and c > b
    # This c would form a forbidden triple (a, b, c)
    # Store as: for each c, the set of pairs (a,b) that would create a violation with c
    # Also for each a, the set of (b,c) pairs; for each b, the set of (a,c) pairs
    
    # dangerous_as_c[c] = set of (a,b) such that a<b<c and c = a*b/(b-a)
    # dangerous_as_b[b] = set of (a,c) such that a<b<c and c = a*b/(b-a)  
    # dangerous_as_a[a] = set of (b,c) such that a<b<c and c = a*b/(b-a)
    
    dangerous_as_c = [set() for _ in range(N+1)]
    dangerous_as_b = [set() for _ in range(N+1)]
    dangerous_as_a = [set() for _ in range(N+1)]
    
    for a in range(1, N+1):
        for b in range(a+1, N+1):
            denom = b - a
            num = a * b
            if num % denom != 0:
                continue
            c = num // denom
            if c > b and c <= N:
                dangerous_as_c[c].add((a, b))
                dangerous_as_b[b].add((a, c))
                dangerous_as_a[a].add((b, c))
    
    def is_valid_set(S):
        """Check if S is a valid set (no forbidden triples)."""
        S_set = set(S)
        for x in S:
            # x as c (largest)
            for (a, b) in dangerous_as_c[x]:
                if a in S_set and b in S_set:
                    return False
            # x as b (middle)
            for (a, c) in dangerous_as_b[x]:
                if a in S_set and c in S_set:
                    return False
            # x as a (smallest)
            for (b, c) in dangerous_as_a[x]:
                if b in S_set and c in S_set:
                    return False
        return True
    
    def greedy_construction(order):
        """Greedy construction following the given order of elements."""
        S = []
        S_set = set()
        for x in order:
            if time.time() > deadline:
                break
            valid = True
            # Check if adding x creates a violation
            # x as c (largest)
            for (a, b) in dangerous_as_c[x]:
                if a in S_set and b in S_set:
                    valid = False
                    break
            if not valid:
                continue
            # x as b (middle)
            for (a, c) in dangerous_as_b[x]:
                if a in S_set and c in S_set:
                    valid = False
                    break
            if not valid:
                continue
            # x as a (smallest)
            for (b, c) in dangerous_as_a[x]:
                if b in S_set and c in S_set:
                    valid = False
                    break
            if not valid:
                continue
            S.append(x)
            S_set.add(x)
        return S
    
    def local_search(S, max_steps=5000):
        """Local search: try swaps to improve the set."""
        S = list(S)
        S_set = set(S)
        current_size = len(S)
        
        # Elements not in S
        not_in_S = [x for x in range(1, N+1) if x not in S_set]
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
            
            # Try adding an element not in S
            added = False
            for x in not_in_S:
                if time.time() > deadline:
                    break
                # Check if x can be added
                valid = True
                for (a, b) in dangerous_as_c[x]:
                    if a in S_set and b in S_set:
                        valid = False
                        break
                if not valid:
                    continue
                for (a, c) in dangerous_as_b[x]:
                    if a in S_set and c in S_set:
                        valid = False
                        break
                if not valid:
                    continue
                for (b, c) in dangerous_as_a[x]:
                    if b in S_set and c in S_set:
                        valid = False
                        break
                if not valid:
                    continue
                # Add x
                S.append(x)
                S_set.add(x)
                not_in_S.remove(x)
                added = True
                break
            
            if not added:
                # Try swaps: remove one, add another
                swapped = False
                for x in not_in_S:
                    if time.time() > deadline:
                        break
                    # Check if x can be added after removing some y
                    for y in S:
                        if time.time() > deadline:
                            break
                        # Temporarily remove y
                        S_set.remove(y)
                        S.remove(y)
                        
                        # Check if x can now be added
                        valid = True
                        for (a, b) in dangerous_as_c[x]:
                            if a in S_set and b in S_set:
                                valid = False
                                break
                        if not valid:
                            # Restore y
                            S_set.add(y)
                            S.append(y)
                            continue
                        for (a, c) in dangerous_as_b[x]:
                            if a in S_set and c in S_set:
                                valid = False
                                break
                        if not valid:
                            S_set.add(y)
                            S.append(y)
                            continue
                        for (b, c) in dangerous_as_a[x]:
                            if b in S_set and c in S_set:
                                valid = False
                                break
                        if not valid:
                            S_set.add(y)
                            S.append(y)
                            continue
                        
                        # Swap succeeded
                        S.append(x)
                        S_set.add(x)
                        not_in_S.remove(x)
                        swapped = True
                        break
                    
                    if swapped:
                        break
            
            if not added and not swapped:
                break
        
        return S
    
    # Strategy 1: Greedy from large to small
    best_S = greedy_construction(range(N, 0, -1))
    
    # Strategy 2: Greedy from small to large
    if time.time() < deadline:
        S2 = greedy_construction(range(1, N+1))
        if len(S2) > len(best_S):
            best_S = S2
    
    # Strategy 3: Random orderings with local search
    rng = random.Random(42)
    for restart in range(20):
        if time.time() > deadline - 10:
            break
        
        # Create a random ordering
        order = list(range(1, N+1))
        rng.shuffle(order)
        
        S = greedy_construction(order)
        
        # Apply local search
        S = local_search(S, max_steps=2000)
        
        if len(S) > len(best_S):
            best_S = S
    
    # Final local search on the best solution found
    if time.time() < deadline - 5:
        best_S = local_search(best_S, max_steps=10000)
    
    # Ensure we return a nonempty list
    if not best_S:
        best_S = [N]
    
    return best_S
