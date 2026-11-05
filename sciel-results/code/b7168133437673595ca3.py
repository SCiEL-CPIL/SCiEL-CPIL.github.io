import time
import random

def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    
    # Precompute all forbidden triples (a, b, c) with a < b < c and 1/a = 1/b + 1/c
    # This is equivalent to (b-a)(c-a) = a^2
    forbidden = set()
    for a in range(1, 91):
        a2 = a * a
        for d in range(1, a):
            if a2 % d == 0:
                b = a + d
                c = a + a2 // d
                if b < c and b <= 180 and c <= 180:
                    forbidden.add((a, b, c))
    
    # Precompute for each element, the list of forbidden triples it's in
    triples_by_elem = {i: [] for i in range(1, 181)}
    for (a, b, c) in forbidden:
        triples_by_elem[a].append((a, b, c))
        triples_by_elem[b].append((a, b, c))
        triples_by_elem[c].append((a, b, c))
    
    def can_add_fast(x, S):
        """Faster check using precomputed triples."""
        for (a, b, c) in triples_by_elem[x]:
            if x == a:
                if b in S and c in S:
                    return False
            elif x == b:
                if a in S and c in S:
                    return False
            elif x == c:
                if a in S and b in S:
                    return False
        return True
    
    # Start with a greedy solution: include all numbers > 90, then greedily add smaller ones
    S = set(range(91, 181))
    
    for a in range(1, 91):
        if time.time() > deadline - 10:
            break
        if can_add_fast(a, S):
            S.add(a)
    
    best_S = set(S)
    best_size = len(S)
    
    # Systematic hill-climbing: try to add any excluded element
    all_nums = set(range(1, 181))
    excluded = list(all_nums - S)
    
    # Pass 1: Add all possible elements (repeat until no more can be added)
    improved = True
    while improved and time.time() < deadline - 15:
        improved = False
        new_excluded = []
        for a in excluded:
            if can_add_fast(a, S):
                S.add(a)
                improved = True
            else:
                new_excluded.append(a)
        excluded = new_excluded
        if len(S) > best_size:
            best_S = set(S)
            best_size = len(S)
    
    # Pass 2: Try swaps - for each excluded element, try removing one element from S
    random.seed(123)
    for iteration in range(50000):
        if time.time() > deadline - 5:
            break
        if not excluded:
            break
        
        a = random.choice(excluded)
        if not can_add_fast(a, S):
            # Try to find an element b in S such that removing b allows adding a
            # Focus on elements that conflict with a
            conflicting_with_a = set()
            for (x, y, z) in triples_by_elem[a]:
                if a == x:
                    if y in S:
                        conflicting_with_a.add(y)
                    if z in S:
                        conflicting_with_a.add(z)
                elif a == y:
                    if x in S:
                        conflicting_with_a.add(x)
                    if z in S:
                        conflicting_with_a.add(z)
                elif a == z:
                    if x in S:
                        conflicting_with_a.add(x)
                    if y in S:
                        conflicting_with_a.add(y)
            
            # Try removing one of the conflicting elements
            if conflicting_with_a:
                b = random.choice(list(conflicting_with_a))
                S.remove(b)
                if can_add_fast(a, S):
                    S.add(a)
                    excluded.remove(a)
                    excluded.append(b)
                    if len(S) > best_size:
                        best_S = set(S)
                        best_size = len(S)
                else:
                    S.add(b)
    
    # Pass 3: Final cleanup - add any remaining possible elements
    excluded = list(all_nums - S)
    for a in excluded:
        if can_add_fast(a, S):
            S.add(a)
    if len(S) > best_size:
        best_S = set(S)
        best_size = len(S)
    
    return sorted(best_S)
