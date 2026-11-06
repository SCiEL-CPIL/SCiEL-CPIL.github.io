import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin
    
    # Precompute all forbidden triples
    # A triple (a,b,c) with a<b<c is forbidden if 1/a = 1/b + 1/c
    # which means bc = a(b+c)
    triples = []  # list of (a, b, c) with a<b<c
    triple_map = defaultdict(set)  # element -> set of triples it's in
    
    for a in range(1, 181):
        for b in range(a + 1, 181):
            # 1/a = 1/b + 1/c => c = ab/(b-a)
            num = a * b
            den = b - a
            if den != 0 and num % den == 0:
                c = num // den
                if c > b and c <= 180:
                    triple = (a, b, c)
                    triples.append(triple)
                    triple_map[a].add(triple)
                    triple_map[b].add(triple)
                    triple_map[c].add(triple)
    
    # Precompute for each pair (x,y), the set of z that would form a triple
    # For efficiency, we'll use a different approach
    
    def check_conflict(x, S):
        """Check if adding x to set S creates any forbidden triple."""
        for triple in triple_map[x]:
            a, b, c = triple
            other1 = b if a == x else (a if c == x else c)
            other2 = c if a == x else (a if b == x else b)
            if other1 in S and other2 in S:
                return True
        return False
    
    def greedy_build(order):
        """Build a set greedily following the given order."""
        S = set()
        for x in order:
            if not check_conflict(x, S):
                S.add(x)
        return S
    
    # Multiple greedy strategies
    best_S = set()
    best_size = 0
    
    # Strategy 1: Start with large numbers, add small ones
    S1 = set(range(91, 181))
    candidates = list(range(1, 91))
    random.shuffle(candidates)
    for x in candidates:
        if not check_conflict(x, S1):
            S1.add(x)
    if len(S1) > best_size:
        best_size = len(S1)
        best_S = set(S1)
    
    # Strategy 2: Pure greedy from all numbers in random order
    all_nums = list(range(1, 181))
    random.shuffle(all_nums)
    S2 = greedy_build(all_nums)
    if len(S2) > best_size:
        best_size = len(S2)
        best_S = set(S2)
    
    # Strategy 3: Greedy with small numbers first
    S3 = greedy_build(list(range(1, 181)))
    if len(S3) > best_size:
        best_size = len(S3)
        best_S = set(S3)
    
    # Strategy 4: Greedy with large numbers first
    S4 = greedy_build(list(range(180, 0, -1)))
    if len(S4) > best_size:
        best_size = len(S4)
        best_S = set(S4)
    
    # Strategy 5: Multiple random restarts
    for _ in range(5):
        if time.time() > deadline - 30:
            break
        nums = list(range(1, 181))
        random.shuffle(nums)
        S5 = greedy_build(nums)
        if len(S5) > best_size:
            best_size = len(S5)
            best_S = set(S5)
    
    # Local search on best_S
    S = set(best_S)
    
    # Phase 1: Try to add any remaining elements
    remaining = [x for x in range(1, 181) if x not in S]
    for x in remaining:
        if not check_conflict(x, S):
            S.add(x)
    
    best_size = len(S)
    best_S = set(S)
    
    # Phase 2: Local search - remove one, add multiple
    while time.time() < deadline - 5:
        improved = False
        S_list = list(S)
        random.shuffle(S_list)
        
        for remove_elem in S_list:
            if time.time() > deadline - 5:
                break
            S.remove(remove_elem)
            
            # Try to add elements
            added = []
            for x in remaining:
                if x in S:
                    continue
                if not check_conflict(x, S):
                    S.add(x)
                    added.append(x)
            
            if len(added) >= 1 and len(S) > len(best_S):
                best_size = len(S)
                best_S = set(S)
                improved = True
                remaining = [x for x in range(1, 181) if x not in S]
            else:
                # Revert
                for x in added:
                    S.discard(x)
                S.add(remove_elem)
        
        if not improved:
            break
    
    # Phase 3: Pair removal - remove two, try to add three
    S = set(best_S)
    remaining = [x for x in range(1, 181) if x not in S]
    
    while time.time() < deadline - 5:
        improved = False
        S_list = list(S)
        random.shuffle(S_list)
        
        for i in range(len(S_list)):
            if time.time() > deadline - 5:
                break
            for j in range(i + 1, len(S_list)):
                if time.time() > deadline - 5:
                    break
                r1, r2 = S_list[i], S_list[j]
                S.remove(r1)
                S.remove(r2)
                
                added = []
                for x in remaining:
                    if x in S:
                        continue
                    if not check_conflict(x, S):
                        S.add(x)
                        added.append(x)
                
                if len(added) >= 3 and len(S) > len(best_S):
                    best_size = len(S)
                    best_S = set(S)
                    improved = True
                    remaining = [x for x in range(1, 181) if x not in S]
                else:
                    for x in added:
                        S.discard(x)
                    S.add(r1)
                    S.add(r2)
        
        if not improved:
            break
    
    # Final verification
    result = list(best_S)
    if not result:
        result = list(range(91, 181))
    
    return result
