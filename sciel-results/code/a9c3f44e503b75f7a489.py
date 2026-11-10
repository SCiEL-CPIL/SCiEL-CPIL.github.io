"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin for 90s limit
    
    N = 128
    elements = list(range(1, N + 1))
    
    # Precompute LCMs
    def get_lcm(a, b):
        return a * b // math.gcd(a, b)

    # Build conflict map: for each pair (a,b), find all c such that {a,b,c} is a bad triple
    # A bad triple means lcm(a,b) == lcm(a,c) == lcm(b,c)
    # normalized_conflict[(min(a,b), max(a,b))] = set of c
    
    normalized_conflict = {}
    
    # Efficiently build conflict map by iterating triples
    # For each pair (a,b), L = lcm(a,b). c must divide L and lcm(a,c)==L and lcm(b,c)==L
    # But it's faster to just iterate all triples and check
    
    # Precompute lcm for all pairs to avoid repeated computation
    lcm_table = {}
    for a in range(1, N + 1):
        for b in range(a, N + 1):
            lcm_table[(a, b)] = get_lcm(a, b)
    
    # Build conflict map by iterating all triples
    # This is O(N^3) = 128^3 = 2M operations, which is fast
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            L_ab = lcm_table[(a, b)]
            for c in range(b + 1, N + 1):
                if lcm_table[(a, c)] == L_ab and lcm_table[(b, c)] == L_ab:
                    # {a, b, c} is a bad triple
                    # Add c to conflict of (a,b)
                    key_ab = (a, b)
                    if key_ab not in normalized_conflict:
                        normalized_conflict[key_ab] = set()
                    normalized_conflict[key_ab].add(c)
                    
                    # Add b to conflict of (a,c)
                    key_ac = (a, c)
                    if key_ac not in normalized_conflict:
                        normalized_conflict[key_ac] = set()
                    normalized_conflict[key_ac].add(b)
                    
                    # Add a to conflict of (b,c)
                    key_bc = (b, c)
                    if key_bc not in normalized_conflict:
                        normalized_conflict[key_bc] = set()
                    normalized_conflict[key_bc].add(a)
    
    # Helper to check if adding v to S is valid
    def can_add(v, S):
        # Check all pairs in S
        # If v is in conflict of any pair (a,b) in S, it's invalid
        for a, b in combinations(S, 2):
            key = (min(a, b), max(a, b))
            if v in normalized_conflict.get(key, set()):
                return False
        return True
    
    # Helper to check if swap u -> {v1, v2} is valid
    def can_swap(u, v1, v2, S):
        # S is the current set containing u
        # New set S' = (S \ {u}) \cup {v1, v2}
        # Check triples involving v1 or v2
        S_minus_u = S - {u}
        
        # 1. Triples {v1, v2, x} for x in S_minus_u
        key_v1v2 = (min(v1, v2), max(v1, v2))
        conflict_v1v2 = normalized_conflict.get(key_v1v2, set())
        for x in S_minus_u:
            if x in conflict_v1v2:
                return False
        
        # 2. Triples {v1, x, y} for x, y in S_minus_u
        # Check if v1 is in conflict of any pair (x, y) in S_minus_u
        for x, y in combinations(S_minus_u, 2):
            key_xy = (min(x, y), max(x, y))
            if v1 in normalized_conflict.get(key_xy, set()):
                return False
        
        # 3. Triples {v2, x, y} for x, y in S_minus_u
        for x, y in combinations(S_minus_u, 2):
            key_xy = (min(x, y), max(x, y))
            if v2 in normalized_conflict.get(key_xy, set()):
                return False
                
        return True
    
    best_set = set()
    best_size = 0
    
    # Precompute a "constraint score" for each element: number of bad triples it participates in
    # This helps with ordering heuristics
    elem_constraint = [0] * (N + 1)
    for (a, b), cs in normalized_conflict.items():
        for c in cs:
            elem_constraint[a] += 1
            elem_constraint[b] += 1
            elem_constraint[c] += 1
    
    # Generate multiple orderings for greedy construction
    random.seed(42)
    orderings = []
    
    # Ordering 1: natural order
    orderings.append(list(range(1, N + 1)))
    
    # Ordering 2: reverse order
    orderings.append(list(range(N, 0, -1)))
    
    # Ordering 3: large range first
    orderings.append(list(range(65, 129)) + list(range(1, 65)))
    
    # Ordering 4: sorted by descending constraint (most constrained first)
    orderings.append(sorted(range(1, N + 1), key=lambda x: -elem_constraint[x]))
    
    # Ordering 5: sorted by ascending constraint (least constrained first)
    orderings.append(sorted(range(1, N + 1), key=lambda x: elem_constraint[x]))
    
    # Ordering 6: sorted by value descending
    orderings.append(list(range(N, 0, -1)))
    
    # Orderings 7-12: random shuffles
    for _ in range(6):
        order = list(range(1, N + 1))
        random.shuffle(order)
        orderings.append(order)
    
    # Ordering 13: large range shuffled + small range shuffled
    large = list(range(65, 129))
    small = list(range(1, 65))
    random.shuffle(large)
    random.shuffle(small)
    orderings.append(large + small)
    
    # Ordering 14: alternate large and small
    large = list(range(65, 129))
    small = list(range(1, 65))
    random.shuffle(large)
    random.shuffle(small)
    interleaved = []
    i, j = 0, 0
    while i < len(large) or j < len(small):
        if i < len(large):
            interleaved.append(large[i])
            i += 1
        if j < len(small):
            interleaved.append(small[j])
            j += 1
    orderings.append(interleaved)
    
    for order in orderings:
        if time.time() > deadline:
            break
        
        # Greedy construction
        S = set()
        for x in order:
            if time.time() > deadline:
                break
            if can_add(x, S):
                S.add(x)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S.copy()
        
        # Phase 1: Try to add more elements (repeatedly)
        improved = True
        while improved and (time.time() < deadline):
            improved = False
            for v in elements:
                if v not in S:
                    if can_add(v, S):
                        S.add(v)
                        improved = True
                        break
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S.copy()
        
        # Phase 2: 1-for-2 Swaps (replace one element with two)
        improved = True
        while improved and (time.time() < deadline):
            improved = False
            S_list = list(S)
            random.shuffle(S_list)
            for u in S_list:
                if time.time() > deadline:
                    break
                not_in_S = [x for x in elements if x not in S]
                random.shuffle(not_in_S)
                for i in range(len(not_in_S)):
                    v1 = not_in_S[i]
                    for j in range(i + 1, len(not_in_S)):
                        v2 = not_in_S[j]
                        if can_swap(u, v1, v2, S):
                            S.remove(u)
                            S.add(v1)
                            S.add(v2)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S.copy()
        
        # Phase 3: 1-for-1 Swaps (replace one element with another to enable further additions)
        if time.time() < deadline:
            improved = True
            while improved and (time.time() < deadline):
                improved = False
                S_list = list(S)
                random.shuffle(S_list)
                for u in S_list:
                    if time.time() > deadline:
                        break
                    not_in_S = [x for x in elements if x not in S]
                    random.shuffle(not_in_S)
                    for v in not_in_S:
                        # Check if swapping u for v is valid
                        S_minus_u = S - {u}
                        valid = True
                        for a, b in combinations(S_minus_u, 2):
                            key = (min(a, b), max(a, b))
                            if v in normalized_conflict.get(key, set()):
                                valid = False
                                break
                        if valid:
                            # Try the swap and see if it enables more additions
                            S_test = S_minus_u | {v}
                            # Try to add more elements
                            added_any = False
                            for w in elements:
                                if w not in S_test:
                                    if can_add(w, S_test):
                                        S_test.add(w)
                                        added_any = True
                            if len(S_test) > len(S):
                                S = S_test
                                improved = True
                                break
                    if improved:
                        break
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S.copy()
    
    return list(best_set)
# EVOLVE-BLOCK-END
