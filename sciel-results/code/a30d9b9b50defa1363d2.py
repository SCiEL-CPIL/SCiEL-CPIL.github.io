import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 5s margin
    
    N = 128
    numbers = list(range(1, N + 1))
    
    # Precompute LCMs
    # lcm[i][j] for 1 <= i,j <= N
    # Using math.lcm
    from math import lcm
    
    # Precompute bad triples and conflicts
    # conflicts[x] is a set of frozensets or tuples (a,b) with a<b
    # Actually, just tuples (a,b) is fine, order doesn't matter for set intersection if we normalize.
    # Let's store pairs as (min(a,b), max(a,b))
    
    conflicts = [set() for _ in range(N + 1)]
    
    # Iterate over all triples
    # This is O(N^3) which is 128^3 ~ 2M. Fine.
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            ab_lcm = lcm(a, b)
            for c in range(b + 1, N + 1):
                # Check if lcm(a,b) == lcm(a,c) == lcm(b,c)
                # Optimization: if ab_lcm > c, then lcm(a,c) >= c? No.
                # Just compute.
                ac_lcm = lcm(a, c)
                if ac_lcm != ab_lcm:
                    continue
                bc_lcm = lcm(b, c)
                if bc_lcm == ab_lcm:
                    # Bad triple {a, b, c}
                    # Add to conflicts for a, b, c
                    pair_bc = (b, c)
                    pair_ac = (a, c)
                    pair_ab = (a, b)
                    
                    conflicts[a].add(pair_bc)
                    conflicts[b].add(pair_ac)
                    conflicts[c].add(pair_ab)
                    
    # Greedy Initialization
    # Sort numbers. Maybe by degree in conflict graph?
    # Degree = len(conflicts[i])
    # Lower degree is better to add first?
    order = sorted(numbers, key=lambda x: len(conflicts[x]))
    
    S = set()
    pairs_in_S = set()
    
    for x in order:
        # Check if x can be added
        if not (conflicts[x] & pairs_in_S):
            S.add(x)
            # Update pairs_in_S
            for y in S:
                if y != x:
                    p = (min(x, y), max(x, y))
                    pairs_in_S.add(p)
            
    best_size = len(S)
    best_S = set(S)
    
    # Local Search
    # We will try to improve the set.
    # Strategy:
    # 1. Try to add a random element not in S.
    # 2. If fails, try to swap: remove one element from S to allow adding x.
    
    random.seed(42)
    
    while time.time() < deadline:
        # Try to add
        # Pick a random candidate not in S
        if len(S) < N:
            candidates = [x for x in numbers if x not in S]
            random.shuffle(candidates)
            added = False
            for x in candidates:
                if not (conflicts[x] & pairs_in_S):
                    # Can add
                    S.add(x)
                    for y in S:
                        if y != x:
                            p = (min(x, y), max(x, y))
                            pairs_in_S.add(p)
                    added = True
                    if len(S) > best_size:
                        best_size = len(S)
                        best_S = set(S)
                    break
            
            if added:
                continue
        
        # Try to swap
        # Pick a random x not in S
        if len(S) < N:
            candidates = [x for x in numbers if x not in S]
            random.shuffle(candidates)
            
            swapped = False
            for x in candidates:
                # Find conflicts in S
                conf_in_S = conflicts[x] & pairs_in_S
                if not conf_in_S:
                    # Should have been added in previous step, but maybe logic gap?
                    # Actually if conf_in_S is empty, we add.
                    S.add(x)
                    for y in S:
                        if y != x:
                            p = (min(x, y), max(x, y))
                            pairs_in_S.add(p)
                    if len(S) > best_size:
                        best_size = len(S)
                        best_S = set(S)
                    swapped = True
                    break
                
                # Try to resolve conflicts by removing an element
                # Pick a random pair from conf_in_S
                # The pair (a,b) is a conflict. We must remove a or b.
                # We try to remove one and check if the rest is valid.
                
                # To speed up, we can pick the pair with the element that appears in most conflicts?
                # For now, random pair.
                pair = random.choice(list(conf_in_S))
                a, b = pair
                
                # Try removing a
                # New pairs_in_S would be pairs_in_S - {p | a in p}
                # We need to check if conflicts[x] intersects with (pairs_in_S - pairs_with_a)
                
                # Optimization:
                # Let C = conflicts[x]
                # Let P = pairs_in_S
                # We remove a. New P' = P \ { (min(a,y), max(a,y)) for y in S\{a} }
                # Check if C & P' is empty.
                
                # This check is O(|P|) if we iterate.
                # But we can precompute for each element a, the set of pairs involving a?
                # Or just iterate through conf_in_S?
                # If we remove a, any pair in conf_in_S that contains a is "satisfied" (removed).
                # Any pair in conf_in_S that does NOT contain a must be removed by removing b?
                # No, we only remove a.
                # So if there is ANY pair in conf_in_S that does not contain a, it remains in P'.
                # Thus, removing a is valid iff ALL pairs in conf_in_S contain a.
                
                # Wait, this logic is flawed.
                # P' contains ALL pairs from P that do not involve a.
                # conf_in_S is a subset of P.
                # If there is a pair (u,v) in conf_in_S such that u != a and v != a, then (u,v) is in P'.
                # Then C & P' is not empty.
                # So removing a is valid iff every pair in conf_in_S contains a.
                
                # Similarly, removing b is valid iff every pair in conf_in_S contains b.
                
                # Let's check this condition.
                valid_remove_a = all(a in p for p in conf_in_S)
                valid_remove_b = all(b in p for p in conf_in_S)
                
                if valid_remove_a:
                    # Remove a, add x
                    # Update S and pairs_in_S
                    S.remove(a)
                    # Remove pairs involving a
                    to_remove = [p for p in pairs_in_S if a in p]
                    for p in to_remove:
                        pairs_in_S.remove(p)
                    
                    S.add(x)
                    for y in S:
                        if y != x:
                            p = (min(x, y), max(x, y))
                            pairs_in_S.add(p)
                            
                    if len(S) > best_size:
                        best_size = len(S)
                        best_S = set(S)
                    swapped = True
                    break
                elif valid_remove_b:
                    # Remove b, add x
                    S.remove(b)
                    to_remove = [p for p in pairs_in_S if b in p]
                    for p in to_remove:
                        pairs_in_S.remove(p)
                    
                    S.add(x)
                    for y in S:
                        if y != x:
                            p = (min(x, y), max(x, y))
                            pairs_in_S.add(p)
                            
                    if len(S) > best_size:
                        best_size = len(S)
                        best_S = set(S)
                    swapped = True
                    break
                else:
                    # Try another pair from conf_in_S?
                    # If we pick a different pair (c,d), maybe removing c works?
                    # The condition "all pairs contain c" must hold.
                    # Let's collect all elements that are in ALL pairs of conf_in_S.
                    # If such an element exists, removing it works.
                    # Intersection of all sets in conf_in_S.
                    if conf_in_S:
                        common_elements = set(conf_in_S[0])
                        for p in conf_in_S:
                            common_elements.intersection_update(p)
                        
                        # Try removing any element in common_elements
                        for elem in common_elements:
                            # Remove elem, add x
                            S.remove(elem)
                            to_remove = [p for p in pairs_in_S if elem in p]
                            for p in to_remove:
                                pairs_in_S.remove(p)
                            
                            S.add(x)
                            for y in S:
                                if y != x:
                                    p = (min(x, y), max(x, y))
                                    pairs_in_S.add(p)
                            
                            if len(S) > best_size:
                                best_size = len(S)
                                best_S = set(S)
                            swapped = True
                            break
                        if swapped:
                            break

        # If no swap or add happened, we might be stuck in a local optimum.
        # We can try a "kick": remove a random element and try to re-add it or others.
        # Or just continue loop, random shuffle of candidates might find a way.
        # To avoid infinite loop on same state, we can track state?
        # Given time limit, just continuing is probably fine.

    return list(best_S)
