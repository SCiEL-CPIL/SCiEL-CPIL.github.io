import time
import random
from fractions import Fraction

def construct():
    N = 256
    # Precompute APs
    ap_list = []
    aps_containing = {i: [] for i in range(1, N+1)}
    
    for b in range(2, N):
        min_a = max(1, 2*b - N)
        for a in range(min_a, b):
            c = 2*b - a
            ap = (a, b, c)
            ap_list.append(ap)
            aps_containing[a].append(ap)
            aps_containing[b].append(ap)
            aps_containing[c].append(ap)
            
    # Helper to check if adding y to S (which is a set) creates a 3-AP
    def can_add(S, y):
        for ap in aps_containing[y]:
            a, b, c = ap
            if y == a:
                if b in S and c in S:
                    return False
            elif y == b:
                if a in S and c in S:
                    return False
            else: # y == c
                if a in S and b in S:
                    return False
        return True
        
    # Initial Greedy (ascending)
    S = set()
    for x in range(1, N + 1):
        if can_add(S, x):
            S.add(x)
            
    best_S = set(S)
    best_score = sum(Fraction(1, x) for x in S)
    
    start_time = time.time()
    timeout = 85.0 # Safety margin for 90s limit
    
    # Local Search with Hill Climbing
    # We use a bitmask or a boolean array for fast membership checks if needed, 
    # but sets are fine for N=256.
    
    # To speed up, we can precompute the list of candidates for each x that could replace it?
    # Or just iterate.
    
    # We will implement a more aggressive local search:
    # 1. Try to add any element.
    # 2. Try to swap any pair (x in S, y not in S) such that 1/y > 1/x (i.e., y < x).
    #    Actually, we should try all swaps that improve the score.
    #    Since 1/y > 1/x iff y < x, we only care about y < x.
    
    improved = True
    while improved and (time.time() - start_time) < timeout:
        improved = False
        
        # Phase 1: Add
        # Try to add smallest y first (highest gain)
        for y in range(1, N + 1):
            if y not in S:
                if can_add(S, y):
                    S.add(y)
                    improved = True
                    break
        
        # Phase 2: Swap
        # We want to find a swap (remove x, add y) such that y < x.
        # This increases the sum of reciprocals.
        # We should prioritize swaps with the largest gain (1/y - 1/x).
        # So we should iterate y from 1 upwards, and for each y, try to find an x > y to remove.
        # If we remove the "least useful" x (largest x), we preserve the set size while maximizing the gain?
        # Actually, removing a large x is "cheap" in terms of reciprocal loss.
        # So for a fixed y, we should try to remove the largest x in S such that S\{x} U {y} is valid.
        
        swapped = False
        for y in range(1, N + 1):
            if y in S:
                continue
            if (time.time() - start_time) > timeout:
                break
            
            # Try to find an x in S such that x > y and S' = S - {x} + {y} is valid.
            # We prefer to remove the largest x to minimize the loss 1/x.
            # So iterate x in sorted(S, reverse=True)
            # But we must check validity.
            
            # Optimization: 
            # If we add y, we must not form an AP.
            # The APs containing y are fixed.
            # For each AP (a,b,c) containing y, the other two elements must not both be in S\{x}.
            # This means for each AP containing y, at least one of the other two elements must be x.
            # So x must be in the "conflict set" of y with respect to S.
            # Conflict set C_y = { z in S | there exists an AP containing y and z, and the third element is in S }.
            # Wait, that's not quite right.
            # Condition: For all APs (u, v, w) containing y, it is NOT the case that (other1 in S and other1 != x) and (other2 in S and other2 != x).
            # This is equivalent to: For all APs containing y, at least one of the other two elements is NOT in S\{x}.
            # i.e., (other1 not in S) or (other1 == x) or (other2 not in S) or (other2 == x).
            
            # Let's just check validity directly for candidate x's.
            # To be efficient, we can identify which x's are "candidates" for removal.
            # If we remove x, it can only resolve conflicts for APs containing x.
            # If y does not conflict with S at all, we can just add y (handled in Phase 1).
            # If y conflicts with S, we must remove at least one element from each conflicting AP.
            
            # Let's collect the set of x's in S that are involved in conflicts with y.
            # If we remove x, it helps if x is in a conflicting AP.
            # If we remove an x that is NOT in any conflicting AP, it doesn't help resolve the conflict, 
            # unless removing it breaks an AP in a more complex way? No, APs are local.
            # So x MUST be in the union of the other elements of all APs containing y that are also in S.
            
            candidate_xs = set()
            for ap in aps_containing[y]:
                a, b, c = ap
                others = [z for z in (a,b,c) if z != y]
                if others[0] in S and others[1] in S:
                    # This AP is a conflict. We must remove either others[0] or others[1].
                    candidate_xs.add(others[0])
                    candidate_xs.add(others[1])
            
            if not candidate_xs:
                # No conflicts, should have been added in Phase 1.
                continue
                
            # We want to remove an x from candidate_xs such that x > y (to gain score) 
            # and the resulting set is valid.
            # We prefer the largest x to minimize loss.
            # So sort candidate_xs descending.
            for x in sorted(candidate_xs, reverse=True):
                if x <= y:
                    continue
                # Check if removing x makes y addable.
                # We can use a fast check:
                # For each AP containing y, check if the other two are in S\{x}.
                valid_swap = True
                for ap in aps_containing[y]:
                    a, b, c = ap
                    if y == a:
                        if (b in S and b != x) and (c in S and c != x):
                            valid_swap = False
                            break
                    elif y == b:
                        if (a in S and a != x) and (c in S and c != x):
                            valid_swap = False
                            break
                    else: # y == c
                        if (a in S and a != x) and (b in S and b != x):
                            valid_swap = False
                            break
                
                if valid_swap:
                    S.remove(x)
                    S.add(y)
                    improved = True
                    swapped = True
                    break
            
            if swapped:
                break
                
        current_score = sum(Fraction(1, x) for x in S)
        if current_score > best_score:
            best_score = current_score
            best_S = set(S)
            
    if not best_S:
        return [1]
    return list(best_S)
