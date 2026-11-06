import time
import numpy as np
from fractions import Fraction

def construct():
    N = 256
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin

    # Precompute all 3-APs as a set of frozensets for O(1) lookup
    # Actually, let's precompute a "forbidden" structure:
    # For each pair (a, c) that are ends of an AP, the middle is b = (a+c)//2
    # When checking if we can add x, we need to verify no AP is formed.
    
    # Strategy: maintain S as a set, and for validity check of adding x:
    # An AP is formed iff there exist a, b in S such that:
    #   x, a, b form AP (x is one end)
    #   a, x, b form AP (x is middle)
    #   a, b, x form AP (x is other end)
    
    # Equivalent: for each a in S, check if 2*a - x in S (x is middle)
    #             for each a in S, check if a + (a-x) in S i.e. 2*a-x... wait
    # Let me be more careful:
    # AP (p, q, r) with p < q < r means q - p = r - q, so 2q = p + r.
    # If we add x, a new AP can form if:
    #   x is the smallest: x, a, b with 2a = x + b => b = 2a - x, need a,b in S
    #   x is the middle: a, x, b with 2x = a + b => b = 2x - a, need a,b in S  
    #   x is the largest: a, b, x with 2b = a + x => a = 2b - x, need a,b in S
    
    # So for adding x, check:
    #   For each a in S: if (2*a - x) in S and 2*a - x != a and 2*a - x != x: AP (x, a, 2a-x)
    #   For each a in S: if (2*x - a) in S and 2*x - a != x and 2*x - a != a: AP (a, x, 2x-a)
    #   For each b in S: if (2*b - x) in S and 2*b - x != b and 2*b - x != x: AP (2b-x, b, x)
    # But cases 1 and 3 are the same check! If we iterate a over S and check if 2a-x in S,
    # that covers both "x is smallest" and "x is largest" depending on which is smaller.
    # Actually no: case 1 has x smallest, so 2a - x > a > x. Case 3 has x largest, so 2b - x < b < x.
    # These are different. But the check "for a in S, is 2a-x in S" catches both if we don't filter by order.
    # The only issue is avoiding degenerate APs where elements aren't distinct.
    
    def can_add(x, S):
        """Check if x can be added to S without creating a 3-AP."""
        for a in S:
            # Check if a is the middle: x, a, 2a-x or 2a-x, a, x
            b = 2 * a - x
            if b in S and b != a and b != x:
                return False
            # Check if a is an end and x is the other end with some middle
            # If x and a are ends, middle is (x+a)//2
            s = x + a
            if s % 2 == 0:
                mid = s // 2
                if mid in S and mid != x and mid != a:
                    return False
        return True
    
    # Greedy initialization: add elements in order of increasing index (highest weight first)
    S = set()
    for i in range(1, N + 1):
        if time.time() > deadline:
            break
        if can_add(i, S):
            S.add(i)
    
    # Local search with hill climbing
    weights = {i: 1.0 / i for i in range(1, N + 1)}
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try single additions
        candidates = sorted([i for i in range(1, N + 1) if i not in S], key=lambda x: weights[x], reverse=True)
        for i in candidates:
            if time.time() > deadline:
                break
            if can_add(i, S):
                S.add(i)
                improved = True
                break
        
        if improved:
            continue
        
        # Try single swaps: remove j, add i where weights[i] > weights[j]
        removals = sorted(list(S), key=lambda x: weights[x])  # low weight first (good to remove)
        additions = sorted([i for i in range(1, N + 1) if i not in S], key=lambda x: weights[x], reverse=True)  # high weight first
        
        found_swap = False
        for j in removals:
            if time.time() > deadline or found_swap:
                break
            for i in additions:
                if weights[i] <= weights[j]:
                    break
                # Try swap
                S.remove(j)
                if can_add(i, S):
                    S.add(i)
                    improved = True
                    found_swap = True
                    break
                else:
                    S.add(j)
        
        if improved:
            continue
        
        # Try 2-for-2 swaps: remove {j1, j2}, add {i1, i2}
        # Only if total weight increases
        S_list = list(S)
        not_in_S = [i for i in range(1, N + 1) if i not in S]
        
        # Prioritize: remove low-weight pairs, add high-weight pairs
        # But checking all pairs is O(|S|^2 * |not_in_S|^2) which is too expensive
        # Instead, try: for each pair to remove, find best 2 to add
        # Limit to a subset for feasibility
        
        if time.time() < deadline - 5.0:  # Only do 2-for-2 if we have time
            # Try removing the two lowest weight elements and adding two highest weight that fit
            S_sorted_by_w = sorted(S, key=lambda x: weights[x])
            not_in_S_sorted_by_w = sorted(not_in_S, key=lambda x: weights[x], reverse=True)
            
            # Try a limited number of 2-for-2 swaps
            max_2for2_attempts = 50
            attempts = 0
            for j1_idx in range(min(5, len(S_sorted_by_w))):
                if time.time() > deadline or found_swap:
                    break
                j1 = S_sorted_by_w[j1_idx]
                for j2_idx in range(j1_idx + 1, min(10, len(S_sorted_by_w))):
                    if time.time() > deadline or found_swap or attempts >= max_2for2_attempts:
                        break
                    j2 = S_sorted_by_w[j2_idx]
                    w_remove = weights[j1] + weights[j2]
                    
                    # Find best pair to add
                    best_pair = None
                    best_w = 0
                    for i1 in not_in_S_sorted_by_w:
                        if time.time() > deadline or found_swap:
                            break
                        if weights[i1] <= w_remove - best_w:
                            break
                        S.remove(j1)
                        S.remove(j2)
                        if can_add(i1, S):
                            for i2 in not_in_S_sorted_by_w:
                                if i2 == i1:
                                    continue
                                if weights[i1] + weights[i2] <= w_remove:
                                    break
                                if can_add(i2, S):
                                    if weights[i1] + weights[i2] > best_w:
                                        best_w = weights[i1] + weights[i2]
                                        best_pair = (i1, i2)
                                    break
                        S.add(j2)
                        S.add(j1)
                        
                        if best_pair:
                            break
                        
                    if best_pair:
                        S.remove(j1)
                        S.remove(j2)
                        S.add(best_pair[0])
                        S.add(best_pair[1])
                        improved = True
                        found_swap = True
                        break
                    attempts += 1
                    
    return list(S)
