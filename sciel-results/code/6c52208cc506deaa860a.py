import time
from math import gcd
from itertools import combinations
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # 255 = 3 * 5 * 17
    # All elements must share a factor with 255, so divisible by 3, 5, or 17
    
    # Start with multiples of 3 (size 85)
    S3 = [x for x in range(1, 256) if x % 3 == 0]
    best = set(S3)
    best_size = len(best)
    
    # Check validity of a set
    def is_valid(s):
        lst = list(s)
        for i in range(len(lst)):
            for j in range(i+1, len(lst)):
                if gcd(lst[i], lst[j]) == 1:
                    return False
        return True
    
    # Try: multiples of 5
    S5 = [x for x in range(1, 256) if x % 5 == 0]
    if len(S5) > best_size:
        best = set(S5)
        best_size = len(best)
    
    # Try: multiples of 17
    S17 = [x for x in range(1, 256) if x % 17 == 0]
    if len(S17) > best_size:
        best = set(S17)
        best_size = len(best)
    
    # Systematic approach: try to build a larger set by considering
    # elements that are multiples of (3&5), (3&17), (5&17), or (3&5&17)
    # and also pure multiples of 5 or 17 that don't conflict
    
    # Key insight: if we include any element divisible by 5 but not 3,
    # then ALL elements must share a factor with it. Since it's not div by 3,
    # they must share 5 or 17. So all elements must be divisible by 5 or 17.
    # Similarly for elements div by 17 but not 3.
    
    # Strategy: Try sets where every element is divisible by 5 or 17
    # (i.e., exclude elements that are only divisible by 3)
    # Then within that universe, find the largest pairwise non-coprime set containing 255.
    
    # Universe: all x in [1,255] with x%5==0 or x%17==0
    universe_5_17 = [x for x in range(1, 256) if x % 5 == 0 or x % 17 == 0]
    
    # Build conflict graph: edge between a,b if gcd(a,b)==1
    # We want maximum clique in the "non-conflict" graph (i.e., max subset with no conflicts)
    # Equivalently, maximum independent set in the conflict graph.
    
    # For a set of size ~60-70, we can use a greedy approach with local search.
    
    # Greedy: start with 255, then greedily add elements that conflict with nothing in current set
    def greedy_build(universe, must_include=255):
        current = {must_include}
        # Sort universe by how many conflicts they have (fewer conflicts = better to add first)
        def conflict_count(x, s):
            return sum(1 for y in s if gcd(x, y) == 1)
        
        remaining = [x for x in universe if x != must_include]
        # Greedy: repeatedly add the element with fewest conflicts
        while remaining:
            if time.time() > deadline:
                break
            best_x = None
            best_conf = float('inf')
            for x in remaining:
                c = conflict_count(x, current)
                if c < best_conf:
                    best_conf = c
                    best_x = x
            if best_x is not None and best_conf == 0:
                current.add(best_x)
                remaining.remove(best_x)
            else:
                break
        return current
    
    g_set = greedy_build(universe_5_17)
    if len(g_set) > best_size:
        best = g_set
        best_size = len(best)
    
    # Now try local search: start from best, try swaps
    if time.time() < deadline - 5:
        random.seed(42)
        current = set(best)
        
        # Precompute gcd matrix for speed
        all_candidates = [x for x in range(1, 256) if x % 3 == 0 or x % 5 == 0 or x % 17 == 0]
        cand_set = set(all_candidates)
        
        # Precompute conflicts: conflict[x] = set of y in all_candidates where gcd(x,y)==1
        conflict_map = {}
        for x in all_candidates:
            conflict_map[x] = set()
        for i in range(len(all_candidates)):
            for j in range(i+1, len(all_candidates)):
                a, b = all_candidates[i], all_candidates[j]
                if gcd(a, b) == 1:
                    conflict_map[a].add(b)
                    conflict_map[b].add(a)
        
        # Local search: try to add elements, remove conflicts, swap
        for iteration in range(5000):
            if time.time() > deadline:
                break
            
            # Try adding an element that has no conflicts with current
            not_in = [x for x in all_candidates if x not in current]
            if not_in:
                # Find elements with zero conflicts
                zero_conf = [x for x in not_in if not (conflict_map[x] & current)]
                if zero_conf:
                    x = random.choice(zero_conf)
                    new_set = current | {x}
                    if len(new_set) > len(best):
                        best = new_set
                        best_size = len(best)
                        current = new_set
                    continue
            
            # Try swap: remove one element, add one or more
            if len(current) > 2:
                removable = [x for x in current if x != 255]
                if removable:
                    y = random.choice(removable)
                    new_set = current - {y}
                    # Try to add elements that now fit
                    not_in_new = [x for x in all_candidates if x not in new_set]
                    added = 0
                    for x in not_in_new:
                        if not (conflict_map[x] & new_set):
                            new_set.add(x)
                            added += 1
                    if len(new_set) > len(best):
                        best = new_set
                        best_size = len(best)
                        current = new_set
    
    return list(best)
