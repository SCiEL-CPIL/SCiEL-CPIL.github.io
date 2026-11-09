"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill
    
    n = 256
    # Build conflict graph: edge (a,b) exists if a,b and a+2(b-a) are all in [1,n]
    # i.e., a, b, c form a 3-AP with c = 2b - a, or equivalently for any pair (a,c) with a<c,
    # b = (a+c)/2 must be integer, so a+c must be even.
    # Actually, the constraint is: no three distinct elements a < b < c with a+c = 2b.
    # So two elements a, c (a < c) conflict if (a+c) is even and (a+c)//2 is also selected.
    # Wait, that's not right. The conflict is between pairs that *would complete* a 3-AP.
    # If we select a and c, we cannot select (a+c)/2 if it's an integer in [1,n].
    # So the "conflict" for independent set formulation: vertices are numbers 1..256.
    # Two vertices a, c (a != c) have an edge if selecting both would force a 3-AP.
    # But actually, the 3-AP is three elements. The independent set formulation:
    # We want a subset S such that no three elements of S form a 3-AP.
    # This is NOT simply an independent set on a pairwise conflict graph, because the 
    # constraint involves triples, not pairs.
    
    # Alternative approach: Use a different formulation.
    # Key insight: For 3-AP-free sets, if a and c are in S, then (a+c)/2 cannot be in S (if integer).
    # So we can build a conflict graph where edge(a,c) exists if a+c is even and 1 <= (a+c)//2 <= 256.
    # But wait - that's still not quite right. The issue is that the "middle" element depends on 
    # the pair. Let me think again.
    
    # Actually, for a 3-AP a, b, c with a < b < c: b = (a+c)/2.
    # So if we have a and c selected, b cannot be selected.
    # This means: for each pair (a,c) with a < c and (a+c) even, if both a and c are in S, 
    # then (a+c)//2 is not in S.
    # 
    # This is a "forbidden triple" constraint, not a simple pairwise conflict.
    # However, we can reformulate: create a conflict graph where we add edges between 
    # elements that "conflict" in the sense that they can't both be present.
    # 
    # Alternative: Think of it as - for each potential middle element b, and for each 
    # valid step d such that b-d >= 1 and b+d <= 256, the triple (b-d, b, b+d) is forbidden.
    # So b-d and b+d cannot both be in S if b is in S. But b-d and b+d can both be in S 
    # if b is not in S.
    #
    # This is a 3-uniform hypergraph independent set problem, which is NP-hard.
    # 
    # Let me try a different structural approach: local search with simulated annealing 
    # or tabu search, which is a different optimization family from the greedy construction.
    
    # Better approach: Use a smart greedy + local search with bitset operations for speed.
    # Represent the selected set as a bitset (integer with 256 bits).
    # Precompute for each number x, the set of numbers that would form a 3-AP with x 
    # and another selected number.
    
    # Let me try: greedy construction with reciprocal weighting, then local search.
    
    # Greedy: process numbers in order of decreasing 1/x (i.e., increasing x), 
    # add x if it doesn't create a 3-AP with existing elements.
    # This naturally favors small numbers (high reciprocal weight).
    
    def greedy_construct():
        selected = set()
        # Process in order of decreasing weight (1/x), i.e., x from 1 to 256
        for x in range(1, 257):
            # Check if adding x creates a 3-AP
            # x can be the first, middle, or last element of a 3-AP
            ok = True
            for y in selected:
                if y < x:
                    # x is last: check if (x-y)/2 is in selected and (x+y)/2... 
                    # Wait: 3-AP a, b, c with a < b < c, b = (a+c)/2
                    # If x is c (last): need a = 2x - y... no.
                    # If y and x are in the set, check if (y+x)/2 is in set (middle)
                    # or if 2x - y is in set (x is middle, y is first)
                    # or if 2y - x is in set (x is first, y is middle)
                    mid = (y + x) / 2
                    if mid == int(mid) and int(mid) in selected:
                        ok = False
                        break
                    if 2 * x - y in selected:
                        ok = False
                        break
                    if 2 * y - x in selected:
                        ok = False
                        break
                elif y > x:
                    mid = (x + y) / 2
                    if mid == int(mid) and int(mid) in selected:
                        ok = False
                        break
                    if 2 * y - x in selected:
                        ok = False
                        break
                    if 2 * x - y in selected:
                        ok = False
                        break
            if ok:
                selected.add(x)
        return selected
    
    # Local search: try to swap out a large element for a small one, or add/remove elements.
    # Use bitset for fast AP checking.
    
    def is_3ap_free(selected_set):
        """Check if selected_set has no 3-AP. O(k^2) where k = len(selected_set)."""
        s_list = list(selected_set)
        for a in s_list:
            for b in s_list:
                if a == b:
                    continue
                c = 2 * b - a
                if c in selected_set and c != a and c != b:
                    return False
        return True
    
    def compute_weight(selected_set):
        """Compute sum of 1/x for x in selected_set."""
        total = Fraction(0, 1)
        for x in selected_set:
            total += Fraction(1, x)
        return total
    
    def can_add(selected_set, x):
        """Check if x can be added to selected_set without creating a 3-AP."""
        for y in selected_set:
            if y == x:
                continue
            # Check all three positions
            mid = (x + y) / 2
            if mid == int(mid) and int(mid) != x and int(mid) != y and int(mid) in selected_set:
                return False
            # x is middle: need z such that z, x, y or y, x, z form AP
            # If x is middle: z = 2x - y
            z = 2*x - y
            if z != x and z != y and z in selected_set:
                return False
            # y is middle: z = 2y - x
            z2 = 2*y - x
            if z2 != x and z2 != y and z2 in selected_set:
                return False
        return True
    
    # Initial greedy solution
    best_set = greedy_construct()
    best_weight = compute_weight(best_set)
    
    # Local search: try single swaps (remove one, add one) and single additions
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        current_set = set(best_set)
        
        # Try adding each unselected number
        for x in range(1, 257):
            if x not in current_set:
                if can_add(current_set, x):
                    new_set = current_set | {x}
                    new_weight = compute_weight(new_set)
                    if new_weight > best_weight:
                        best_set = new_set
                        best_weight = new_weight
                        improved = True
        
        # Try swaps: remove y, add x
        for y in list(best_set):
            for x in range(1, 257):
                if x not in best_set:
                    trial = (best_set - {y}) | {x}
                    if is_3ap_free(trial):
                        new_weight = compute_weight(trial)
                        if new_weight > best_weight:
                            best_set = trial
                            best_weight = new_weight
                            improved = True
        
        # Try 2-for-1 swaps: remove two, add one
        if time.time() < deadline - 10:
            for y1 in list(best_set):
                if time.time() > deadline - 5:
                    break
                for y2 in list(best_set):
                    if y1 >= y2:
                        continue
                    if time.time() > deadline - 5:
                        break
                    for x in range(1, 257):
                        if x not in best_set and x != y1 and x != y2:
                            trial = (best_set - {y1, y2}) | {x}
                            if is_3ap_free(trial):
                                new_weight = compute_weight(trial)
                                if new_weight > best_weight:
                                    best_set = trial
                                    best_weight = new_weight
                                    improved = True
                                    break
                    if time.time() > deadline - 5:
                        break
    
    # Convert to sorted list
    result = sorted(best_set)
    if not result:
        return [1]
    return result
# EVOLVE-BLOCK-END
