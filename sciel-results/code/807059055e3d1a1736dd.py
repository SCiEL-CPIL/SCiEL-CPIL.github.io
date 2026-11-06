"""Rewritten program: structural change to branch-and-bound maximum independent set
on a conflict graph, with precomputed pairwise and subset-sum conflict detection."""
import time
import math
from itertools import combinations

def construct():
    deadline = time.time() + 88  # safety margin before 90s limit
    
    # Precompute LCM of 1..24
    def lcm(a, b):
        return a * b // math.gcd(a, b)
    
    L = 1
    for i in range(1, 25):
        L = lcm(L, i)
    
    # val[i] = L // i, so 1/i = val[i] / L
    val = {i: L // i for i in range(1, 25)}
    
    # Precompute: for each x, which subsets of elements > x can sum to val[x]?
    # We'll do this on demand but cache results.
    
    def can_form_target(target, candidates):
        """Check if target can be formed as subset sum of candidates (all positive ints).
        Returns True if achievable."""
        if not candidates:
            return target == 0
        if target <= 0:
            return target == 0
        # Filter candidates <= target
        filtered = [c for c in candidates if c <= target]
        if not filtered:
            return False
        # Quick check: if sum of all filtered < target, impossible
        if sum(filtered) < target:
            return False
        # Use bitset DP for efficiency
        # achievable is a set of reachable sums
        achievable = {0}
        for c in filtered:
            new_sums = set()
            for s in achievable:
                ns = s + c
                if ns <= target:
                    new_sums.add(ns)
            achievable |= new_sums
            if target in achievable:
                return True
        return False
    
    # Precompute conflict: for pair (i, j) with i < j, does 1/i equal sum of reciprocals
    # of a subset of {j}? That would mean val[i] == val[j], i.e., i == j. Never for distinct.
    # But for larger sets, we need subset sums of multiple elements.
    # Key insight: x is "subsumed" if 1/x = sum of 1/y for y in some subset of OTHER selected.
    # Since all y > x (reciprocals smaller), we only need y > x.
    # Actually no: any other selected element y could be < x or > x.
    # But if y < x, then 1/y > 1/x, so a single 1/y already exceeds 1/x.
    # So only elements y > x can participate in forming 1/x.
    
    # Precompute for each x: for every subset of {y in [x+1, 24]}, does it sum to val[x]?
    # This is expensive to precompute fully, but we can precompute the "minimal" conflicts.
    
    # Better approach: build a conflict graph where edge (i,j) means "if both i and j are selected,
    # AND some other selected elements, the constraint might be violated."
    # Actually the constraint is global: x is violated if ANY subset of others sums to 1/x.
    
    # Let's use a different structural approach: 
    # Branch and bound over the set, with a fast feasibility check.
    
    # Precompute: for each x, the list of elements y > x that could potentially contribute.
    # For x, only y > x matter.
    
    # Precompute val values for quick lookup
    val_list = [0] + [L // i for i in range(1, 25)]
    
    # Fast check: is set S subsum-free?
    def is_subsum_free(S):
        S_list = sorted(S)
        for i, x in enumerate(S_list):
            target = val_list[x]
            # Only elements y > x in S can contribute
            candidates = [val_list[y] for y in S_list[i+1:]]
            if not candidates:
                continue
            if can_form_target(target, candidates):
                return False
        return True
    
    # Branch and bound: try to build maximum subsum-free set
    # Use ordering: try to include elements in some order, with pruning.
    
    best_set = set()
    best_size = 0
    
    # Initial heuristic: greedy from large to small
    greedy_set = set()
    for x in range(24, 0, -1):
        if time.time() >= deadline:
            break
        cand = greedy_set | {x}
        if is_subsum_free(cand):
            greedy_set = cand
    best_set = greedy_set
    best_size = len(greedy_set)
    
    # Branch and bound
    # State: current selected set, remaining candidates to consider
    # We process elements in descending order (larger first, as they're "safer" - 
    # they can't be subsumed by smaller elements since smaller have larger reciprocals)
    
    # If we build in decreasing order: when we add x, all already-added are > x.
    # We need to check if x is subsumed by a subset of already-added elements.
    # This is a subset sum check with the already-added elements as candidates.
    
    # Let's do branch and bound in decreasing order.
    
    # Memoize can_form_target results for speedup
    _memo = {}
    
    def can_form_target_memo(target, candidates_tuple):
        key = (target, candidates_tuple)
        if key in _memo:
            return _memo[key]
        result = can_form_target(target, list(candidates_tuple))
        _memo[key] = result
        return result
    
    def branch_and_bound(selected, next_x, current_size):
        nonlocal best_set, best_size, deadline
        
        if time.time() >= deadline:
            return
        
        # Pruning: even if we add all remaining, can we beat best?
        remaining_count = next_x - 1  # elements 1..next_x-1 remain
        if current_size + remaining_count <= best_size:
            return
        
        if next_x < 1:
            return
        
        x = next_x
        
        # Option 1: include x
        # Check if x is subsumed by selected elements (all > x since we go decreasing)
        target = val_list[x]
        candidates = tuple(val_list[y] for y in selected if y > x)
        can_add = True
        if candidates:
            can_add = not can_form_target_memo(target, candidates)
        
        if can_add:
            new_selected = selected | {x}
            new_size = current_size + 1
            if new_size > best_size:
                best_set = new_selected
                best_size = new_size
            branch_and_bound(new_selected, x - 1, new_size)
        
        # Option 2: exclude x
        branch_and_bound(selected, x - 1, current_size)
    
    # Start branch and bound
    branch_and_bound(set(), 24, 0)
    
    # Additional local search: try to improve by swaps
    if time.time() < deadline:
        all_elements = set(range(1, 25))
        remaining = all_elements - best_set
        
        # Try adding each remaining element (multiple passes)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for x in sorted(remaining, reverse=True):
                if time.time() >= deadline:
                    break
                cand = best_set | {x}
                if is_subsum_free(cand):
                    best_set = cand
                    remaining = all_elements - best_set
                    improved = True
        
        # Try swaps: remove one, add another
        for _ in range(3):  # multiple passes
            if time.time() >= deadline:
                break
            improved = False
            for y in sorted(best_set):
                if time.time() >= deadline:
                    break
                temp = best_set - {y}
                for x in sorted(remaining | ({y})):
                    if x == y:
                        continue
                    if time.time() >= deadline:
                        break
                    cand = temp | {x}
                    if len(cand) > len(best_set) and is_subsum_free(cand):
                        best_set = cand
                        remaining = all_elements - best_set
                        improved = True
            if not improved:
                break
    
    # Final verification
    if not best_set:
        best_set = {24}
    
    return list(best_set)
