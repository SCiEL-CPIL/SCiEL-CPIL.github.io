import time
from math import gcd
from functools import reduce
from itertools import combinations

_START = time.time()
_DEADLINE = 82.0  # Safety margin for 90s limit

def _time_ok():
    return (time.time() - _START) < _DEADLINE

def construct():
    """
    Returns a nonempty subset of [1, 22] such that all reciprocal subset sums are distinct.
    Maximizes cardinality.
    """
    # LCM of 1..22
    L = 232792560
    weights = [L // k for k in range(1, 23)]
    n = 22
    
    # Precompute all conflicts:
    # Two elements i, j conflict if there exists a set T (possibly including i,j) 
    # such that a non-trivial {-1,0,1} combination sums to 0.
    # 
    # For practical purposes, we check:
    # - Pair conflicts: w_i == w_j (none here)
    # - Triple conflicts: w_i + w_j == w_k or permutations
    # - Higher: we'll use the incremental subset-sum check during search
    #
    # But for pruning, we can precompute which triples are bad.
    
    # Build conflict graph: edge (i,j) if {i,j} is NOT dissociated
    # Actually {i,j} is always dissociated since w_i != w_j and w_i+w_j != 0
    # The real conflicts arise when we add a third element.
    
    # Better: precompute for each pair (i,j), which k's would make {i,j,k} non-dissociated
    # w_i + w_j == w_k, w_i + w_k == w_j, w_j + w_k == w_i, or w_i == w_j + w_k, etc.
    
    # Let's precompute a conflict matrix where conflict[i][j] = set of k's that 
    # create a relation with i and j.
    
    # Actually, let's use a different approach:
    # Precompute all "obstruction sets" - minimal sets that are NOT dissociated.
    # Then use hitting set / independent set logic.
    
    # For efficiency, let's just use the incremental subset-sum check but with
    # a much better search strategy:
    
    # 1. Order elements by weight descending (large weights are "safer")
    # 2. Use branch-and-bound with a good upper bound
    # 3. Use a greedy heuristic to find a good initial solution
    
    # Greedy to get initial solution
    def greedy_solution(order):
        """Greedy: add elements in given order if they maintain dissociation."""
        sums = {0}
        chosen = []
        for idx in order:
            w = weights[idx]
            # Check if adding w maintains dissociation
            new_sums = set(sums)
            valid = True
            for s in sums:
                ns = s + w
                if ns in sums:
                    valid = False
                    break
                new_sums.add(ns)
            if valid:
                sums = new_sums
                chosen.append(idx)
        return chosen
    
    # Try multiple orderings for greedy
    indices = list(range(n))
    
    # Order by weight descending
    order_desc = sorted(indices, key=lambda i: weights[i], reverse=True)
    # Order by weight ascending
    order_asc = sorted(indices, key=lambda i: weights[i])
    # Order by "conflict density" - elements that conflict with many others
    # Precompute pairwise conflict scores
    conflict_score = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            # Check if {i, j, k} is bad for any k
            wi, wj = weights[i], weights[j]
            for k in range(n):
                if k == i or k == j:
                    continue
                wk = weights[k]
                # Check all {-1,0,1} relations among wi, wj, wk
                if wi + wj == wk or wi + wk == wj or wj + wk == wi:
                    conflict_score[i] += 1
                    conflict_score[j] += 1
                    conflict_score[k] += 1
    
    # Order by conflict score descending (most constrained first)
    order_conflict = sorted(indices, key=lambda i: conflict_score[i], reverse=True)
    
    # Get best greedy solution
    best_greedy = []
    for order in [order_desc, order_asc, order_conflict]:
        sol = greedy_solution(order)
        if len(sol) > len(best_greedy):
            best_greedy = sol
    
    best_size = len(best_greedy)
    best_set = best_greedy[:]
    
    # Now use backtracking to try to improve
    # Use the greedy solution to set an initial lower bound
    
    def backtrack(current_sums, chosen, remaining, depth):
        nonlocal best_size, best_set
        
        if not _time_ok():
            return
        
        if len(chosen) > best_size:
            best_size = len(chosen)
            best_set = chosen[:]
            
        if not remaining:
            return
            
        # Pruning: even if we add all remaining, can we beat best_size?
        if len(chosen) + len(remaining) <= best_size:
            return
        
        # Try adding each element from remaining
        for i in range(len(remaining)):
            if not _time_ok():
                return
            
            idx = remaining[i]
            w = weights[idx]
            
            # Check if we can add this element
            new_sums = set(current_sums)
            valid = True
            for s in current_sums:
                ns = s + w
                if ns in current_sums:
                    valid = False
                    break
                new_sums.add(ns)
            
            if not valid:
                continue
                
            # Recurse
            new_chosen = chosen + [idx]
            new_remaining = remaining[i+1:]
            
            backtrack(new_sums, new_chosen, new_remaining, depth + 1)
    
    # Start backtracking with the greedy ordering
    backtrack({0}, [], order_desc, 0)
    
    # If we found a set, convert indices back to values (1..22)
    if best_set:
        result = sorted([idx + 1 for idx in best_set])
        return result
    else:
        # Fallback: single element is always valid
        return [1]

# Verification logic
def verify_dissociated(selected_values):
    L = 232792560
    weights = [L // k for k in selected_values]
    
    sums = set()
    n = len(weights)
    for mask in range(1 << n):
        s = 0
        for b in range(n):
            if mask & (1 << b):
                s += weights[b]
        if s in sums:
            return False
        sums.add(s)
    return True

_original_construct = construct

def construct():
    res = _original_construct()
    if not verify_dissociated(res):
        # Fallback to a known valid set
        return [1, 2, 3, 5, 7, 11, 13, 17, 19]
    return res
