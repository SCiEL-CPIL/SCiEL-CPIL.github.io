"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def construct():
    n = 256
    numbers = list(range(2, n + 1))
    num_count = len(numbers)
    
    # Precompute conflicts
    # conflict[i] is a set of indices j such that lcm(numbers[i], numbers[j]) <= 256
    conflict = [set() for _ in range(num_count)]
    for i in range(num_count):
        for j in range(i + 1, num_count):
            a, b = numbers[i], numbers[j]
            if math.lcm(a, b) <= n:
                conflict[i].add(j)
                conflict[j].add(i)
    
    weights = [1.0 / numbers[i] for i in range(num_count)]
    
    # Sort indices by weight descending (i.e., number ascending) to find good solutions early
    # This helps the branch and bound prune more effectively
    order = sorted(range(num_count), key=lambda i: -weights[i])
    
    best_weight = 0.0
    best_set = set()
    start_time = time.time()
    timeout = 55.0 # Leave 5s buffer
    
    # Map original index to position in sorted order? 
    # Actually, it's easier to work with the original indices but iterate in sorted order.
    # However, for B&B, we usually pass a list of candidates.
    # Let's use a recursive function that takes a list of candidate indices.
    
    def bbound(candidates, current_set, current_weight):
        nonlocal best_weight, best_set
        
        if time.time() - start_time > timeout:
            return
            
        if not candidates:
            if current_weight > best_weight:
                best_weight = current_weight
                best_set = set(current_set)
            return
            
        # Upper bound: current_weight + sum of weights of all candidates
        # This is a valid upper bound.
        # We can tighten it: if we pick a node, we lose its neighbors.
        # But sum of all candidates is fast to compute if we maintain it or compute on the fly.
        # Computing sum on the fly is O(N). N is small.
        upper_bound = current_weight + sum(weights[i] for i in candidates)
        
        if upper_bound <= best_weight:
            return
            
        # Pick the first candidate in the list (which is highest weight due to initial sort)
        # To make it faster, we should always pick the highest weight available node.
        # Let's pick the node with the highest weight from candidates.
        # Since candidates are passed as a list, and we remove elements, 
        # it's better to keep candidates sorted or just pick max.
        # Given N=255, picking max is O(N).
        
        # Optimization: Instead of picking max every time, we can maintain the list sorted.
        # But removing neighbors makes it unsorted.
        # Let's just pick the max weight node in candidates.
        v = max(candidates, key=lambda i: weights[i])
        
        # Branch 1: Include v
        # New candidates: those in candidates that are not v and not neighbors of v
        new_candidates_include = [u for u in candidates if u != v and u not in conflict[v]]
        bbound(new_candidates_include, current_set | {v}, current_weight + weights[v])
        
        # Check timeout
        if time.time() - start_time > timeout:
            return
            
        # Branch 2: Exclude v
        new_candidates_exclude = [u for u in candidates if u != v]
        bbound(new_candidates_exclude, current_set, current_weight)
        
    # Initialize candidates as all indices
    initial_candidates = list(range(num_count))
    bbound(initial_candidates, set(), 0.0)
    
    result = [numbers[i] for i in best_set]
    result.sort()
    
    # Fallback if B&B didn't find anything (should not happen)
    if not result:
        # Greedy fallback
        selected = set()
        for i in sorted(range(num_count), key=lambda i: -weights[i]):
            if not (conflict[i] & selected):
                selected.add(i)
        result = [numbers[i] for i in selected]
        result.sort()
        
    return result
# EVOLVE-BLOCK-END
