import sys
import time
import random

def construct():
    """
    Find a maximal Sidon set in [1, 284] containing {1, 2, 5, 11}.
    Uses backtracking with pruning and heuristics.
    """
    N = 284
    required = {1, 2, 5, 11}
    
    # Pre-compute all possible sums? 
    # Max sum is 284 + 283 = 567.
    # We can use a boolean array or set for used sums.
    
    best_set = list(required)
    best_size = len(best_set)
    
    # Initial used sums
    used_sums = set()
    req_list = sorted(required)
    for i in range(len(req_list)):
        for j in range(i, len(req_list)):
            used_sums.add(req_list[i] + req_list[j])
            
    # Candidates are numbers not in required
    candidates = [x for x in range(1, N + 1) if x not in required]
    
    # Heuristic for ordering candidates:
    # Numbers that are "far apart" might be better?
    # Or just sort them.
    # Let's sort candidates. 
    # To improve backtracking, we can sort candidates by how many "conflicts" they cause.
    # Conflict with existing set: 
    # If x is in candidates, it is valid if {x+y for y in current} U {2x} is disjoint from used_sums.
    
    # Since the search space is huge, we need a good heuristic.
    # "Least Constraining Value" heuristic: 
    # Choose the candidate that leaves the most options for the future.
    # Approximation: Choose the candidate that has the fewest conflicts with other candidates?
    
    # Let's use a time-bounded search.
    start_time = time.time()
    time_limit = 110.0 # Leave 10s for overhead
    
    def can_add(x, current_set, used_sums):
        # Check if x can be added
        # Sums to check: x+y for y in current_set, and 2x
        # Optimization: 
        # If we maintain used_sums as a set, checking is O(|current_set|)
        for y in current_set:
            if x + y in used_sums:
                return False
        if 2 * x in used_sums:
            return False
        # Also need to ensure new sums are distinct from each other?
        # x+y1 == x+y2 => y1==y2. Distinct y in set.
        # x+y == 2x => y=x. x not in set.
        # So just checking against used_sums is sufficient for validity.
        return True

    def backtrack(current_set, used_sums, candidates_remaining, depth):
        nonlocal best_set, best_size, start_time, time_limit
        
        if time.time() - start_time > time_limit:
            return
            
        # Pruning: 
        # Even if we add all remaining candidates, can we beat best_size?
        if len(current_set) + len(candidates_remaining) <= best_size:
            return
            
        # Update best
        if len(current_set) > best_size:
            best_set = current_set[:]
            best_size = len(best_set)
            
        if not candidates_remaining:
            return
            
        # Heuristic: Order candidates
        # We want to pick candidates that are "safe" and "constraining".
        # Let's filter candidates that are currently valid.
        valid_candidates = []
        for x in candidates_remaining:
            if can_add(x, current_set, used_sums):
                valid_candidates.append(x)
                
        if not valid_candidates:
            return
            
        # Sort valid candidates
        # Heuristic: Try candidates that create larger sums first? 
        # Or smaller?
        # In Sidon sets, smaller numbers tend to conflict more with each other 
        # because sums are smaller and denser.
        # Larger numbers have larger sums, which are sparser.
        # So, maybe try LARGER numbers first?
        # Let's try descending order.
        valid_candidates.sort(reverse=True)
        
        # To avoid symmetry/permutations, we don't need to enforce order on indices 
        # if we are just picking a subset. 
        # But we must ensure we don't re-explore the same set.
        # Standard backtracking: 
        # For each candidate in valid_candidates:
        #   Include it
        #   Exclude it (implicitly by moving to next)
        
        # Optimization: 
        # If we pick x, we remove x from candidates_remaining.
        # We also need to remove any candidate y that now becomes invalid?
        # No, we just check validity in the next step.
        
        for i, x in enumerate(valid_candidates):
            # Include x
            new_set = current_set + [x]
            new_sums = set(used_sums)
            for y in current_set:
                new_sums.add(x + y)
            new_sums.add(2 * x)
            
            # Remaining candidates: 
            # We can just pass the list minus x. 
            # To be efficient, we can filter out candidates that are now invalid?
            # That might be expensive. Just pass the list and let can_add filter.
            
            # To prevent trying the same combination in different orders, 
            # we should enforce an ordering. 
            # If we sort candidates, and pick from a list, we should only pick 
            # candidates that are "after" the current one in the sorted list?
            # No, that would restrict us to sorted subsets. 
            # But any subset can be sorted. 
            # So, if we enforce that we only pick from candidates > x (in value), 
            # we cover all subsets exactly once.
            
            # Wait, if we sort valid_candidates, and pick x, 
            # the next candidates should be those from valid_candidates that are 
            # "greater" than x? 
            # If we define "greater" by value, yes.
            
            # So, remaining_candidates = [y for y in valid_candidates if y > x]
            # This ensures we don't pick y < x later, which would be the same set.
            
            remaining = [y for y in valid_candidates if y > x]
            
            backtrack(new_set, new_sums, remaining, depth + 1)
            
            # Early termination if we found a perfect solution?
            # Max size is 20. If we reach 20, we can stop?
            # The problem says "Maximize the size". 
            # If we find 20, it's optimal (given the context of 20-mark ruler).
            if best_size == 20:
                return

    # Start backtracking
    # Initial candidates: all non-required
    # We should sort them? 
    # The backtracking function sorts valid_candidates internally.
    # But the initial call passes all candidates.
    
    # To improve pruning, we can sort the initial candidates in descending order?
    # The function handles sorting valid candidates.
    
    backtrack(list(required), used_sums, candidates, 0)
    
    # If the search didn't find a good solution, or timed out with a bad one,
    # we might need a fallback or just return what we have.
    
    # Ensure required elements are present
    # The backtracking starts with required, so they are always present.
    
    return sorted(best_set)
