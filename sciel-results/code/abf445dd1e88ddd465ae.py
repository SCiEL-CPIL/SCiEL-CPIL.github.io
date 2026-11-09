"""
Deterministic Greedy + Systematic Hill-Climbing for reciprocal weight of 3-progression-free sets on [1,256].
Strategy:
1. Construct initial set using a greedy algorithm that prioritizes small numbers (high reciprocal weight)
   but uses a conflict graph to ensure no 3-APs.
2. Perform a systematic local search: iterate through all elements in the set, remove one, 
   then try to add the best possible candidate (highest reciprocal) that doesn't create a 3-AP.
   Repeat until no improvement is found or time limit reached.
"""
import time
import math
from fractions import Fraction

# EVOLVE-BLOCK-START

def _can_add(x, S_set):
    """
    Check if x can be added to S without creating a 3-AP.
    Optimized for speed: uses set lookups.
    """
    for a in S_set:
        # Case 1: x is the middle term. a, x, c. c = 2x - a.
        c = 2 * x - a
        if c in S_set and c != x:
            return False
        # Case 2: x is an endpoint. a, b, x or x, b, a. b = (a+x)/2.
        s_sum = a + x
        if s_sum % 2 == 0:
            b = s_sum // 2
            if b in S_set and b != x:
                return False
    return True

def _greedy_construct(n=256):
    """
    Greedy construction: Add elements in increasing order of x (decreasing 1/x).
    This maximizes the chance of including small numbers which have high weight.
    """
    S = set()
    # We iterate from 1 to n. Since 1/x decreases, this is greedy by weight.
    for x in range(1, n + 1):
        if _can_add(x, S):
            S.add(x)
    return S

def _local_search_improve(S, n=256, deadline=None):
    """
    Systematic hill climbing.
    In each pass, we try to remove one element 'r' from S and add one element 'a' not in S.
    We accept the move if the sum of reciprocals increases.
    To be efficient, we don't check all (r, a) pairs at once.
    Instead, we iterate through 'r' in S. For each 'r', we find the best 'a' that can be added to S\{r}.
    If the best 'a' gives an improvement, we make the swap and restart the pass (or continue carefully).
    
    A simpler and robust approach:
    1. Calculate current sum.
    2. For each element r in S:
       a. Temporarily remove r.
       b. Find the smallest x in [1..n] \ (S\{r}) that can be added to S\{r}.
       c. If such x exists and 1/x > 1/r, then swapping r with x improves the sum.
          We perform the swap and break to restart the loop (since S changed).
    3. If a full pass is completed without any swap, we are at a local optimum.
    
    Optimization:
    Checking "smallest x" is O(n * K) where K is size of S.
    Since n=256 and K~50-80, this is fast.
    """
    current_S = set(S)
    current_val = sum(1.0 / x for x in current_S)
    
    # To avoid infinite loops on equal value swaps, we require strict improvement.
    # Also, we should check if we can just ADD an element without removing?
    # The greedy construction produces a maximal set? Not necessarily.
    # Greedy by order 1..n ensures we try to add small ones. 
    # If we can add an element without removing, that's always an improvement (1/x > 0).
    # So first, try to add any missing element.
    
    improved = True
    while improved and (deadline is None or time.time() < deadline):
        improved = False
        
        # Phase 1: Try to add any element that fits.
        # This handles cases where the set is not maximal.
        for x in range(1, n + 1):
            if x not in current_S:
                if _can_add(x, current_S):
                    current_S.add(x)
                    current_val += 1.0 / x
                    improved = True
                    # Restart phase 1? Or just continue? 
                    # Adding one might allow adding others? No, adding constraints.
                    # But adding one might block others. 
                    # Since we iterate 1..n, we might miss a combination.
                    # But for "add only", if it fits now, it's good.
                    # Let's break and restart the whole loop to be safe with ordering.
                    break
        
        if improved:
            continue # Restart the while loop to check again
            
        # Phase 2: Try swaps.
        # We want to replace a large element r with a smaller element x.
        # Iterate r in sorted order (largest first? or smallest first?)
        # If we replace a large r with small x, gain is 1/x - 1/r.
        # We want to find if there's ANY x < r that fits in S\{r}.
        
        # To be efficient, iterate r from largest to smallest in S.
        # If we can replace a large r with a small x, it's a big gain.
        # If we can't replace a large r, maybe we can replace a smaller r?
        # Actually, if x fits in S\{r}, it might not fit in S\{r'} for r' < r.
        
        # Let's just iterate through all r in S.
        # For each r, find the smallest x in [1, n] \ current_S such that _can_add(x, current_S - {r}).
        # If found and 1/x > 1/r, swap.
        
        # Optimization: Precompute candidates?
        # Candidates are [1..n] \ current_S.
        # Sort them by x (ascending) because we want smallest x (largest 1/x).
        
        candidates = sorted([x for x in range(1, n + 1) if x not in current_S])
        
        # Iterate r in current_S. 
        # To maximize chance of finding a good swap quickly, maybe sort r by descending 1/r? 
        # i.e., largest r first. Replacing a large r is valuable.
        # But if no small x fits, we might waste time.
        
        # Let's just iterate r in arbitrary order, but check all candidates x for each r?
        # That is O(K * N * K) = O(N * K^2) per pass. 256 * 80^2 ~ 1.6M ops. Fast enough.
        
        # Better: For each r, find the BEST x (smallest) that fits.
        # We can break early from candidate list if 1/x <= 1/r (i.e., x >= r) because we only want improvement.
        
        found_swap = False
        # Sort r's to process? 
        # Let's just iterate over a copy of the set.
        for r in list(current_S):
            # Find best x to replace r
            # We want x < r.
            # Iterate candidates x in ascending order.
            for x in candidates:
                if x >= r:
                    break # No further x will be smaller than r, so no improvement possible for this r
                # Check if x can be added to current_S - {r}
                temp_S = current_S - {r}
                if _can_add(x, temp_S):
                    # Swap r with x
                    current_S.remove(r)
                    current_S.add(x)
                    current_val += 1.0 / x - 1.0 / r
                    found_swap = True
                    improved = True
                    # Update candidates list? 
                    # x is now in S, r is out.
                    # Rebuilding candidates is O(N).
                    candidates = sorted([y for y in range(1, n + 1) if y not in current_S])
                    break # Restart the r-loop? Or continue?
                    # Since S changed, other r's validity might change.
                    # Safest to break and restart the outer while loop.
            
            if found_swap:
                break # Break r-loop, go back to while loop
        
    return current_S

def construct():
    """
    Main entry point.
    """
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    # 1. Initial Construction
    # Greedy by increasing x (highest reciprocal first)
    initial_set = _greedy_construct(256)
    
    # 2. Local Search Improvement
    # The local search is deterministic and systematic.
    # It should converge quickly.
    # We pass the deadline to ensure we don't run over.
    final_set = _local_search_improve(initial_set, 256, deadline=deadline)
    
    # 3. Optional: Try a few different greedy orderings if time permits?
    # The prompt asks for a different strategy. 
    # The current strategy (SA) is stochastic. This one is deterministic + systematic.
    # To be robust, we can try a second greedy strategy: 
    # Greedy with random tie-breaking or different weight?
    # But 1/x is fixed. 
    # Maybe a "reverse greedy"? Start with empty, add smallest possible? That's what we did.
    # What if we start with a random maximal set?
    
    # Let's check time.
    if time.time() < deadline - 5.0:
        # Try a randomized local search from a different start?
        # Or just run the same local search on a slightly perturbed start?
        # Actually, the systematic local search is quite strong for this size.
        # Let's try one more variation: 
        # Greedy that picks the "best" x available at each step, where best is 1/x.
        # This is equivalent to sorting by 1/x and picking first valid.
        # Which is what _greedy_construct does (1, 2, 3...).
        
        # Let's try a different heuristic: 
        # Start with empty.
        # At each step, pick x in [1..n] not in S that maximizes 1/x subject to _can_add.
        # This is exactly what the previous step does.
        
        # What if we allow "remove and add" in a more global way?
        # The current local search does single swaps.
        
        # Let's just return the result of the systematic search.
        pass

    return list(final_set)

# EVOLVE-BLOCK-END
