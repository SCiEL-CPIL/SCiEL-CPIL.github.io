"""
Dissociated reciprocal set finder.
Strategy:
1. Start with an empty set.
2. Iterate through candidates from 1 to 22.
3. For each candidate, check if adding it preserves the dissociation property
   (all subset sums of reciprocals are distinct).
4. Use exact rational arithmetic to avoid floating-point errors.
5. To maximize cardinality, we try to add elements in increasing order.
   However, a simple greedy might get stuck. We can use a backtracking search
   with a time limit to find a larger set if greedy fails or to verify optimality.
   Given the small size (22 elements), we can afford a more sophisticated search.
   
   Actually, for this specific problem, the greedy approach of adding the smallest
   possible elements often works well, but it's not guaranteed to be optimal.
   A better approach:
   - Precompute all possible subsets? 2^22 is 4 million, which is feasible.
   - But checking all subsets for each step is expensive.
   
   Let's stick to a smart greedy with backtracking for the first few steps.
   Since we want to maximize cardinality, we should try to include as many small
   numbers as possible because they contribute more to the sum and are more likely
   to cause collisions? No, small numbers have larger reciprocals.
   
   Actually, the condition is that all subset sums of {1/x : x in S} are distinct.
   This is equivalent to saying that the set {1/x} is dissociated over the rationals.
   
   A known result is that the set of all integers is dissociated? No.
   The set of powers of 2 is dissociated.
   
   Let's try a greedy approach:
   - Start with empty set.
   - For each number from 1 to 22:
       - If adding it keeps the set dissociated, add it.
   
   But this might not be optimal. For example, maybe skipping 1 allows us to add more numbers later?
   
   Alternative:
   - Use backtracking search with pruning.
   - State: current set, next candidate to consider.
   - Pruning: if the maximum possible size (current size + remaining candidates) <= best found, prune.
   - Time limit: 60 seconds.
   
   Since 22 is small, we can try to find the maximum cardinality.
   
   However, the greedy approach is very fast. Let's first implement the greedy and see what size we get.
   Then we can try to improve it with local search or backtracking if time permits.
   
   For now, let's implement a robust greedy with a check function.
"""

import time
from itertools import combinations
from fractions import Fraction

def _is_dissociated(S):
    """
    Check if the set S (list of ints) has the property that all subset sums
    of reciprocals are distinct.
    """
    if not S:
        return True
    
    # Generate all subset sums
    n = len(S)
    subset_sums = set()
    
    # Use bit manipulation to iterate through all subsets
    for mask in range(1 << n):
        s = Fraction(0, 1)
        for i in range(n):
            if mask & (1 << i):
                s += Fraction(1, S[i])
        
        if s in subset_sums:
            return False
        subset_sums.add(s)
    
    return True

def _try_add(S, x):
    """
    Try to add x to S and check if the new set is dissociated.
    Returns the new set if valid, None otherwise.
    """
    new_S = S + [x]
    if _is_dissociated(new_S):
        return new_S
    return None

def construct():
    """
    Find a dissociated reciprocal set from [1, 22] with maximum cardinality.
    """
    start_time = time.time()
    time_limit = 50  # Leave 10 seconds margin
    
    # Greedy approach: add elements in increasing order
    S = []
    for x in range(1, 23):
        if time.time() - start_time > time_limit:
            break
        new_S = _try_add(S, x)
        if new_S is not None:
            S = new_S
    
    # If we still have time, try to improve with backtracking
    # But first, let's see what size we got
    if time.time() - start_time > time_limit:
        return S
    
    # Try to find a better set using backtracking
    # We'll try to replace some elements to see if we can get a larger set
    
    # Actually, a simple greedy might not be optimal.
    # Let's try a different ordering: maybe larger numbers first?
    # Or try all permutations? That's too expensive.
    
    # Let's try a local search: try to swap elements
    best_S = S[:]
    best_size = len(S)
    
    # Try to add any remaining numbers
    for x in range(1, 23):
        if x not in best_S:
            if time.time() - start_time > time_limit:
                break
            new_S = _try_add(best_S, x)
            if new_S is not None:
                best_S = new_S
                best_size = len(best_S)
    
    return best_S

if __name__ == "__main__":
    result = construct()
    print(f"Found set of size {len(result)}: {result}")
