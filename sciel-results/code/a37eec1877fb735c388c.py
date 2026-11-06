import time
import sys
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    n = 256
    
    # Weights: w[i] = 1/i for i in 1..n
    # We want to maximize sum of w[i] for i in S, where S is 3-AP free.
    # Since 1/i is decreasing, the "value" of a set is dominated by small integers.
    
    # Precompute conflicts for fast checking.
    # For each x, store the pairs (y, z) such that {x, y, z} is a 3-AP.
    # Actually, it's more efficient to precompute a "forbidden" mask for each x given the current set?
    # No, the standard approach is:
    # If we have a set S (bitmask), adding x is valid iff for all (y, z) in APs involving x,
    # it is not the case that both y and z are in S.
    
    # Let's precompute for each x, the list of pairs (y, z) with y < z, y != x, z != x,
    # such that {x, y, z} is a 3-AP.
    # There are O(n^2) such pairs total, but for a specific x, it's O(n).
    # Total precomputation is O(n^2).
    
    pairs = {i: [] for i in range(1, n+1)}
    for b in range(2, n):
        for a in range(1, b):
            c = 2*b - a
            if c <= n:
                # AP: a, b, c
                pairs[a].append((b, c))
                pairs[b].append((a, c))
                pairs[c].append((a, b))
    
    # To speed up checking "is x compatible with mask":
    # We can precompute for each x, a list of masks? No, mask is dynamic.
    # We can use a bitset representation for the "potential conflicts".
    
    # A faster check:
    # For x to be compatible with mask M, we need that for no (y,z) in pairs[x],
    # (M has y) AND (M has z).
    
    # Let's optimize the B&B.
    # Key insight: The weights 1/i decrease rapidly.
    # The optimal solution will likely contain most of the small numbers that don't conflict.
    # We can use a strong upper bound:
    # UB = current_sum + sum of weights of all "available" numbers.
    # "Available" means compatible with current mask.
    # But even available numbers conflict with each other.
    
    # A better UB:
    # Solve a relaxation. For example, a greedy independent set on the remaining available numbers
    # gives a lower bound, not an upper bound.
    # The sum of all available weights is an upper bound.
    
    # Let's try to improve the UB by considering that if we pick x, we might lose y and z.
    # This is essentially the MWIS problem again.
    
    # Strategy:
    # 1. Start with an empty set.
    # 2. Use Branch and Bound.
    # 3. Branch on the smallest index i not yet decided.
    # 4. Try including i (if compatible) and excluding i.
    # 5. Use Fraction for exact comparison in pruning.
    
    # To make the B&B efficient, we need a very tight UB.
    # The "sum of available weights" UB is often too loose for hard instances, but n=256 is small.
    # Also, the structure of 3-APs is somewhat sparse.
    
    # Let's add a heuristic to order the branches:
    # Always try "Include" first if it leads to a higher potential?
    # Since weights are decreasing, including smaller numbers is generally better.
    
    # We will use a global variable for the best solution found so far.
    best_score = Fraction(0)
    best_mask = 0
    
    # To avoid deep recursion and overhead, we can use an iterative deepening or just a deep recursion.
    # Depth is at most 256. Python recursion limit is usually 1000, so we are safe.
    sys.setrecursionlimit(10000)
    
    def is_compatible(x, mask):
        # Check if x can be added to the set represented by mask
        for y, z in pairs[x]:
            if (mask & (1 << (y-1))) and (mask & (1 << (z-1))):
                return False
        return True
        
    def calculate_ub(mask, start_idx, current_score):
        # Calculate an upper bound on the score achievable from start_idx onwards
        # given the current mask and score.
        # UB = current_score + sum(1/i for i in start_idx..n if i is compatible with mask)
        # This is a valid UB because any valid extension must be a subset of the compatible numbers.
        
        ub_add = Fraction(0)
        for i in range(start_idx, n+1):
            if is_compatible(i, mask):
                ub_add += Fraction(1, i)
        return current_score + ub_add

    def search(i, mask, current_score):
        nonlocal best_score, best_mask
        
        if time.time() > deadline:
            return
            
        if i > n:
            if current_score > best_score:
                best_score = current_score
                best_mask = mask
            return
            
        # Pruning check
        # If the best possible score from here is <= best_score, prune
        # We calculate UB using the sum of compatible weights.
        # This loop is O(n-i). With n=256, this is fast enough.
        ub = calculate_ub(mask, i, current_score)
        
        if ub <= best_score:
            return
            
        # Branch 1: Exclude i
        search(i+1, mask, current_score)
        
        # Branch 2: Include i
        if is_compatible(i, mask):
            new_mask = mask | (1 << (i-1))
            new_score = current_score + Fraction(1, i)
            search(i+1, new_mask, new_score)
            
    # Initial call
    search(1, 0, Fraction(0))
    
    # Convert best_mask to list
    result = []
    for i in range(1, n+1):
        if best_mask & (1 << (i-1)):
            result.append(i)
            
    if not result:
        # Fallback: just return [1]
        result = [1]
        
    return result

# EVOLVE-BLOCK-END
