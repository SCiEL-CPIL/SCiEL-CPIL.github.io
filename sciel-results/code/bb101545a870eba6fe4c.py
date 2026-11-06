import time
from fractions import Fraction

def construct():
    deadline = time.time() + 55  # Safety margin before 60s limit
    
    denoms = list(range(2, 49))
    N = len(denoms)
    fracs = [Fraction(1, d) for d in denoms]
    
    # We will search in descending order of fraction value (ascending denominator).
    # This allows us to fill up the sum quickly, and the "exclude" branch (which keeps the sum lower) 
    # is less likely to be pruned immediately if we are close to 1, but more importantly,
    # taking large items first constrains the space of "what can we add" more tightly.
    
    # Precompute suffix sums for pruning
    suffix = [Fraction(0)] * (N + 1)
    for i in range(N - 1, -1, -1):
        suffix[i] = suffix[i + 1] + fracs[i]
        
    best_sum = Fraction(0)
    best_idxs = []
    
    # Iterative DFS to avoid recursion depth issues and allow for faster loop overhead
    # Stack entries: (idx, curr_sum, curr_idxs)
    # We use a list as a stack.
    # To optimize, we can represent the state more compactly if needed, but 47 items is small.
    
    stack = [(0, Fraction(0), [])]
    
    while stack:
        if time.time() > deadline:
            break
            
        idx, curr_sum, curr_idxs = stack.pop()
        
        if idx == N:
            if curr_sum > best_sum:
                best_sum = curr_sum
                best_idxs = curr_idxs[:]
            continue
            
        # Pruning:
        # 1. If we take all remaining, can we beat best?
        #    curr_sum + suffix[idx] <= best_sum -> prune
        if curr_sum + suffix[idx] <= best_sum:
            continue
            
        # 2. If we take current item, is it valid (< 1)?
        #    If curr_sum + fracs[idx] >= 1, we cannot take it.
        
        # Option 1: Skip current item
        # Push to stack. Note: we want to explore "Take" first to find good solutions quickly?
        # Or "Skip" first? 
        # Usually, to maximize sum, we want to try taking large items first.
        # So we push "Skip" first, then "Take", so "Take" is popped first (LIFO).
        
        # Skip
        stack.append((idx + 1, curr_sum, curr_idxs))
        
        # Take
        new_sum = curr_sum + fracs[idx]
        if new_sum < 1:
            new_idxs = curr_idxs + [idx]
            # Prune before pushing? 
            # If new_sum + suffix[idx+1] <= best_sum, no need to push
            if new_sum + suffix[idx + 1] > best_sum:
                stack.append((idx + 1, new_sum, new_idxs))
            elif new_sum > best_sum:
                # Even if we can't improve further by adding more, this specific sum is better
                best_sum = new_sum
                best_idxs = new_idxs
        
        # Update best if current state (without taking more) is better
        if curr_sum > best_sum:
            best_sum = curr_sum
            best_idxs = curr_idxs[:]
            
    # Local search / refinement?
    # The DFS should be quite effective with the suffix pruning.
    # However, Fraction arithmetic is slow.
    # Let's see if we can improve speed.
    # The number of states might be large.
    # N=47.
    
    if not best_idxs:
        return [0]
        
    return best_idxs
