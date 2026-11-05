"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time
import numpy as np

N = 256

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 55  # safety margin before 60s kill
    
    # Precomputation:
    # For each x in [1, N], we want to know which other elements y are "blocked" by x.
    # Two elements x and y are incompatible if adding both to an empty set creates a conflict?
    # No, {x, y} is always Sidon.
    # The conflict arises when we have a set S.
    # Let's use a different perspective for the "Hitting Set" model.
    
    # A set S is Sidon iff all sums a+b (a<=b, a,b in S) are distinct.
    # A set S is Maximal iff for all x not in S, S U {x} is NOT Sidon.
    
    # "S U {x} is not Sidon" means there exists a, b, c, d in S U {x} such that a+b = c+d
    # with {a,b} != {c,d}.
    # Since S is Sidon, any conflict must involve x.
    # So, either:
    # 1. x+x = a+b for some a,b in S (a<=b)
    # 2. x+a = b+c for some a,b,c in S (with a,b,c distinct? No, a,b,c in S)
    #    Specifically, x+a = b+c.
    #    Note that if a=b, then x+a = 2a. This is covered by case 1 if we swap roles?
    #    Case 1: 2x = a+b.
    #    Case 2: x+a = b+c.
    
    # Let's define the "blocking power" of a set S on an element x.
    # x is blocked by S if:
    # (1) 2x is a sum of two elements in S.
    # (2) There exist a, b, c in S such that x + a = b + c.
    
    # We want to construct S such that ALL x not in S are blocked by S.
    # And S itself is Sidon.
    
    # Greedy Strategy:
    # 1. Start with S = {}.
    # 2. We want to pick elements for S.
    #    When we pick x, it must be compatible with current S (S U {x} is Sidon).
    #    Picking x "helps block" other elements y.
    #    How do we measure how much x helps?
    #    It's complex because blocking is a property of the whole set S.
    
    # Alternative Strategy: Reverse Greedy / Pruning
    # 1. Generate a large maximal Sidon set S0 using a simple greedy (e.g., random order).
    # 2. We want to remove elements from S0 to make it smaller, while keeping it maximal.
    # 3. An element x in S can be removed if S \ {x} is still maximal.
    #    S \ {x} is maximal if every y not in S \ {x} is blocked by S \ {x}.
    #    Note that y is already in S or not.
    #    If y is in S and y != x, y is in the new set.
    #    If y is not in S, y was blocked by S0. Is y still blocked by S0 \ {x}?
    #    If the only reason y was blocked was due to interactions with x, then removing x might unblock y.
    #    So, we can only remove x if no y (not in S \ {x}) becomes unblocked.
    
    # This "Removal" approach is essentially what the current program does, but it's slow because it checks maximality from scratch.
    # We can speed it up by tracking "witnesses" for each blocked element.
    
    # Let's try a different structural approach: **Simulated Annealing on the set of selected elements**.
    # State: Bitmask of selected elements.
    # Objective: Minimize popcount, subject to being Sidon and Maximal.
    # Penalties:
    #   - Large penalty if not Sidon.
    #   - Large penalty if not Maximal (i.e., there exists an addable element).
    #   - Small penalty for size (to minimize cardinality).
    
    # However, the space of maximal sets is disconnected.
    
    # Let's stick to the **Greedy Hitting Set** idea but implemented efficiently.
    # We want to select a set S such that every x in [1, N] \ S is "hit" (blocked).
    # A "hit" for x occurs if x conflicts with some pair in S.
    
    # Let's precompute for each x, the set of "conflict signatures".
    # Actually, let's use the removal strategy but with a much faster maximality check.
    
    # Precompute:
    # sums_mask[S] is not feasible for all S.
    
    # Let's use a fast bit-manipulation based check.
    # Represent S as a list.
    # sum_mask is an integer where bit k is 1 if k is a sum of two elements in S.
    
    # To check if x is addable to S:
    # new_sums = {2x} U {x+s for s in S}
    # Check if new_sums intersects with sum_mask[S].
    
    # To check if S is maximal:
    # For all x not in S, check if x is addable.
    
    # Optimization:
    # When we remove an element r from S, we get S'.
    # We only need to check if any x not in S' is addable to S'.
    # Most x were blocked by S. They might still be blocked by S'.
    # Only x that were blocked *exclusively* by r might become addable.
    # Identifying those is hard.
    
    # Let's try a different search: **Backtracking with Pruning for Small Maximal Sets**.
    # We search for a Sidon set S of size k that is maximal.
    # We try k = 1, 2, 3, ...
    # We use backtracking to build S.
    # Pruning: If the current partial set S_partial cannot be extended to a maximal set of size k, prune.
    # This is hard to estimate.
    
    # Let's go back to the **Removal** strategy but optimize the inner loop heavily.
    
    def get_sum_mask(selected):
        mask = 0
        n = len(selected)
        for i in range(n):
            a = selected[i]
            mask |= (1 << (2*a))
            for j in range(i+1, n):
                b = selected[j]
                mask |= (1 << (a+b))
        return mask

    def is_addable(x, selected, sum_mask):
        new_mask = 1 << (2*x)
        for a in selected:
            new_mask |= (1 << (x+a))
        return (new_mask & sum_mask) == 0

    def is_maximal(selected, sum_mask):
        sel_set = set(selected)
        for x in range(1, N+1):
            if x in sel_set:
                continue
            if is_addable(x, selected, sum_mask):
                return False
        return True

    def fast_greedy_maximal(order):
        selected = []
        sum_mask = 0
        for x in order:
            if is_addable(x, selected, sum_mask):
                selected.append(x)
                sum_mask |= (1 << (2*x))
                for a in selected[:-1]:
                    sum_mask |= (1 << (x+a))
        return selected, sum_mask

    def try_remove_optimized(selected, sum_mask):
        current_selected = selected[:]
        current_sum_mask = sum_mask
        current_set = set(current_selected)
        
        # Sort elements by a heuristic: 
        # Elements that "block" many others are harder to remove.
        # Elements that block few others are easier to remove.
        # We can estimate "blocking power" by checking how many x not in S 
        # would become addable if we removed this element.
        # This is expensive. 
        # Let's just try random removals and greedy removals.
        
        improved = True
        while improved:
            improved = False
            # Try removing each element
            # To speed up, we can shuffle the order of removal attempts
            candidates = current_selected[:]
            random.shuffle(candidates)
            
            for x in candidates:
                if x not in current_set:
                    continue
                
                new_selected = [y for y in current_selected if y != x]
                if not new_selected:
                    continue
                
                new_sum_mask = get_sum_mask(new_selected)
                
                if is_maximal(new_selected, new_sum_mask):
                    current_selected = new_selected
                    current_sum_mask = new_sum_mask
                    current_set.discard(x)
                    improved = True
                    # We can break and restart the while loop to re-evaluate
                    # or continue trying to remove more in this pass
                    # Continuing is faster if we don't re-shuffle immediately
        return current_selected

    best_result = None
    best_size = float('inf')
    
    # Run multiple iterations with different random orders
    # We also vary the initial greedy strategy
    for i in range(1000):
        if time.time() > deadline:
            break
        
        # Random shuffle
        order = list(range(1, N+1))
        random.shuffle(order)
        
        selected, sum_mask = fast_greedy_maximal(order)
        
        # Try to reduce size
        reduced = try_remove_optimized(selected, sum_mask)
        
        if len(reduced) < best_size:
            best_size = len(reduced)
            best_result = reduced[:]
            
        # Early exit if we find a very small set
        if best_size <= 10:
            break
            
    return best_result
# EVOLVE-BLOCK-END
