from fractions import Fraction
import random

def _check_valid(S):
    """Check if all reciprocal subset sums are distinct."""
    sums = {Fraction(0, 1)}
    for x in S:
        new_sums = set()
        for s in sums:
            new_sums.add(s + Fraction(1, x))
        if sums & new_sums:
            return False
        sums |= new_sums
    return True

def construct():
    best_set = []
    best_size = 0
    
    # 1. Deterministic Greedy: Largest First
    current_set = []
    for x in range(22, 0, -1):
        new_set = current_set + [x]
        if _check_valid(new_set):
            current_set = new_set
    if len(current_set) > best_size:
        best_size = len(current_set)
        best_set = current_set
        
    # 2. Deterministic Greedy: Smallest First
    current_set = []
    for x in range(1, 23):
        new_set = current_set + [x]
        if _check_valid(new_set):
            current_set = new_set
    if len(current_set) > best_size:
        best_size = len(current_set)
        best_set = current_set
        
    # 3. Randomized Greedy with Fill
    # Run many random permutations, greedily add, then try to fill remaining spots
    for _ in range(200):
        rand_order = list(range(1, 23))
        random.shuffle(rand_order)
        current_set = []
        for x in rand_order:
            new_set = current_set + [x]
            if _check_valid(new_set):
                current_set = new_set
                
        # Fill step: Try to add any remaining element in any order
        used = set(current_set)
        unused = [x for x in range(1, 23) if x not in used]
        changed = True
        while changed:
            changed = False
            for x in unused:
                new_set = current_set + [x]
                if _check_valid(new_set):
                    current_set = new_set
                    used.add(x)
                    unused.remove(x)
                    changed = True
                    break
                    
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set
            
    # 4. Local Search (Swaps) on the best found set
    # Try to improve best_set by swapping elements
    current_best = list(best_set)
    improved = True
    while improved:
        improved = False
        used = set(current_best)
        unused = [x for x in range(1, 23) if x not in used]
        
        # Try to add any unused
        for x in unused:
            if _check_valid(current_best + [x]):
                current_best.append(x)
                improved = True
                break
                
        if not improved:
            # Try swaps
            for i in range(len(current_best)):
                for j in range(len(unused)):
                    x_out = current_best[i]
                    x_in = unused[j]
                    trial_set = current_best[:i] + current_best[i+1:] + [x_in]
                    if _check_valid(trial_set):
                        # Check if this swap allows us to add more later?
                        # For now, just accept if it's valid. 
                        # But we need to ensure we don't loop forever.
                        # Only accept if it leads to a state where we can add more, 
                        # or just accept and rely on the 'add' loop to expand.
                        current_best = trial_set
                        improved = True
                        break
                if improved:
                    break
                    
        if len(current_best) > best_size:
            best_size = len(current_best)
            best_set = current_best
            
    return best_set
