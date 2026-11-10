def construct():
    from fractions import Fraction
    import time
    import random

    deadline = time.time() + 50
    denominators = list(range(2, 49))
    n = len(denominators)
    fractions = [Fraction(1, d) for d in denominators]
    
    best_sum = Fraction(0)
    best_indices = []

    def evaluate(indices):
        if not indices:
            return Fraction(0), False
        s = Fraction(0)
        for i in indices:
            s += fractions[i]
        return s, s < 1

    def greedy_init():
        selected = []
        total = Fraction(0)
        available = set(range(n))
        # Sort by fraction value descending
        indices_sorted = sorted(range(n), key=lambda i: fractions[i], reverse=True)
        for i in indices_sorted:
            if total + fractions[i] < 1:
                total += fractions[i]
                selected.append(i)
        return selected, total

    # Initial greedy
    curr_indices, curr_sum = greedy_init()
    best_sum = curr_sum
    best_indices = curr_indices

    # Local Search
    # Neighbors: remove one, add one, swap one
    # We only consider moves that keep sum < 1 and increase sum
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        curr_set = set(curr_indices)
        
        # Try removing
        for i in list(curr_indices):
            new_indices = curr_indices.copy()
            new_indices.remove(i)
            new_sum, valid = evaluate(new_indices)
            if valid and new_sum > curr_sum: # This won't happen, removing decreases sum
                pass
            
        # Try adding
        for i in range(n):
            if i not in curr_set:
                new_indices = curr_indices + [i]
                new_sum, valid = evaluate(new_indices)
                if valid and new_sum > curr_sum:
                    curr_indices = new_indices
                    curr_sum = new_sum
                    if curr_sum > best_sum:
                        best_sum = curr_sum
                        best_indices = curr_indices
                    improved = True
                    break # Restart neighborhood search from new state
        
        # If no add improved, try swaps? 
        # Actually, if we can't add anything, we are stuck in a local optimum for additions.
        # We should try swaps to escape.
        if not improved:
             # Try swaps: remove i, add j
             for i in curr_indices:
                 for j in range(n):
                     if j in curr_set: continue
                     # Check if swap improves
                     # New sum = curr_sum - fractions[i] + fractions[j]
                     # Must be < 1 and > curr_sum
                     if fractions[j] > fractions[i]:
                         new_sum = curr_sum - fractions[i] + fractions[j]
                         if new_sum < 1 and new_sum > curr_sum:
                             new_indices = [x for x in curr_indices if x != i] + [j]
                             curr_indices = new_indices
                             curr_sum = new_sum
                             if curr_sum > best_sum:
                                 best_sum = curr_sum
                                 best_indices = curr_indices
                             improved = True
                             break
                 if improved: break

    return best_indices
