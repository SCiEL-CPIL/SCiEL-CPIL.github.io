"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random

def construct():
    n = 12
    m = 40
    
    # Generate all possible subsets as frozensets for fast comparison
    all_subsets = []
    for mask in range(1 << n):
        s = frozenset(i for i in range(n) if mask & (1 << i))
        all_subsets.append(s)
        
    def is_comparable(A, B):
        # Strict subset check
        if len(A) == len(B):
            return False
        if len(A) < len(B):
            return A.issubset(B)
        else:
            return B.issubset(A)
            
    # Greedy Initialization
    current_sets = []
    used = set()
    
    for _ in range(m):
        best_score = -1
        best_set = None
        for s in all_subsets:
            if s in used:
                continue
            gain = 0
            for t in current_sets:
                if is_comparable(s, t):
                    gain += 1
            if gain > best_score:
                best_score = gain
                best_set = s
        
        if best_set is not None:
            current_sets.append(best_set)
            used.add(best_set)
            
    # Local Search (Hill Climbing)
    improved = True
    max_iterations = 100 # Safety valve
    iteration = 0
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        for i in range(m):
            current_set = current_sets[i]
            best_gain = 0
            best_swap = None
            for s in all_subsets:
                if s in used:
                    continue
                # Calculate delta: (gain from s) - (loss from current_set)
                gain_s = 0
                gain_c = 0
                for j in range(m):
                    if j == i:
                        continue
                    t = current_sets[j]
                    if is_comparable(s, t):
                        gain_s += 1
                    if is_comparable(current_set, t):
                        gain_c += 1
                delta = gain_s - gain_c
                if delta > best_gain:
                    best_gain = delta
                    best_swap = s
            
            if best_swap is not None and best_gain > 0:
                used.remove(current_set)
                used.add(best_swap)
                current_sets[i] = best_swap
                improved = True
                break # Restart the for loop to re-evaluate other indices with new state
                
    result = [list(s) for s in current_sets]
    return result

# EVOLVE-BLOCK-END
