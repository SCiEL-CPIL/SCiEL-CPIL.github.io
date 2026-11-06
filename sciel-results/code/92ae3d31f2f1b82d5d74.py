"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import numpy as np
from itertools import combinations_with_replacement

def construct():
    n_max = 4096
    target_size = 80
    
    # Conflict graph approach:
    # Two numbers x and y conflict if there exist a, b, c, d in {x, y} such that
    # the triple sums created by including both x and y collide with some other triple sum.
    # More precisely, x and y conflict if there exist i,j,k,l in {x,y} (with repetition allowed)
    # such that x+i+j = y+k+l and {x,i,j} != {y,k,l} as multisets.
    
    # Actually, the conflict is: when we have a set S, adding x creates new sums.
    # x conflicts with y if for some existing elements, the new sums from x overlap with sums from y.
    
    # Simpler: precompute for each pair (x,y) whether they can coexist in a B_3[1] set.
    # x and y are compatible if no triple sum involving x equals any triple sum involving y
    # where the multisets are different.
    
    # For two distinct numbers x < y, they conflict if there exist a,b,c,d in {x,y}
    # such that x+a+b = y+c+d and the multiset {x,a,b} != {y,c,d}.
    # Since a,b,c,d can only be x or y, we have limited cases.
    
    # Let's enumerate: possible triple sums involving x (with other elements being x or y):
    # 3x, 2x+y, x+2y, 3y
    # Possible triple sums involving y:
    # 3y, 2y+x, y+2x, 3x
    # These are the same set! So x and y don't directly conflict with each other
    # in terms of pairwise sums.
    
    # The real conflict arises when we have 3+ elements. We need a different approach.
    
    # Let me think differently. Use the incremental approach but with better heuristics.
    # The key insight: numbers that are far apart are less likely to cause collisions.
    # Use a greedy approach with multiple random orderings and keep the best.
    
    best_set = []
    
    # Pre-compute: for efficiency, use the incremental sum tracking approach
    # but with smarter candidate ordering.
    
    def greedy_build(order, target):
        S = []
        existing_sums = set()
        
        for x in order:
            if len(S) >= target:
                break
            
            # Generate all new triple sums involving x and elements of S
            new_sums = []
            new_sums.append(3 * x)
            
            for s in S:
                new_sums.append(2 * x + s)
            
            for i in range(len(S)):
                si = S[i]
                for j in range(i, len(S)):
                    new_sums.append(x + si + S[j])
            
            # Check for collision
            collision = False
            for ns in new_sums:
                if ns in existing_sums:
                    collision = True
                    break
            
            if collision:
                continue
            
            S.append(x)
            existing_sums.update(new_sums)
        
        return S
    
    # Multiple strategies with different random seeds
    num_trials = 30
    
    for trial in range(num_trials):
        candidates = list(range(1, n_max + 1))
        
        if trial % 3 == 0:
            # Pure random
            random.seed(trial * 7 + 1)
            random.shuffle(candidates)
        elif trial % 3 == 1:
            # Random with bias toward spread-out numbers
            random.seed(trial * 13 + 3)
            # Assign random priorities but weight larger gaps
            candidates.sort(key=lambda x: random.random() ** 0.5 * x)
        else:
            # Start with well-spaced numbers, then fill in
            random.seed(trial * 17 + 5)
            # Create a base of evenly spaced numbers, then add random
            step = n_max // 100
            base = list(range(step, n_max + 1, step))
            random.shuffle(base)
            remaining = [x for x in range(1, n_max + 1) if x not in set(base)]
            random.shuffle(remaining)
            candidates = base + remaining
        
        result = greedy_build(candidates, target_size)
        
        if len(result) > len(best_set):
            best_set = result
        
        if len(best_set) == target_size:
            break
    
    return best_set
# EVOLVE-BLOCK-END
