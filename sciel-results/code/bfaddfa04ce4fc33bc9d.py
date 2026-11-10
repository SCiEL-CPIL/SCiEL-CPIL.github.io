"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5 second safety margin
    
    n = 128
    all_nums = list(range(1, n + 1))
    
    # Precompute: for each pair (i,j) with i<j, check if i+j is in range
    # We need to avoid: three distinct a,b,c in S such that a+b, a+c, b+c all in S
    
    # Start with the seed solution: all numbers > 64
    S = set(range(65, n + 1))
    best_S = set(S)
    best_size = len(S)
    
    # Try to add elements one by one, checking validity
    # An element x can be added if it doesn't create a forbidden triple with existing elements
    
    def is_valid_addition(S, x):
        """Check if adding x to S keeps S valid."""
        # For every pair (a,b) in S with a != b and a != x and b != x,
        # check if x+a, x+b, a+b are all in S ∪ {x}
        # Actually, we need to check: for every pair (a,b) in S (a≠b),
        # if x+a in S∪{x} and x+b in S∪{x} and a+b in S∪{x}, then forbidden.
        
        # More precisely: after adding x, for every triple of distinct elements
        # from S∪{x}, check the condition.
        # New triples involving x: (x, a, b) for a,b in S, a≠b
        # For such a triple, check if x+a, x+b, a+b are all in S∪{x}
        
        for a in S:
            if a == x:
                continue
            for b in S:
                if b == x or b == a:
                    continue
                sa = x + a
                sb = x + b
                ab = a + b
                # Check if all three sums are in S ∪ {x}
                if (sa in S or sa == x) and (sb in S or sb == x) and (ab in S or ab == x):
                    return False
        return True
    
    # Greedy: try to add each element from 1 to 64 in some order
    # Try multiple random orders to find a better solution
    best_size = len(best_S)
    
    for attempt in range(200):
        if time.time() > deadline:
            break
        
        random.seed(attempt)
        S = set(range(65, n + 1))
        
        # Try adding elements from 1 to 64 in random order
        candidates = list(range(1, 65))
        random.shuffle(candidates)
        
        for x in candidates:
            if time.time() > deadline:
                break
            if is_valid_addition(S, x):
                S.add(x)
        
        if len(S) > best_size:
            best_size = len(S)
            best_S = set(S)
    
    # Local search: try to swap elements to improve
    for iteration in range(500):
        if time.time() > deadline:
            break
        
        improved = False
        # Try removing one element and adding another
        S_list = list(best_S)
        random.seed(iteration)
        random.shuffle(S_list)
        
        for remove_idx in range(len(S_list)):
            if time.time() > deadline:
                break
            x_remove = S_list[remove_idx]
            S_temp = set(best_S)
            S_temp.discard(x_remove)
            
            # Try adding each element not in S_temp
            for x_add in range(1, n + 1):
                if x_add in S_temp:
                    continue
                if is_valid_addition(S_temp, x_add):
                    S_temp.add(x_add)
                    if len(S_temp) > len(best_S):
                        best_S = S_temp
                        best_size = len(best_S)
                        improved = True
                        break
            if improved:
                break
        if not improved and iteration > 50:
            break
    
    # Ensure we return Python ints in a list
    result = sorted([int(x) for x in best_S])
    return result
# EVOLVE-BLOCK-END
