import time
import random
from math import gcd

def construct():
    """Return a nonempty subset of [1,255] containing 255, with gcd(a,b)>1 for all distinct pairs. Maximize cardinality."""
    deadline = time.time() + 85  # Safety margin
    
    # Precompute gcd matrix for speed
    N = 255
    # gcd_matrix[i][j] is gcd(i, j)
    # Using 1-based indexing for simplicity in loops, but 0 is unused
    gcd_mat = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = gcd(i, j)
            gcd_mat[i][j] = g
            gcd_mat[j][i] = g

    def is_valid_subset(S):
        """Check if all pairs in S have gcd > 1."""
        S_list = list(S)
        for i in range(len(S_list)):
            for j in range(i + 1, len(S_list)):
                if gcd_mat[S_list[i]][S_list[j]] == 1:
                    return False
        return True

    def get_conflicts(x, S):
        """Return set of elements in S that are coprime to x."""
        return {y for y in S if gcd_mat[x][y] == 1}

    # Initial set: multiples of 3
    current_set = set(x for x in range(1, N + 1) if x % 3 == 0)
    if 255 not in current_set:
        current_set.add(255)
    
    best_set = set(current_set)
    best_size = len(best_set)
    
    # Local Search / Simulated Annealing
    # We want to maximize size.
    # Moves:
    # 1. Add x if no conflicts.
    # 2. Swap y (conflict) for x.
    # 3. Remove y (to enable future additions? No, just size).
    
    # To improve efficiency, we can precompute conflicts for all pairs?
    # gcd_mat is already there.
    
    # Simple Hill Climbing with Random Restarts
    max_iterations = 50000
    
    for restart in range(10):
        if time.time() > deadline:
            break
            
        # Initialize
        if restart == 0:
            S = set(best_set)
        else:
            # Perturb best_set
            S = set(best_set)
            # Remove a few random elements
            for _ in range(5):
                if S:
                    S.remove(random.choice(list(S)))
            # Add 255 back if removed
            if 255 not in S:
                S.add(255)
        
        current_size = len(S)
        
        # Local search
        for _ in range(max_iterations):
            if time.time() > deadline:
                break
                
            # Try to add a random element not in S
            candidates = [x for x in range(1, N + 1) if x not in S]
            if not candidates:
                break
            
            x = random.choice(candidates)
            conflicts = get_conflicts(x, S)
            
            if not conflicts:
                # Can add directly
                S.add(x)
                current_size += 1
                if current_size > best_size:
                    best_size = current_size
                    best_set = set(S)
            else:
                # Try to swap: remove one conflict, add x
                # This keeps size same, but might enable better configuration
                # Only do this if we are "stuck" or randomly
                if random.random() < 0.1:
                    y = random.choice(list(conflicts))
                    S.remove(y)
                    S.add(x)
                    # Size unchanged
                    
        # After local search, try to greedily add more
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for x in range(1, N + 1):
                if x in S:
                    continue
                if not get_conflicts(x, S):
                    S.add(x)
                    current_size += 1
                    changed = True
                    if current_size > best_size:
                        best_size = current_size
                        best_set = set(S)
                    
    # Final check
    if 255 not in best_set:
        best_set.add(255)
        
    return sorted(best_set)
