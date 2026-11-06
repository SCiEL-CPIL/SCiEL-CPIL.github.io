import random
import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    N = 128
    FULL_MASK = (1 << N) - 1
    
    # Precompute GCD table
    gcd_table = [[0]*(N+1) for _ in range(N+1)]
    for i in range(1, N+1):
        for j in range(i, N+1):
            g = gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g
    
    # Precompute bad triples and conflict structures
    # bad_pairs_with[x] is a list of bitmasks (1<<a | 1<<b) 
    # such that (x, a, b) is a bad triple.
    bad_pairs_with = [[] for _ in range(N+1)]
    
    for a in range(1, N+1):
        for b in range(a+1, N+1):
            g_ab = gcd_table[a][b]
            for c in range(b+1, N+1):
                if gcd_table[b][c] == g_ab and gcd_table[c][a] == g_ab:
                    # Triple (a, b, c) is bad
                    mask_ab = (1 << a) | (1 << b)
                    mask_bc = (1 << b) | (1 << c)
                    mask_ca = (1 << c) | (1 << a)
                    
                    bad_pairs_with[a].append(mask_bc)
                    bad_pairs_with[b].append(mask_ca)
                    bad_pairs_with[c].append(mask_ab)

    def can_add(S, x):
        """Check if x can be added to S without creating a bad triple."""
        for m in bad_pairs_with[x]:
            if (S & m) == m:
                return False
        return True

    def popcount(mask):
        return bin(mask).count('1')

    def greedy_repair(S):
        """
        Greedily add elements to S to maximize size while maintaining validity.
        Returns the new set.
        """
        current_S = S
        # We will try to add elements in a specific order.
        # To be fast, we can just iterate through all candidates repeatedly.
        # But that's slow.
        
        # Better: Maintain a list of candidates.
        # Since N is small (128), we can just scan.
        
        changed = True
        while changed and time.time() < deadline - 1.0:
            changed = False
            # Find the best candidate to add.
            # Heuristic: Pick the one with the fewest existing conflicts (0) 
            # and perhaps the smallest number of total bad pairs (less restrictive).
            best_x = -1
            min_conflicts = 1
            min_degree = 999
            
            for x in range(1, N+1):
                if current_S & (1 << x):
                    continue
                
                # Check if x can be added
                # We need to count conflicts to see if it's 0.
                # If > 0, it can't be added.
                # If 0, it's a candidate.
                
                # Optimization: Just check can_add.
                # To rank candidates, we can look at len(bad_pairs_with[x])
                # but that's static.
                
                if can_add(current_S, x):
                    # It's a candidate.
                    # To break ties, prefer x with smaller len(bad_pairs_with[x])?
                    # Or just randomize to avoid bias.
                    # Let's pick randomly from all addable ones to explore.
                    # But picking the "least constrained" might lead to larger sets.
                    deg = len(bad_pairs_with[x])
                    if deg < min_degree:
                        min_degree = deg
                        best_x = x
                    # If we want randomness:
                    # if random.random() < 0.1: best_x = x # No, let's stick to heuristic
                    
            if best_x != -1:
                current_S |= (1 << best_x)
                changed = True
                
        return current_S

    # Initial Solution
    # Start with empty set and repair
    initial_S = 0
    best_S = greedy_repair(initial_S)
    best_size = popcount(best_S)
    current_S = best_S
    current_size = best_size
    
    # Iterated Local Search
    # Perturbation: Remove k random elements
    # Repair: Greedy add
    
    k_remove = 1
    no_improve_count = 0
    
    while time.time() < deadline - 5.0:
        # Perturb
        perturbed_S = current_S
        active_elements = [i for i in range(1, N+1) if perturbed_S & (1 << i)]
        
        # Remove k random elements
        k = min(k_remove, len(active_elements))
        if k > 0:
            to_remove = random.sample(active_elements, k)
            for x in to_remove:
                perturbed_S &= ~(1 << x)
        
        # Repair
        repaired_S = greedy_repair(perturbed_S)
        repaired_size = popcount(repaired_S)
        
        # Acceptance
        if repaired_size > best_size:
            best_S = repaired_S
            best_size = repaired_size
            current_S = repaired_S
            current_size = repaired_size
            no_improve_count = 0
            # Maybe decrease k_remove if we found a new best?
            # Or keep it.
        elif repaired_size >= current_size:
            # Accept if it's not worse than current, to allow wandering
            # But if it's strictly worse, we might reject.
            # Standard ILS accepts if better than best, or with probability.
            # Let's accept if it's equal or better than current.
            current_S = repaired_S
            current_size = repaired_size
            no_improve_count += 1
        else:
            no_improve_count += 1
            
        # Adapt k_remove
        if no_improve_count > 20:
            # Increase perturbation strength
            k_remove = min(k_remove + 1, 5)
            no_improve_count = 0
        elif no_improve_count > 50:
            # Reset to small k
            k_remove = 1
            no_improve_count = 0

    # Final check and return
    result = [i for i in range(1, N+1) if best_S & (1 << i)]
    if not result:
        result = [1]
    return result
