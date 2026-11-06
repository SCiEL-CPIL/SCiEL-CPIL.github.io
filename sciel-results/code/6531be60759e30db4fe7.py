"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill

    N = 128
    # Precompute GCD table
    gcd_table = [[0]*(N+1) for _ in range(N+1)]
    for i in range(1, N+1):
        for j in range(i+1, N+1):
            gcd_table[i][j] = gcd(i, j)
            gcd_table[j][i] = gcd_table[i][j]
    
    # Build conflict structure using bitmasks
    # conflict[i] = bitmask of all j such that (i, j) participates in some bad triple
    # More precisely: for each pair (i,j), find all k where gcd(i,j)==gcd(i,k)==gcd(j,k)
    # conflict_pair[i][j] = bitmask of k values that form a bad triple with (i,j)
    # For efficiency: forbidden[i] = bitmask of j such that adding i when j is present
    #   would require checking all pairs... 
    
    # Better approach: 
    # conflict[i] = bitmask of j where there exists some k already in set such that
    # (i,j,k) is a bad triple. But this depends on the set.
    # 
    # Alternative: Precompute for each pair (i,j) the set of k that conflict.
    # Then when building a set S, adding x is valid iff for all (a,b) in S,
    # x is not in conflict_pair[a][b].
    
    # Precompute: for each pair (i,j), bitmask of k that form bad triple
    conflict_pair = [[0]*(N+1) for _ in range(N+1)]
    for i in range(1, N+1):
        for j in range(i+1, N+1):
            g = gcd_table[i][j]
            mask = 0
            for k in range(1, N+1):
                if k != i and k != j and gcd_table[i][k] == g and gcd_table[j][k] == g:
                    mask |= (1 << k)
            conflict_pair[i][j] = mask
            conflict_pair[j][i] = mask
    
    # For a set S (as bitmask), can we add x?
    # x can be added iff for all pairs (a,b) in S, x is not in conflict_pair[a][b]
    # Equivalently: the union of conflict_pair[a][b] over all pairs (a,b) in S does not contain x
    
    # Precompute: for each x, precompute which pairs (a,b) would conflict with x
    # pair_conflicts_with_x[x] = list of (a,b) pairs where x is in conflict_pair[a][b]
    # But that's too large. Instead:
    # For a given set S (bitmask), to check if x can be added:
    #   For each a in S, for each b in S with b>a, check if x in conflict_pair[a][b]
    # This is O(|S|^2) per check. With |S| up to ~60, that's ~3600 checks per element.
    
    # Faster: precompute for each x a bitmask conflict_with_x[x] = 
    #   bitmask over pairs... no, let's think differently.
    # 
    # For each x, define: bad_pairs_for_x = set of (a,b) such that x is in conflict_pair[a][b]
    # Then x cannot be added to S if S contains both a and b for any (a,b) in bad_pairs_for_x
    # Equivalently: x cannot be added if (S & pair_mask_a) and (S & pair_mask_b) for some pair
    # 
    # Even faster: For each x, precompute a list of (a, b) pairs.
    # Then to check if x can be added to S: for each (a,b) in bad_pairs_for_x, 
    # check if both a and b are in S.
    
    bad_pairs_for_x = [[] for _ in range(N+1)]
    for i in range(1, N+1):
        for j in range(i+1, N+1):
            mask = conflict_pair[i][j]
            # For each k in mask, add (i,j) to bad_pairs_for_x[k]
            k = 1
            while k <= N:
                if mask & (1 << k):
                    bad_pairs_for_x[k].append((i, j))
                k += 1
    
    def can_add_bitmask(x, S_mask):
        """Check if x can be added to set represented by bitmask S_mask."""
        for (a, b) in bad_pairs_for_x[x]:
            if (S_mask & (1 << a)) and (S_mask & (1 << b)):
                return False
        return True
    
    def greedy_remove():
        """Start with all elements, greedily remove elements that cause conflicts."""
        S = (1 << (N+1)) - 2  # bits 1..N set
        # Remove elements in order of "damage" - how many conflicts they cause
        remaining = list(range(1, N+1))
        
        while remaining:
            # Find the element that participates in the most bad triples with other remaining elements
            worst = -1
            worst_count = 0
            for x in remaining:
                count = 0
                for (a, b) in bad_pairs_for_x[x]:
                    if (S & (1 << a)) and (S & (1 << b)):
                        count += 1
                if count > worst_count:
                    worst_count = count
                    worst = x
            
            if worst == -1 or worst_count == 0:
                break
            
            S &= ~(1 << worst)
            remaining.remove(worst)
        
        return S
    
    def greedy_add(order):
        """Build set by adding elements in given order."""
        S = 0
        for x in order:
            if can_add_bitmask(x, S):
                S |= (1 << x)
        return S
    
    def local_search_improve(S_mask, time_limit):
        """Try to improve by adding elements and swapping."""
        S = S_mask
        
        while time.time() < time_limit:
            improved = False
            
            # Try adding elements
            for x in range(1, N+1):
                if S & (1 << x):
                    continue
                if can_add_bitmask(x, S):
                    S |= (1 << x)
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try 1-for-2 swap: remove one, add two
            for i in range(1, N+1):
                if not (S & (1 << i)):
                    continue
                S_temp = S & ~(1 << i)
                added = 0
                for x in range(1, N+1):
                    if (S_temp & (1 << x)) or x == i:
                        continue
                    if can_add_bitmask(x, S_temp):
                        S_temp |= (1 << x)
                        added += 1
                        if added >= 2:
                            break
                if added > 1:
                    S = S_temp
                    improved = True
                    break
            
            if not improved:
                break
        
        return S
    
    # Phase 1: Multiple greedy approaches
    best_S = 0
    
    # Greedy removal
    S1 = greedy_remove()
    if bin(S1).count('1') > bin(best_S).count('1'):
        best_S = S1
    
    if time.time() > deadline - 20:
        result = [i for i in range(1, N+1) if best_S & (1 << i)]
        return result if result else [1]
    
    # Greedy add: ascending
    S2 = greedy_add(range(1, N+1))
    if bin(S2).count('1') > bin(best_S).count('1'):
        best_S = S2
    
    # Greedy add: descending
    S3 = greedy_add(range(N, 0, -1))
    if bin(S3).count('1') > bin(best_S).count('1'):
        best_S = S3
    
    # Greedy add: by number of conflicts (fewer conflicts first)
    def conflict_count(x):
        return len(bad_pairs_for_x[x])
    
    order_conflicts = sorted(range(1, N+1), key=lambda x: (conflict_count(x), x))
    S4 = greedy_add(order_conflicts)
    if bin(S4).count('1') > bin(best_S).count('1'):
        best_S = S4
    
    # Random orderings
    rng = random.Random(42)
    for trial in range(30):
        if time.time() > deadline - 10:
            break
        order = list(range(1, N+1))
        rng.shuffle(order)
        S = greedy_add(order)
        if bin(S).count('1') > bin(best_S).count('1'):
            best_S = S
    
    # Local search on best
    best_S = local_search_improve(best_S, deadline - 2)
    
    # Additional local search restarts
    for trial in range(10):
        if time.time() > deadline - 2:
            break
        # Perturb: remove a random element and re-add
        S_temp = best_S
        elements = [i for i in range(1, N+1) if S_temp & (1 << i)]
        if elements:
            remove_idx = rng.randint(0, len(elements)-1)
            S_temp &= ~(1 << elements[remove_idx])
            S_temp = local_search_improve(S_temp, min(time.time() + 1, deadline - 2))
            if bin(S_temp).count('1') > bin(best_S).count('1'):
                best_S = S_temp
    
    result = [i for i in range(1, N+1) if best_S & (1 << i)]
    return result if result else [1]
# EVOLVE-BLOCK-END
