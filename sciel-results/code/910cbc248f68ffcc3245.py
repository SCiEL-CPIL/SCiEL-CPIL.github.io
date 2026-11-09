"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from random import Random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    n = 128
    
    # Precompute gcd table for [1,128]
    gcd_table = [[0]*(n+1) for _ in range(n+1)]
    for i in range(1, n+1):
        for j in range(i, n+1):
            g = gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g
    
    # Precompute bad triples: (a,b,c) where a<b<c and gcd(a,b)==gcd(a,c)==gcd(b,c)
    bad_triples = set()
    for a in range(1, n+1):
        for b in range(a+1, n+1):
            g_ab = gcd_table[a][b]
            for c in range(b+1, n+1):
                if gcd_table[a][c] == g_ab and gcd_table[b][c] == g_ab:
                    bad_triples.add((a, b, c))
    
    def valid_addition(selected_set, x):
        """Check if adding x to selected_set maintains the no-equal-gcd-triple property."""
        sel = list(selected_set)
        for i in range(len(sel)):
            a = sel[i]
            for j in range(i+1, len(sel)):
                b = sel[j]
                # Create sorted triple
                triple = tuple(sorted((a, b, x)))
                if triple in bad_triples:
                    return False
        return True
    
    def local_search(selected_set, rng):
        """Try to improve the set by additions and swaps."""
        selected_list = list(selected_set)
        best_list = selected_list[:]
        best_size = len(best_list)
        
        # Repeatedly try to add elements until no more can be added
        while True:
            remaining = [x for x in range(1, n+1) if x not in set(best_list)]
            rng.shuffle(remaining)
            added = False
            for x in remaining:
                if valid_addition(set(best_list), x):
                    best_list.append(x)
                    best_size = len(best_list)
                    added = True
                    break
            if not added:
                break
        
        # Try swaps: remove one element, then try to add multiple others
        improved = True
        while improved:
            improved = False
            best_list_set = set(best_list)
            remaining = [x for x in range(1, n+1) if x not in best_list_set]
            rng.shuffle(remaining)
            
            for u in list(best_list_set):
                if time.time() > deadline - 3:
                    return best_list
                
                trial = [x for x in best_list if x != u]
                trial_set = set(trial)
                
                # Try to add elements from remaining (excluding u)
                addable = [x for x in remaining if x != u]
                rng.shuffle(addable)
                
                new_list = trial[:]
                new_size = len(new_list)
                
                for x in addable:
                    if valid_addition(set(new_list), x):
                        new_list.append(x)
                        new_size = len(new_list)
                
                if new_size > best_size:
                    best_list = new_list
                    best_size = new_size
                    improved = True
                    break
        
        return best_list
    
    best_set = set()
    
    # Multiple random restarts with greedy construction
    for restart in range(500):
        if time.time() > deadline - 5:
            break
        
        rng = Random(restart * 7919 + 13)
        
        # Greedy construction with random ordering
        candidates = list(range(1, n+1))
        rng.shuffle(candidates)
        
        current = set()
        for x in candidates:
            if valid_addition(current, x):
                current.add(x)
        
        # Local search to improve
        improved = local_search(current, rng)
        
        if len(improved) > len(best_set):
            best_set = set(improved)
            if time.time() > deadline - 10:
                break
    
    # Final local search on best set if time permits
    if time.time() < deadline - 5:
        rng = Random(42)
        final = local_search(best_set, rng)
        if len(final) > len(best_set):
            best_set = set(final)
    
    # Verify the result is valid
    result = sorted(best_set)
    m = len(result)
    for i in range(m):
        a = result[i]
        for j in range(i+1, m):
            b = result[j]
            gab = gcd_table[a][b]
            for k in range(j+1, m):
                c = result[k]
                if gcd_table[a][c] == gab and gcd_table[b][c] == gab:
                    # Invalid - should not happen, but safeguard
                    pass
    
    if not result:
        return [1]
    
    return result
# EVOLVE-BLOCK-END
