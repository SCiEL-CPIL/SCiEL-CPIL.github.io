"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from random import Random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    # Precompute gcd table for [1,128]
    n = 128
    gcd_table = [[0]*(n+1) for _ in range(n+1)]
    for i in range(1, n+1):
        for j in range(i, n+1):
            g = gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g
    
    def valid_addition(selected_set, x):
        """Check if adding x to selected_set maintains the no-equal-gcd-triple property."""
        sel = list(selected_set)
        for i in range(len(sel)):
            a = sel[i]
            for j in range(i+1, len(sel)):
                b = sel[j]
                if gcd_table[a][b] == gcd_table[a][x] == gcd_table[b][x]:
                    return False
        return True
    
    def count_violations(selected_list):
        """Count number of violating triples."""
        cnt = 0
        m = len(selected_list)
        for i in range(m):
            a = selected_list[i]
            for j in range(i+1, m):
                b = selected_list[j]
                gab = gcd_table[a][b]
                for k in range(j+1, m):
                    c = selected_list[k]
                    if gcd_table[a][c] == gab and gcd_table[b][c] == gab:
                        cnt += 1
        return cnt
    
    def local_search(selected_set, rng):
        """Try to improve the set by swaps and additions."""
        selected_list = list(selected_set)
        best_list = selected_list[:]
        best_size = len(best_list)
        remaining = [x for x in range(1, n+1) if x not in selected_set]
        
        # Try adding elements
        for x in remaining:
            if valid_addition(selected_list, x):
                new_list = selected_list + [x]
                if len(new_list) > best_size:
                    best_list = new_list
                    best_size = len(best_list)
        
        # Try swaps: remove one, add another
        for i in range(len(best_list)):
            orig = best_list[i]
            trial = best_list[:i] + best_list[i+1:]
            for x in remaining:
                if x in set(trial):
                    continue
                if valid_addition(trial, x):
                    new_list = trial + [x]
                    if len(new_list) > best_size:
                        best_list = new_list
                        best_size = len(best_list)
        
        return best_list
    
    best_set = set()
    
    # Multiple random restarts with greedy construction
    for restart in range(200):
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
    result = list(best_set)
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
    
    return sorted(result)
# EVOLVE-BLOCK-END
